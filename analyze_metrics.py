#coding: utf-8
"""
Análise de métricas do CES2 cruzando eventos + série temporal do cluster.

Diferentemente do legado analyze_events.py (que só lia o JSON e produzia
métricas fictícias de antecipação/atraso), este módulo cruza:
  - events_{model}_{ts}.json        (ações de wake/shutdown/sla/initial/final)
  - cluster_workload_{model}_{ts}.csv  (série temporal real: ram_avg, predicted_ram)

para calcular, de forma real, quando a carga cruza os limiares:
antecipação, atraso, ativações desnecessárias, tempo ativo, SLA e economia.

Uso:
  python analyze_metrics.py \
      --reactive-events events_default_<ts1>.json --reactive-csv cluster_workload_default_<ts1>.csv \
      --lstm-events events_lstm_<ts2>.json        --lstm-csv cluster_workload_lstm_<ts2>.csv

Saídas:
  - analysis_<ts1>__vs__<ts2>.json  (máquina)
  - TCC_METRICS_<ts>.md             (humano)
"""

import argparse
import json
import re
from datetime import timedelta

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_events(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def load_cluster_csv(filename):
    """Load cluster_workload CSV into a DataFrame indexed by time_stamp."""
    df = pd.read_csv(filename)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values('time_stamp').set_index('time_stamp')
    return df


def ts(x):
    """Coerce to pandas Timestamp."""
    return pd.Timestamp(x)


def csv_duration_hours(csv_df):
    if csv_df.empty:
        return 0.0
    return (csv_df.index.max() - csv_df.index.min()).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Window alignment
# ---------------------------------------------------------------------------

def align_windows(events_r, csv_r, events_l, csv_l):
    """
    Truncate both experiments to the shorter duration T_min so they are
    comparable. Each experiment is truncated from its own start.
    Returns dict with duration_hours and per-experiment start timestamps.
    """
    dur_r = csv_duration_hours(csv_r)
    dur_l = csv_duration_hours(csv_l)
    t_min = min(dur_r, dur_l)
    start_r = csv_r.index.min() if not csv_r.empty else None
    start_l = csv_l.index.min() if not csv_l.empty else None
    return {
        'duration_hours': t_min,
        'start_reactive': str(start_r) if start_r is not None else None,
        'start_lstm': str(start_l) if start_l is not None else None,
    }


def _truncate_csv(csv_df, duration_h):
    if csv_df.empty or duration_h <= 0:
        return csv_df
    start = csv_df.index.min()
    end = start + timedelta(hours=duration_h)
    return csv_df[(csv_df.index >= start) & (csv_df.index <= end)]


def _events_in_window(events, duration_h, csv_df):
    """Restrict events to [csv_start, csv_start + duration_h]."""
    if not events or duration_h <= 0 or csv_df.empty:
        return events
    start = csv_df.index.min()
    end = start + timedelta(hours=duration_h)
    out = []
    for e in events:
        try:
            t = ts(e['timestamp'])
        except Exception:
            continue
        if start <= t <= end:
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# Crossing detection on the ram_avg time series
# ---------------------------------------------------------------------------

def first_ascending_crossing(series, threshold, after_ts):
    """First index where ram_avg crosses from < threshold to >= threshold, after after_ts."""
    s = series[series.index >= after_ts].dropna()
    if len(s) < 2:
        return None
    prev_below = (s.iloc[:-1] < threshold).values
    curr_above = (s.iloc[1:] >= threshold).values
    cross = np.where(prev_below & curr_above)[0]
    if len(cross) == 0:
        return None
    return s.index[cross[0] + 1]


def last_descending_crossing(series, threshold, before_ts):
    """Last index where ram_avg crosses from >= threshold to < threshold, before before_ts."""
    s = series[series.index <= before_ts].dropna()
    if len(s) < 2:
        return None
    prev_above = (s.iloc[:-1] >= threshold).values
    curr_below = (s.iloc[1:] < threshold).values
    cross = np.where(prev_above & curr_below)[0]
    if len(cross) == 0:
        return None
    return s.index[cross[-1] + 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def unnecessary_activations(events, csv_df, lim_max, duration_h):
    """
    % of wake events whose ram_avg did NOT exceed lim_max within
    VALIDATION_WINDOW_MIN minutes after the wake (= wake not justified by load).

    Note: a wake adds a fresh (low-RAM) host, which dilutes the mean and may
    mask a genuinely necessary wake. Interpret with that caveat.
    """
    events = _events_in_window(events, duration_h, csv_df)
    wakes = [e for e in events if e['event_type'] == 'wake']
    if not wakes:
        return 0.0
    ram = csv_df['ram_avg']
    window = timedelta(minutes=config.VALIDATION_WINDOW_MIN)
    unnecessary = 0
    for w in wakes:
        try:
            wts = ts(w['timestamp'])
        except Exception:
            continue
        seg = ram[(ram.index >= wts) & (ram.index <= wts + window)]
        seg = pd.to_numeric(seg, errors='coerce').dropna()
        peaked = (seg.max() > lim_max) if not seg.empty else False
        if not peaked:
            unnecessary += 1
    return (unnecessary / len(wakes)) * 100.0


def late_shutdown_time(events, csv_df, lim_med, duration_h):
    """
    Mean (shutdown_ts - last descending crossing of lim_med) in minutes.
    Measures how long after load dropped below lim_med the host was shut down.
    """
    events = _events_in_window(events, duration_h, csv_df)
    shutdowns = [e for e in events if e['event_type'] == 'shutdown']
    if not shutdowns:
        return 0.0
    ram = pd.to_numeric(csv_df['ram_avg'], errors='coerce')
    lates = []
    for s in shutdowns:
        try:
            sts = ts(s['timestamp'])
        except Exception:
            continue
        cross = last_descending_crossing(ram, lim_med, sts)
        if cross is not None:
            lates.append((sts - cross).total_seconds() / 60.0)
    return (sum(lates) / len(lates)) if lates else 0.0


def anticipation_time(events, csv_df, lim_max, duration_h):
    """
    Mean (first ascending crossing of lim_max after wake - wake_ts) in minutes,
    for lstm_predictive wakes only. Measures how early the wake anticipated
    the real threshold crossing. Returns 0.0 if no predictive wakes.
    """
    events = _events_in_window(events, duration_h, csv_df)
    wakes = [e for e in events if e.get('event_type') == 'wake'
             and e.get('trigger_type') == 'lstm_predictive']
    if not wakes:
        return 0.0
    ram = pd.to_numeric(csv_df['ram_avg'], errors='coerce')
    anticipations = []
    for w in wakes:
        try:
            wts = ts(w['timestamp'])
        except Exception:
            continue
        cross = first_ascending_crossing(ram, lim_max, wts)
        if cross is not None:
            anticipations.append((cross - wts).total_seconds() / 60.0)
    return (sum(anticipations) / len(anticipations)) if anticipations else 0.0


def _clip(a, b, start, end):
    return max(0.0, (min(b, end) - max(a, start)).total_seconds())


def total_active_hours_aligned(events, duration_h):
    """
    Sum of host-ON hours within the aligned window, reconstructed from
    initial_state / wake / shutdown / final_state events. Consecutive wakes
    without an intervening shutdown are de-duplicated.
    """
    if not events or duration_h <= 0:
        return 0.0
    inits = [e for e in events if e['event_type'] == 'initial_state']
    start = min(ts(e['timestamp']) for e in inits) if inits \
        else min(ts(e['timestamp']) for e in events)
    end = start + timedelta(hours=duration_h)

    by_host = {}
    for e in events:
        h = e.get('hostname')
        if not h:
            continue
        by_host.setdefault(h, []).append(e)

    total_seconds = 0.0
    for h, evs in by_host.items():
        evs.sort(key=lambda e: ts(e['timestamp']))
        on = False
        on_since = None
        for e in evs:
            t = ts(e['timestamp'])
            et = e['event_type']
            if et == 'initial_state':
                if e.get('initial_state') == 'up':
                    on, on_since = True, start
                else:
                    on = False
            elif et == 'wake':
                if not on:
                    on, on_since = True, t
            elif et == 'shutdown':
                if on:
                    total_seconds += _clip(on_since, t, start, end)
                    on = False
        if on:
            total_seconds += _clip(on_since, end, start, end)
    return total_seconds / 3600.0


def sla_episodes(events, csv_df, duration_h):
    """
    Count SLA episodes: sla_violation events (already per-host, transition-based
    via verifier.sla_violating_hosts). trigger_type indicates the reason:
    'ram_over_threshold' or 'host_inaccessible'.
    """
    events = _events_in_window(events, duration_h, csv_df)
    return len([e for e in events
                if e['event_type'] == 'sla_violation' and e.get('hostname') is not None])


def energy_economy(hours_r, hours_l, p):
    """
    Energy saved by LSTM vs reactive, using average host power.
      reactive_kwh = hours_r * P_AVG / 1000
      lstm_kwh     = hours_l  * P_AVG / 1000
    """
    reactive_kwh = hours_r * p.P_AVG / 1000.0
    lstm_kwh = hours_l * p.P_AVG / 1000.0
    saved = reactive_kwh - lstm_kwh
    pct = (saved / reactive_kwh * 100.0) if reactive_kwh > 0 else 0.0
    return {'kwh': round(saved, 4), 'pct': round(pct, 2)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_scenario(events, csv_df, lim_max, lim_med, duration_h):
    return {
        'unnecessary_pct': round(unnecessary_activations(events, csv_df, lim_max, duration_h), 2),
        'late_shutdown_min': round(late_shutdown_time(events, csv_df, lim_med, duration_h), 2),
        'anticipation_min': round(anticipation_time(events, csv_df, lim_max, duration_h), 2),
        'active_hours': round(total_active_hours_aligned(events, duration_h), 3),
        'sla_episodes': sla_episodes(events, csv_df, duration_h),
    }


def extract_ts(filename, prefix=None):
    """Extract the {ts} token (YYYYMMDD_HHMMSS) from a filename like
    events_default_20260709_233217.json or cluster_workload_lstm_20260709_235022.csv."""
    m = re.search(r'_(\d{8}_\d{6})(?=\.|$)', filename)
    return m.group(1) if m else 'unknown'


def build_report(window, r, l, lim_max, lim_med):
    lines = []
    lines.append('# Relatório de Métricas — CES2 (Reativo vs LSTM)\n')
    lines.append(f"- Janela alinhada: **{window['duration_hours']:.2f} h** "
                 f"(início reativo: {window['start_reactive']} | início LSTM: {window['start_lstm']})")
    lines.append(f"- Limiares: lim_max={lim_max}% | lim_med={lim_med}%")
    lines.append(f"- Potência: P_IDLE={config.P_IDLE}W | P_LOAD={config.P_LOAD}W | P_AVG={config.P_AVG}W\n")

    lines.append('| Métrica | Reativo (default) | LSTM |')
    lines.append('|---|---|---|')
    lines.append(f"| Ativações desnecessárias (%) | {r['unnecessary_pct']:.1f} | {l['unnecessary_pct']:.1f} |")
    lines.append(f"| Atraso de shutdown (min) | {r['late_shutdown_min']:.2f} | {l['late_shutdown_min']:.2f} |")
    lines.append(f"| Antecipação (min) | {r['anticipation_min']:.2f} | {l['anticipation_min']:.2f} |")
    lines.append(f"| Horas ativas (host·h) | {r['active_hours']:.2f} | {l['active_hours']:.2f} |")
    lines.append(f"| Episódios SLA | {r['sla_episodes']} | {l['sla_episodes']} |")
    lines.append('')

    reduction = (((r['active_hours'] - l['active_hours']) / r['active_hours'] * 100.0)
                 if r['active_hours'] > 0 else 0.0)
    energy = energy_economy(r['active_hours'], l['active_hours'], config)
    lines.append('## Economia de energia')
    lines.append(f"- Redução de horas ativas: **{reduction:.1f}%**")
    lines.append(f"- Energia economizada: **{energy['kwh']} kWh** ({energy['pct']:.1f}%)")
    lines.append('')
    lines.append('## Interpretação')
    lines.append('- **Antecipação (LSTM)**: média de quão cedo o wake preditivo ocorreu antes '
                 'do cruzamento real de lim_max. Valor > 0 indica que o LSTM acordou hosts com antecedência.')
    lines.append('- **Atraso de shutdown**: quanto tempo após a carga cair abaixo de lim_med o host foi desligado. '
                 'Menor = melhor consolidação.')
    lines.append('- **Ativações desnecessárias**: % de wakes sem pico de carga acima de lim_max na janela de '
                 f'{config.VALIDATION_WINDOW_MIN} min. *Caveat:* um wake adiciona um host novo (RAM baixa) que '
                 'dilui a média, podendo mascarar wakes necessários.')
    lines.append('- **Episódios SLA evitáveis**: violações com hosts offline disponíveis (offline>0); '
                 'limite de capacidade (offline==0) não conta contra o modelo.')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Análise de métricas CES2 (reativo vs LSTM)')
    parser.add_argument('--reactive-events', required=True)
    parser.add_argument('--reactive-csv', required=True)
    parser.add_argument('--lstm-events', required=True)
    parser.add_argument('--lstm-csv', required=True)
    parser.add_argument('--lim-max', type=float, default=config.LIM_MAX)
    parser.add_argument('--lim-med', type=float, default=config.LIM_MED)
    args = parser.parse_args()

    events_r = load_events(args.reactive_events)
    csv_r = load_cluster_csv(args.reactive_csv)
    events_l = load_events(args.lstm_events)
    csv_l = load_cluster_csv(args.lstm_csv)

    window = align_windows(events_r, csv_r, events_l, csv_l)
    dur = window['duration_hours']

    r = analyze_scenario(events_r, csv_r, args.lim_max, args.lim_med, dur)
    l = analyze_scenario(events_l, csv_l, args.lim_max, args.lim_med, dur)

    energy = energy_economy(r['active_hours'], l['active_hours'], config)
    reduction = (((r['active_hours'] - l['active_hours']) / r['active_hours'] * 100.0)
                 if r['active_hours'] > 0 else 0.0)

    r_ts = extract_ts(args.reactive_events, 'events')
    l_ts = extract_ts(args.lstm_events, 'events')

    result = {
        'window': {'start': window['start_reactive'], 'duration_hours': round(dur, 3)},
        'unnecessary_reactive': r['unnecessary_pct'],
        'unnecessary_lstm': l['unnecessary_pct'],
        'late_shutdown_reactive_min': r['late_shutdown_min'],
        'late_shutdown_lstm_min': l['late_shutdown_min'],
        'anticipation_reactive_min': r['anticipation_min'],
        'anticipation_lstm_min': l['anticipation_min'],
        'baseline_hours': r['active_hours'],
        'lstm_active_hours': l['active_hours'],
        'reduction_pct': round(reduction, 2),
        'energy_kwh': energy['kwh'],
        'energy_pct': energy['pct'],
        'sla_episodes_reactive': r['sla_episodes'],
        'sla_episodes_lstm': l['sla_episodes'],
    }

    json_out = f'analysis_{r_ts}__vs__{l_ts}.json'
    with open(json_out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'✓ JSON: {json_out}')

    report = build_report(window, r, l, args.lim_max, args.lim_med)
    from datetime import datetime
    md_out = f'TCC_METRICS_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    with open(md_out, 'w') as f:
        f.write(report)
    print(f'✓ Relatório: {md_out}')

    print('\n' + json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
