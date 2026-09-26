#coding: utf-8
"""Histórico de predições para métricas de erro do TCC (MAE, RMSE, etc.).

Cada predição ao vivo (registrada em verifier.calculate_ram_average) entra
aqui com carimbo de tempo; um thread em background, quando o instante-alvo
chega, cruza com a leitura REAL de mem% em workload_mv/{host}.csv e grava a
linha no CSV do run:

  time_stamp, target_time_stamp, hostname, model, model_file, steps_ahead,
  predicted, mem_at_pred, actual, abs_error

`mem_at_pred` (âncora) é a leitura real de mem% no INSTANTE da predição (t),
passada por calculate_ram_average — base direta para Directional Accuracy e
para a persistência pareada (MASE_p/MSESS_p) sem precisar do join posterior
com workload_mv. Linhas de CSVs antigos (sem a coluna) podem reconstruir a
âncora por join com workload_mv/{host}.csv.

Protocolo de avaliação: toda predição (de qualquer modelo) é avaliada contra o
real em t + STEPS_AHEAD * MV_SAMPLE_INTERVAL_S (3 min à frente com os defaults),
para que lstm/naive/arima/default sejam comparáveis entre si. Para o modelo
`default` a "predição" é a própria leitura do momento (persistência), que serve
de baseline de erro.

O CSV é nomeado como os demais artefatos do run (mesmo {ts} de events_/cluster_
workload_): predictions_{model}_{ts}.csv — definido por set_file() no início do
experimento. Linhas cujo alvo não pôde ser resolvido (host fora, sem amostra
próxima em até MAX_PENDING_AGE_S) são gravadas com `actual` vazio — excluí-las
da análise. Predições pendentes não resolvidas quando o processo termina são
perdidas (janela máxima: STEPS_AHEAD * MV_SAMPLE_INTERVAL_S + MAX_PENDING_AGE_S).
"""
import csv
import os
import threading
from datetime import datetime
from time import sleep, time

import config

SAMPLE_S = config.MV_SAMPLE_INTERVAL_S
MATCH_TOLERANCE_S = SAMPLE_S * 1.5  # amostra mais próxima admitida como "o real no alvo"
MAX_PENDING_AGE_S = 600             # desiste de resolver após isso (grava actual vazio)
RESOLVE_INTERVAL_S = 15             # cadência do thread resolvedor

_pending = []          # predições aguardando o instante-alvo
_lock = threading.Lock()
_file = None           # CSV corrente (definido por set_file)
_thread_started = False


def set_file(filename):
    """Define o CSV de saída do run (sem criar o arquivo; só ao gravar a 1ª linha)."""
    global _file
    with _lock:
        _file = filename


def record(hostname, model, predicted, steps_ahead=None, model_file=None, pred_ts=None,
           mem_now=None):
    """Registra uma predição; o real no instante-alvo é resolvido depois.

    mem_now: leitura real de mem% no instante da predição (âncora p/ DA e
    persistência pareada). None -> coluna vazia (recuperável por join).
    """
    global _thread_started
    pred_ts = time() if pred_ts is None else pred_ts
    steps = config.STEPS_AHEAD if steps_ahead is None else int(steps_ahead)
    with _lock:
        _pending.append({
            'hostname': hostname,
            'model': model,
            'model_file': model_file or '',
            'steps_ahead': steps,
            'predicted': float(predicted),
            'mem_now': float(mem_now) if mem_now is not None else None,
            'pred_ts': pred_ts,
            'target_ts': pred_ts + steps * SAMPLE_S,
        })
        if not _thread_started:
            _thread_started = True
            threading.Thread(target=_resolve_loop, daemon=True).start()


def _mem_at(hostname, target_ts):
    """mem% real mais próxima de target_ts em workload_mv/{host}.csv (None se longe demais)."""
    path = f'workload_mv/{hostname}.csv'
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd  # lazy: manter as utilidades manuais do orchestrator leves
        df = pd.read_csv(path, usecols=['time_stamp', 'mem'])
        ts = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
        target = pd.Timestamp(datetime.fromtimestamp(target_ts))
        delta = (ts - target).abs()
        i = delta.idxmin()
        if delta.iloc[i].total_seconds() <= MATCH_TOLERANCE_S:
            return float(df['mem'].iloc[i])
    except Exception:
        pass
    return None


def _append_rows(rows):
    """Grava linhas resolvidas no CSV do run (append; header só no arquivo novo)."""
    if not rows:
        return
    with _lock:
        filename = _file or 'predictions.csv'
        try:
            new_file = not os.path.exists(filename)
            with open(filename, 'a', newline='') as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(['time_stamp', 'target_time_stamp', 'hostname', 'model',
                                'model_file', 'steps_ahead', 'predicted', 'mem_at_pred',
                                'actual', 'abs_error'])
                w.writerows(rows)
        except Exception as e:
            print(f'[PREDICT LOG ERROR] {e}')


def _resolve_loop():
    """Resolve predições cujo instante-alvo já chegou (daemon, iniciado no 1º record)."""
    while True:
        sleep(RESOLVE_INTERVAL_S)
        now = time()
        with _lock:
            due = [p for p in _pending if p['target_ts'] <= now]
            _pending[:] = [p for p in _pending if p['target_ts'] > now]
        if not due:
            continue
        rows, unresolved = [], []
        for p in due:
            actual = _mem_at(p['hostname'], p['target_ts'])
            if actual is None and now - p['target_ts'] < MAX_PENDING_AGE_S:
                unresolved.append(p)  # amostra ainda não chegou (host fora/coleção atrasada); tenta de novo
                continue
            rows.append([
                datetime.fromtimestamp(p['pred_ts']).strftime('%Y-%m-%d %H:%M:%S'),
                datetime.fromtimestamp(p['target_ts']).strftime('%Y-%m-%d %H:%M:%S'),
                p['hostname'], p['model'], p['model_file'], p['steps_ahead'],
                f"{p['predicted']:.2f}",
                f"{p['mem_now']:.2f}" if p.get('mem_now') is not None else '',
                f'{actual:.2f}' if actual is not None else '',
                f'{abs(p["predicted"] - actual):.2f}' if actual is not None else '',
            ])
        with _lock:
            _pending.extend(unresolved)
        _append_rows(rows)
