#coding: utf-8
"""Métricas de erro de previsão a partir de predictions_{model}_{ts}.csv (predict_log).

Calcula, por braço de modelo (default, default_tend, naive, lstm, lstm_tend, ...):
  MAE, RMSE, MAPE, sMAPE, R²  -> direto do par (predicted, actual)
  Directional Accuracy        -> usa a âncora (mem real no instante da predição):
                                 coluna mem_at_pred do CSV (runs com predict_log
                                 novo) ou, onde ausente, join com
                                 workload_mv/{host}.csv (tolerância 60 s)
  MASE / MSESS                -> skill scores vs o braço NAIVE (MAE/MSE do braço) e
                                 vs persistência PAREADA na mesma origem (âncora
                                 mem_at_pred/join), que controla as janelas
                                 distintas dos braços.

Protocolo:
  - linhas com actual vazio (alvo não resolvido, host fora) são EXCLUÍDAS;
  - MAPE/sMAPE exigem actual > 0 (sempre verdade p/ mem% de host up);
  - MASE < 1 e MSESS > 0 = melhor que o naive;
  - a DA (e a leitura absoluta) dos braços de persistência (default/default_tend/
    naive) é piso/ruído — a predição deles é a própria leitura do momento ou o
    último registro; a DA é realmente informativa nos braços lstm.

Uso (onde estiverem os artefatos do run, ex.: na VPS):
  python analyze_predictions.py                            # todos os braços
  python analyze_predictions.py --model lstm --by-host     # um braço, por host
  python analyze_predictions.py --by-model-file            # decompõe lstm por .keras
"""
import argparse
import glob

import numpy as np
import pandas as pd


def load(fn):
    df = pd.read_csv(fn)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df['target_time_stamp'] = pd.to_datetime(df['target_time_stamp'], format='%Y-%m-%d %H:%M:%S')
    for c in ('predicted', 'actual'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _anchor_cache(host, cache={}):
    if host not in cache:
        df = pd.read_csv(f'workload_mv/{host}.csv', usecols=['time_stamp', 'mem'])
        df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
        cache[host] = df.sort_values('time_stamp').reset_index(drop=True)
    return cache[host]


def anchor_mem(host, pred_ts, cache={}):
    """mem real no instante da predição (última amostra <= pred_ts, tolerância 60 s)."""
    w = _anchor_cache(host, cache)
    i = w['time_stamp'].searchsorted(pred_ts, side='right') - 1
    if i < 0:
        return np.nan
    row = w.iloc[i]
    return row['mem'] if (pred_ts - row['time_stamp']).total_seconds() <= 60 else np.nan


def metrics(df, anchor_cache=None):
    d = df.dropna(subset=['predicted', 'actual'])
    err = d['predicted'] - d['actual']
    out = {
        'n': len(d),
        'sem_actual': int(df['actual'].isna().sum()),
        'MAE': err.abs().mean(),
        'MSE': float((err ** 2).mean()),
        'RMSE': float(np.sqrt((err ** 2).mean())),
        'MAPE': (err.abs() / d['actual']).mean() * 100,
        'sMAPE': (2 * err.abs() / (d['predicted'].abs() + d['actual'].abs())).mean() * 100,
        'R2': 1 - (err ** 2).sum() / ((d['actual'] - d['actual'].mean()) ** 2).sum(),
    }
    cache = anchor_cache if anchor_cache is not None else {}
    # âncora: coluna mem_at_pred (runs com predict_log novo); onde ausente,
    # reconstrói por join com workload_mv (CSVs antigos)
    if 'mem_at_pred' in df.columns:
        anch_col = pd.to_numeric(df.loc[d.index, 'mem_at_pred'], errors='coerce')
        anch_join = np.array([anchor_mem(h, t, cache)
                              for h, t in zip(d['hostname'], d['time_stamp'])])
        anch = anch_col.fillna(pd.Series(anch_join, index=d.index)).values
    else:
        anch = np.array([anchor_mem(h, t, cache)
                         for h, t in zip(d['hostname'], d['time_stamp'])])
    ok = ~np.isnan(anch)
    out['cobDA'] = int(ok.sum())
    if ok.sum():
        dp = np.sign(d['predicted'].values[ok] - anch[ok])
        dr = np.sign(d['actual'].values[ok] - anch[ok])
        out['DA'] = (dp == dr).mean() * 100
        # persistência PAREADA na mesma origem: naive_err = |ancora - real_alvo|
        e_m = d['predicted'].values[ok] - d['actual'].values[ok]
        e_n = anch[ok] - d['actual'].values[ok]
        den = float(np.abs(e_n).mean())
        out['MASE_p'] = float(np.abs(e_m).mean() / den) if den > 0 else np.nan
        den_sq = float((e_n ** 2).sum())
        out['MSESS_p'] = float(1 - (e_m ** 2).sum() / den_sq) if den_sq > 0 else np.nan
    else:
        out['DA'] = np.nan
        out['MASE_p'] = np.nan
        out['MSESS_p'] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description='Métricas de erro das predições (predictions_*.csv)')
    ap.add_argument('--model', help='braço específico (ex.: lstm_tend); padrão: todos')
    ap.add_argument('--by-model-file', action='store_true',
                    help='decompõe cada braço por model_file (útil no lstm)')
    ap.add_argument('--by-host', action='store_true', help='decompõe cada braço por host')
    ap.add_argument('--ts', help='timestamp do run (padrão: mais recente por braço)')
    args = ap.parse_args()

    models = [args.model] if args.model else ['baseline', 'default', 'default_tend', 'naive',
                                              'naive_tend', 'arima', 'lstm', 'lstm_tend']
    cache = {}
    results = {}
    files = {}
    for model in models:
        pat = f'predictions_{model}_{args.ts}.csv' if args.ts else f'predictions_{model}_2*.csv'
        fns = sorted(glob.glob(pat))
        if not fns:
            continue
        files[model] = fns[-1]
        results[model] = metrics(load(fns[-1]), cache)

    print('%-13s %5s %7s %6s  %6s %6s %7s %7s %7s %7s' % (
        'modelo', 'n', 's/acut', 'cobDA', 'MAE', 'RMSE', 'MAPE', 'sMAPE', 'DA%', 'R2'))
    for model, r in results.items():
        print('%-13s %5d %7d %6d  %6.2f %6.2f %6.1f%% %6.1f%% %6.1f%% %7.3f' % (
            model, r['n'], r['sem_actual'], r['cobDA'],
            r['MAE'], r['RMSE'], r['MAPE'], r['sMAPE'], r['DA'], r['R2']))
        if args.by_host:
            for host, g in load(files[model]).groupby('hostname'):
                hr = metrics(g, cache)
                print('   %-10s n=%5d  MAE=%6.2f RMSE=%6.2f DA=%5.1f%% R2=%7.3f' % (
                    host, hr['n'], hr['MAE'], hr['RMSE'], hr['DA'], hr['R2']))
        if args.by_model_file:
            for mf, g in load(files[model]).groupby('model_file'):
                if mf == '' or len(g) < 10:
                    continue
                mr = metrics(g, cache)
                print('   n=%4d MAE=%6.2f RMSE=%6.2f DA=%5.1f%% R2=%7.3f  %s' % (
                    mr['n'], mr['MAE'], mr['RMSE'], mr['DA'], mr['R2'], mf[:60]))

    # ------------------------------------------------------------------
    # Skill scores vs Naive
    # ------------------------------------------------------------------
    ref = results.get('naive')
    print('\n=== Skill scores vs Naive ===')
    if ref is not None and ref['n'] > 0:
        print('referência (braço naive): MAE=%.2f | MSE=%.2f' % (ref['MAE'], ref['MSE']))
    print('%-13s %8s %8s   %9s %9s' % ('modelo', 'MASE', 'MSESS', 'MASE_par', 'MSESS_par'))
    for model, r in results.items():
        if ref is not None and ref['n'] > 0:
            mase = r['MAE'] / ref['MAE']
            msess = 1 - r['MSE'] / ref['MSE'] if ref['MSE'] > 0 else np.nan
            ref_str = '%8.3f %8.3f' % (mase, msess)
        else:
            ref_str = '%8s %8s' % ('-', '-')
        print('%-13s %s   %9.3f %9.3f' % (model, ref_str, r['MASE_p'], r['MSESS_p']))
    print('MASE < 1 / MSESS > 0 = melhor que o naive.')
    print('MASE/MSESS       = vs o braço naive (janelas distintas, runs sequenciais).')
    print('MASE_par/MSESS_par = vs persistência PAREADA na mesma origem (âncora),')
    print('                     controlando diferenças de janela entre braços.')


if __name__ == '__main__':
    main()
