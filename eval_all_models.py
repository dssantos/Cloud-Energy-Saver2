#coding: utf-8
"""Avalia TODOS os modelos .keras salvos no intervalo do modo LSTM (versão batch).

Para cada modelo salvo, carrega modelo + normalização e faz previsão em BATCH
sobre todas as janelas do período do modo LSTM (02/08 17:34 -> 03/08 07:34),
calculando RMSE, MAE, sMAPE e correlação de Pearson sobre o valor absoluto de RAM.
"""
import io, os, glob, json, logging, warnings
from contextlib import redirect_stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('keras').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

with redirect_stderr(io.StringIO()):
    import numpy as np
    from numpy import array
    import pandas as pd
    from keras.models import load_model

import predict_mv as pm

STEPS_AHEAD = 2
INI = pd.Timestamp('2026-08-02 17:34:17')
FIM = pd.Timestamp('2026-08-03 07:34:22')

MODEL_DIRS = [
    '/home/danilo/dev/python/Cloud-Energy-Saver2/models_mv',
    '/home/danilo/dev/python/Cloud-Energy-Saver2/backup_models_20260809_223401/models_mv',
    '/home/danilo/dev/python/Cloud-Energy-Saver2/backup_models_20260814_235651/models_mv',
]
MEM_IDX = pm.FEATURES.index(pm.TARGET)


def build_windows(dfe):
    """Constrói TODAS as janelas do período, retornando arrays para batch predict.

    Retorna dict com: X (n_janelas, n_steps, n_feat) variando n_steps, ou seja,
    para cada n_steps distinto usado pelos modelos, um conjunto de janelas.
    Como n_steps varia por modelo, construímos janelas por n_steps sob demanda.
    """
    segments = [s for s in pm._split_segments(dfe)]
    return segments


def make_windows_for(segments, n_steps):
    """Constrói janelas (raw) para um n_steps específico.
    Retorna (X_raw shape (n, n_steps, nfeat), last_mem_raw, fut_raw).
    """
    X, last_mem, fut = [], [], []
    for seg in segments:
        if len(seg) < n_steps + STEPS_AHEAD:
            continue
        vals = seg[pm.FEATURES].values.astype('float64')
        for i in range(len(vals) - n_steps - STEPS_AHEAD + 1):
            X.append(vals[i:i + n_steps])
            last_mem.append(vals[i + n_steps - 1, MEM_IDX])
            fut.append(vals[i + n_steps + STEPS_AHEAD - 1, MEM_IDX])
    if not X:
        return None, None, None
    return array(X), array(last_mem), array(fut)


def metrics(actual, pred):
    e = actual - pred
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    smape = float(np.mean(2 * np.abs(e) / (np.abs(actual) + np.abs(pred) + 1e-9)) * 100)
    if np.std(pred) > 0 and np.std(actual) > 0:
        corr = float(np.corrcoef(actual, pred)[0, 1])
    else:
        corr = float('nan')
    return rmse, mae, smape, corr


def list_models(hostname):
    out, seen = [], set()
    for d in MODEL_DIRS:
        for f in glob.glob(f'{d}/{hostname}/*.keras'):
            base = os.path.basename(f)
            if base in seen:
                continue
            seen.add(base)
            parts = base.replace('.keras', '').split('_')
            if len(parts) < 6:
                continue
            try:
                val_loss = float(parts[0]); ts = int(parts[1])
                epochs = int(parts[2]); n_steps = int(parts[3]); units = int(parts[4])
            except (ValueError, IndexError):
                continue
            if not (20260802173417 <= ts <= 20260803073422):
                continue
            out.append((val_loss, ts, base, f, epochs, n_steps, units))
    return out


def main():
    hosts = ['compute1', 'compute2', 'compute3']
    all_results = []

    for host in hosts:
        df = pd.read_csv(f'{pm.MV_DIR}/{host}.csv')
        df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('time_stamp').sort_index()
        dfe = pm._engineer(df)[pm.FEATURES].dropna()
        dfe = dfe.loc[(dfe.index >= INI) & (dfe.index <= FIM)]
        if len(dfe) < 30:
            continue
        segments = build_windows(dfe)

        models = list_models(host)
        print(f'{host}: {len(models)} modelos | {len(dfe)} amostras no período', flush=True)

        # Baseline Naive (persistência)
        naive_act, naive_pred = [], []
        for seg in segments:
            vals = seg[pm.FEATURES].values.astype('float64')
            if len(vals) < 2 + STEPS_AHEAD:
                continue
            for i in range(len(vals) - 2 - STEPS_AHEAD + 1):
                naive_act.append(vals[i + 1 + STEPS_AHEAD - 1, MEM_IDX])
                naive_pred.append(vals[i, MEM_IDX])
        n_rmse, n_mae, n_smape, n_corr = metrics(array(naive_act), array(naive_pred))
        print(f'  [NAIVE] RMSE={n_rmse:.2f} MAE={n_mae:.2f} sMAPE={n_smape:.1f}% corr={n_corr:.4f} n={len(naive_act)}', flush=True)

        # cache janelas por n_steps (evita reconstruir)
        win_cache = {}

        for (val_loss, ts, base, f, epochs, n_steps, units) in models:
            try:
                with open(f + '.norm.json') as fh:
                    nm = json.load(fh)
                mn = array(nm['min']); mx = array(nm['max'])
                if len(mn) != len(pm.FEATURES):
                    continue
                model = load_model(f, compile=False)
            except Exception as e:
                continue

            if n_steps not in win_cache:
                X, last_mem, fut = make_windows_for(segments, n_steps)
                win_cache[n_steps] = (X, last_mem, fut)
            else:
                X, last_mem, fut = win_cache[n_steps]

            if X is None or len(X) == 0:
                continue

            # batch predict
            Xs = pm._scale(X, mn, mx)
            last_mem_scaled = (last_mem - mn[MEM_IDX]) / (mx[MEM_IDX] - mn[MEM_IDX])
            d = model.predict(Xs, verbose=0).flatten()
            pred_mem_scaled = last_mem_scaled + d
            pred = np.clip(pred_mem_scaled * (mx[MEM_IDX] - mn[MEM_IDX]) + mn[MEM_IDX], 0, 100)
            actual = fut

            rmse, mae, smape, corr = metrics(actual, pred)
            all_results.append(dict(
                host=host, val_loss=val_loss, ts=ts, filename=base,
                epochs=epochs, n_steps=n_steps, units=units,
                rmse=rmse, mae=mae, smape=smape, corr=corr, n=len(actual),
            ))
            print(f'  {base[:44]:46s} RMSE={rmse:6.2f} MAE={mae:6.2f} sMAPE={smape:5.1f}% corr={corr:.4f}', flush=True)

    if not all_results:
        print('Nenhum modelo avaliado.')
        return

    print('\n' + '=' * 90)
    print('MELHORES MODELOS POR MÉTRICA (valor absoluto mem %, período do modo LSTM)')
    print('=' * 90)
    for metric, better in [('rmse', 'menor'), ('mae', 'menor'), ('smape', 'menor'), ('corr', 'maior')]:
        reverse = (metric == 'corr')
        best = sorted(all_results, key=lambda r: r[metric], reverse=reverse)[0]
        print(f'\nMelhor por {metric.upper()} ({metric}={best[metric]:.4f}, {better} é melhor):')
        print(f'  host={best["host"]} | {best["filename"]}')
        print(f'  epochs={best["epochs"]} n_steps={best["n_steps"]} units={best["units"]}')
        print(f'  RMSE={best["rmse"]:.2f} MAE={best["mae"]:.2f} sMAPE={best["smape"]:.1f}% corr={best["corr"]:.4f}')

    with open('eval_all_models_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSalvo em eval_all_models_results.json ({len(all_results)} modelos)')


if __name__ == '__main__':
    main()
