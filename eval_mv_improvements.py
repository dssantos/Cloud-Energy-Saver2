#coding: utf-8
"""Offline comparison of the extended LSTM (multi-layer + new features) vs persistence.

Evaluates the changes introduced on the lstm-hp-expansion branch against the
reactive-mode baseline (persistence = mem[t]) on the existing workload_mv CSVs:

  - 5feat-L1 : previous feature set (5) + single LSTM layer (baseline)
  - 7feat-L1 : new features (mem_ma, mem_diff) + single layer
  - 7feat-L2 : new features + stacked LSTM (return_sequences + 2nd layer)

Uses predict_mv's own feature engineering / scaling / segmentation, a temporal
70/30 split (scaler fit on train only), delta target, and walk-forward forecast.
Reports RMSE/MAE per host and pooled so we can see if the extension beats naive.
"""
import io, os, logging, warnings
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
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping
    import keras.optimizers

import predict_mv as pm

STEPS_AHEAD = 2
SEED = 7

FEATS5 = ['mem', 'vms', 'rec_create', 'rec_delete', 'loadavg']   # previous feature set
FEATS7 = list(pm.FEATURES)                                       # new (mem_ma, mem_diff)

CONFIGS = [
    dict(label='5feat-L1-u128',   feats=FEATS5, num_layers=1, units=128, units2=0),
    dict(label='7feat-L1-u128',   feats=FEATS7, num_layers=1, units=128, units2=0),
    dict(label='7feat-L2-u128-64', feats=FEATS7, num_layers=2, units=128, units2=64),
    dict(label='7feat-L2-u128-128', feats=FEATS7, num_layers=2, units=128, units2=128),
]

FIXED = dict(n_steps=30, dropout=0.2, lr=0.002, epochs=150, batch=32)


def build_model(feat_n, n_steps, num_layers, units, units2, dropout, lr):
    m = Sequential()
    if num_layers == 2:
        m.add(LSTM(units, activation='relu', return_sequences=True,
                   input_shape=(n_steps, feat_n)))
        m.add(Dropout(dropout))
        m.add(LSTM(units2, activation='relu'))
        m.add(Dropout(dropout))
    else:
        m.add(LSTM(units, activation='relu', input_shape=(n_steps, feat_n)))
        m.add(Dropout(dropout))
    m.add(Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return m


def evaluate_host(hostname, cfg, epochs=None):
    import tensorflow as tf
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    feats = cfg['feats']
    n_steps = FIXED['n_steps']
    epochs = epochs or FIXED['epochs']

    df = pd.read_csv(f'{pm.MV_DIR}/{hostname}.csv')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('time_stamp').sort_index()
    dfe = pm._engineer(df)[feats].dropna()

    mem_idx = feats.index('mem')
    segments = [s for s in pm._split_segments(dfe) if len(s) >= n_steps + STEPS_AHEAD]
    if not segments:
        return None

    raw_X, raw_mem_now, raw_mem_fut = [], [], []
    for seg in segments:
        vals = seg[feats].values.astype('float64')
        for i in range(len(vals) - n_steps - STEPS_AHEAD + 1):
            raw_X.append(vals[i:i + n_steps])
            raw_mem_now.append(vals[i + n_steps - 1, mem_idx])
            raw_mem_fut.append(vals[i + n_steps + STEPS_AHEAD - 1, mem_idx])

    split = len(raw_X) * 7 // 10
    tr_X = array(raw_X[:split]); tr_now = array(raw_mem_now[:split]); tr_fut = array(raw_mem_fut[:split])
    te_X = array(raw_X[split:]); te_now = array(raw_mem_now[split:]); te_fut = array(raw_mem_fut[split:])

    mn, mx = pm._fit_minmax(tr_X.reshape(-1, len(feats)))

    def to_delta(Xs, nows, futs):
        Xs_s = pm._scale(Xs, mn, mx)
        now_s = (nows - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
        fut_s = (futs - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
        return Xs_s, fut_s - now_s

    Xtr, ytr = to_delta(tr_X, tr_now, tr_fut)
    Xte_s, _ = to_delta(te_X, te_now, te_fut)

    model = build_model(len(feats), n_steps, cfg['num_layers'], cfg['units'],
                        cfg['units2'], FIXED['dropout'], FIXED['lr'])
    es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, min_delta=1e-6)
    model.fit(Xtr, ytr, epochs=epochs, batch_size=FIXED['batch'], verbose=0,
              validation_split=0.1, callbacks=[es])

    deltas = model.predict(Xte_s, verbose=0).flatten()
    last_mem_scaled = (te_now - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
    pred_mem = np.clip(last_mem_scaled + deltas, 0, 1)
    pred_mem = pred_mem * (mx[mem_idx] - mn[mem_idx]) + mn[mem_idx]
    actual = te_fut
    persist = te_now

    e_lstm = actual - pred_mem
    e_naive = actual - persist
    return dict(n=len(actual),
                lstm_rmse=float(np.sqrt(np.mean(e_lstm ** 2))),
                lstm_mae=float(np.mean(np.abs(e_lstm))),
                naive_rmse=float(np.sqrt(np.mean(e_naive ** 2))),
                naive_mae=float(np.mean(np.abs(e_naive))),
                lstm_e=e_lstm, naive_e=e_naive)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hosts', default='compute1,compute2,compute3')
    ap.add_argument('--epochs', type=int, default=None, help='override epochs for a quick run')
    args = ap.parse_args()
    hosts = args.hosts.split(',')

    print(f'Extended LSTM vs persistence | STEPS_AHEAD={STEPS_AHEAD} | fixed={FIXED}')
    print(f'hosts={hosts} | epochs={args.epochs or FIXED["epochs"]}\n')

    results = {}
    for cfg in CONFIGS:
        all_lstm, all_naive = [], []
        per_host = []
        for h in hosts:
            r = evaluate_host(h, cfg, epochs=args.epochs)
            if r is None:
                print(f'  {cfg["label"]:18s} | {h}: sem dados suficientes')
                continue
            all_lstm.extend(r['lstm_e']); all_naive.extend(r['naive_e'])
            win = 'LSTM' if r['lstm_mae'] < r['naive_mae'] else 'naive'
            per_host.append(f'{h}:{r["lstm_mae"]:.2f}/{r["naive_mae"]:.2f}={win}')
        if all_lstm:
            el = array(all_lstm); en = array(all_naive)
            results[cfg['label']] = dict(
                lstm_rmse=float(np.sqrt(np.mean(el ** 2))),
                naive_rmse=float(np.sqrt(np.mean(en ** 2))),
                lstm_mae=float(np.mean(np.abs(el))),
                naive_mae=float(np.mean(np.abs(en))),
            )
            r = results[cfg['label']]
            print(f'{cfg["label"]:18s} | LSTM RMSE={r["lstm_rmse"]:.2f} MAE={r["lstm_mae"]:.2f}'
                  f' | naive RMSE={r["naive_rmse"]:.2f} MAE={r["naive_mae"]:.2f}')
            print(f'{"":18s} | {" | ".join(per_host)}')
        print()

    print('=== SUMMARY (pooled, sorted by LSTM RMSE) ===')
    for label, r in sorted(results.items(), key=lambda kv: kv[1]['lstm_rmse']):
        print(f'{label:18s} RMSE={r["lstm_rmse"]:.2f} MAE={r["lstm_mae"]:.2f}'
              f'  (naive RMSE={r["naive_rmse"]:.2f} MAE={r["naive_mae"]:.2f})')


if __name__ == '__main__':
    main()
