#coding: utf-8
"""Verification of the delta-target LSTM vs persistence (naive) on existing data.

Reuses predict_mv's feature engineering / scaling / segmentation, trains a
delta LSTM on the TRAIN split (scaler fit on train only), then does a
walk-forward evaluation on the TEST split comparing:
  - delta LSTM : mem[t] + predicted_delta
  - persistence : mem[t]   (= the implicit forecast of the reactive mode)

Reports RMSE/MAE per host and pooled, plus a DM-style significance check.
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
    import keras.optimizers

import predict_mv as pm

STEPS_AHEAD = 2
SEED = 7
# Fixed config for reproducibility (not random search)
HP = dict(n_steps=30, units=128, epochs=150, batch=32, dropout=0.2, lr=0.002)


def build_delta_model(feat_n, n_steps, units, dropout, lr):
    m = Sequential()
    m.add(LSTM(units, activation='relu', input_shape=(n_steps, feat_n)))
    m.add(Dropout(dropout))
    m.add(Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return m


def evaluate_host(hostname):
    import tensorflow as tf
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    df = pd.read_csv(f'{pm.MV_DIR}/{hostname}.csv')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('time_stamp').sort_index()
    dfe = pm._engineer(df)[pm.FEATURES].dropna()

    n_steps = HP['n_steps']
    mem_idx = pm.FEATURES.index(pm.TARGET)
    segments = [s for s in pm._split_segments(dfe) if len(s) >= n_steps + STEPS_AHEAD]
    if not segments:
        return None

    # Build raw windows (unscaled)
    raw_X, raw_mem_now, raw_mem_fut = [], [], []
    for seg in segments:
        vals = seg[pm.FEATURES].values.astype('float64')
        for i in range(len(vals) - n_steps - STEPS_AHEAD + 1):
            raw_X.append(vals[i:i + n_steps])
            raw_mem_now.append(vals[i + n_steps - 1, mem_idx])
            raw_mem_fut.append(vals[i + n_steps + STEPS_AHEAD - 1, mem_idx])

    # Temporal 70/30 split
    split = len(raw_X) * 7 // 10
    tr_X = array(raw_X[:split])
    tr_now = array(raw_mem_now[:split])
    tr_fut = array(raw_mem_fut[:split])
    te_X = array(raw_X[split:])
    te_now = array(raw_mem_now[split:])
    te_fut = array(raw_mem_fut[split:])

    mn, mx = pm._fit_minmax(tr_X.reshape(-1, len(pm.FEATURES)))

    def to_delta(Xs, nows, futs):
        Xs_s = pm._scale(Xs, mn, mx)
        now_s = (nows - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
        fut_s = (futs - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
        return Xs_s, fut_s - now_s

    Xtr, ytr = to_delta(tr_X, tr_now, tr_fut)
    Xte_s, _ = to_delta(te_X, te_now, te_fut)

    model = build_delta_model(len(pm.FEATURES), n_steps, HP['units'], HP['dropout'], HP['lr'])
    model.fit(Xtr, ytr, epochs=HP['epochs'], batch_size=HP['batch'], verbose=0, validation_split=0.1)

    # Walk-forward forecast on test: pred_delta from window, add to last mem
    pred_mem = []
    for i in range(len(te_X)):
        win = Xte_s[i]
        last_mem_scaled = (te_now[i] - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
        d = model.predict(win.reshape(1, n_steps, len(pm.FEATURES)), verbose=0)[0][0]
        pred_mem_scaled = last_mem_scaled + d
        pred_mem.append(pred_mem_scaled * (mx[mem_idx] - mn[mem_idx]) + mn[mem_idx])
    pred_mem = np.clip(array(pred_mem), 0, 100)
    actual = te_fut
    persist = te_now  # persistence forecast = mem[t]

    e_lstm = actual - pred_mem
    e_naive = actual - persist
    return dict(
        n=len(actual),
        lstm_rmse=float(np.sqrt(np.mean(e_lstm ** 2))),
        lstm_mae=float(np.mean(np.abs(e_lstm))),
        naive_rmse=float(np.sqrt(np.mean(e_naive ** 2))),
        naive_mae=float(np.mean(np.abs(e_naive))),
        lstm_e=e_lstm, naive_e=e_naive,
    )


def main():
    hosts = ['compute1', 'compute2', 'compute3']
    print(f'Delta LSTM vs persistence (walk-forward, STEPS_AHEAD={STEPS_AHEAD}, HP={HP})\n')
    all_lstm, all_naive = [], []
    for h in hosts:
        r = evaluate_host(h)
        if r is None:
            print(f'{h}: sem dados suficientes')
            continue
        all_lstm.extend(r['lstm_e'])
        all_naive.extend(r['naive_e'])
        win = 'LSTM' if r['lstm_mae'] < r['naive_mae'] else 'naive'
        print(f'{h}: n={r["n"]} | LSTM RMSE={r["lstm_rmse"]:.2f} MAE={r["lstm_mae"]:.2f} | '
              f'naive RMSE={r["naive_rmse"]:.2f} MAE={r["naive_mae"]:.2f} | melhor={win}')
    if all_lstm:
        el = array(all_lstm); en = array(all_naive)
        print(f'\nPOOLED: LSTM RMSE={np.sqrt(np.mean(el**2)):.2f} MAE={np.mean(np.abs(el)):.2f} | '
              f'naive RMSE={np.sqrt(np.mean(en**2)):.2f} MAE={np.mean(np.abs(en)):.2f}')
        # DM-style (simple, no HAC) on squared loss differential
        d = el ** 2 - en ** 2
        dm = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))) if np.std(d) > 0 else float('nan')
        print(f'DM (approx) = {dm:.3f}  (negativo = LSTM melhor) | mean(d)={np.mean(d):.4f}')


if __name__ == '__main__':
    main()
