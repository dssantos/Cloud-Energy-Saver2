#coding: utf-8
"""Offline evaluation at STEPS_AHEAD=2 (1 min), comparing:
  OLD:  raw features (swap, sentinel secs, raw loadavg), old HP pool, no ES
  NEW:  cleaned features (recency, log loadavg, no swap), new HP pool, ES
  PERSISTENCE: predict = current mem (t, not t-2)

Fixed seed for reproducibility. Reports MAE/RMSE in mem % on temporal test split.
"""
import os, io, json, logging, warnings
from contextlib import redirect_stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('keras').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

with redirect_stderr(io.StringIO()):
    import numpy as np
    from numpy import array, where, exp, log1p, clip
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import keras.optimizers

MV_DIR = 'workload_mv'
STEPS = 2
SEED = 7
RECEIVE_DECAY_S = 300.0
LOADAVG_CLIP = 16.0

# --- Feature sets ---
RAW_COLS = ['mem', 'vms', 'secs_since_vm_created', 'secs_since_vm_deleted', 'swap', 'loadavg']
NEW_FEATS = ['mem', 'vms', 'rec_create', 'rec_delete', 'loadavg']


def engineer(df):
    out = pd.DataFrame(index=df.index)
    out['mem'] = df['mem'].astype('float64')
    out['vms'] = df['vms'].astype('float64')
    cre = df['secs_since_vm_created'].astype('float64').values
    dele = df['secs_since_vm_deleted'].astype('float64').values
    out['rec_create'] = where(cre >= 9998, 0.0, exp(-cre / RECEIVE_DECAY_S))
    out['rec_delete'] = where(dele >= 9998, 0.0, exp(-dele / RECEIVE_DECAY_S))
    out['loadavg'] = log1p(clip(df['loadavg'].astype('float64').values, 0, LOADAVG_CLIP))
    return out


def minmax_fit(vals):
    mn = np.array([vals[:, i].min() for i in range(vals.shape[1])])
    mx = np.array([vals[:, i].max() for i in range(vals.shape[1])])
    mx = np.where(mx > mn, mx, mn + 1.0)
    return mn, mx


def scale(arr2d, mn, mx):
    return (arr2d - mn) / (mx - mn)


def segs_and_windows(df, feats, n_steps):
    """Return (X, y_scaled, y_raw) arrays with temporal ordering."""
    dff = df[feats].astype('float64')
    vals = dff.values
    raw = df['mem'].astype('float64').values
    times = dff.index.to_series()
    grp = (times.diff().dt.total_seconds().fillna(0) > 240).cumsum()
    mn, mx = minmax_fit(vals)
    Xs, ys, yr = [], [], []
    for _, sub in dff.groupby(grp):
        if len(sub) < n_steps + STEPS:
            continue
        sv = scale(sub.values, mn, mx)
        sr = pd.Series(raw, index=dff.index).loc[sub.index].values
        ti = list(feats).index('mem')
        for i in range(len(sub) - n_steps - STEPS + 1):
            Xs.append(sv[i:i + n_steps])
            ys.append(sv[i + n_steps + STEPS - 1, ti])
            yr.append(sr[i + n_steps + STEPS - 1])
    if not Xs:
        return None, None, None, mn, mx, None
    return array(Xs), array(ys), array(yr), mn, mx, ti


def train_eval(hostname, feats, n_steps, units, epochs, dropout, lr, use_es=True):
    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    df = pd.read_csv(f'{MV_DIR}/{hostname}.csv')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('time_stamp').sort_index()
    # Build feature df
    if feats == NEW_FEATS:
        dff = engineer(df)
    else:
        dff = df[feats].astype('float64')
    X, ys, yr, mn, mx, ti = segs_and_windows(dff, feats, n_steps)
    if X is None:
        return None
    split = len(X) * 8 // 10
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = ys[:split], ys[split:]
    yr_te = yr[split:]

    m = Sequential()
    m.add(LSTM(units, input_shape=(n_steps, len(feats))))
    m.add(Dropout(dropout))
    m.add(Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    if use_es:
        es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, min_delta=1e-5)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)
        hist = m.fit(Xtr, ytr, epochs=epochs, batch_size=32, verbose=0,
                     validation_split=0.1, callbacks=[es, rlr])
        best_val = float(min(hist.history['val_loss']))
    else:
        hist = m.fit(Xtr, ytr, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)
        best_val = float(hist.history['val_loss'][-1])

    pred = m.predict(Xte, verbose=0).flatten()
    pred = pred * (mx[ti] - mn[ti]) + mn[ti]
    pred = np.clip(pred, 0, 100)
    mae = float(np.mean(np.abs(pred - yr_te)))
    rmse = float(np.sqrt(np.mean((pred - yr_te) ** 2)))
    # Persistence baseline on same test windows
    persist = Xte[:, -1, ti] * (mx[ti] - mn[ti]) + mn[ti]
    p_mae = float(np.mean(np.abs(persist - yr_te)))
    return dict(mae=mae, rmse=rmse, p_mae=p_mae, best_val=best_val, n_test=len(yr_te))


HOSTS = ['compute1', 'compute2', 'compute3']

# Configs: (label, feats, n_steps, units, epochs, dropout, lr, use_es)
CONFIGS = [
    ('OLD   (raw feats, hp10-20, no ES)', RAW_COLS,       15, 128, 120, 0.2, 0.003, False),
    ('OLD   (raw feats, hp10-20, no ES)', RAW_COLS,       20, 128, 120, 0.2, 0.003, False),
    ('NEW   (clean feats, n=20, ES)     ', NEW_FEATS,     20, 128, 250, 0.2, 0.003, True),
    ('NEW   (clean feats, n=30, ES)     ', NEW_FEATS,     30, 128, 250, 0.2, 0.003, True),
    ('NEW   (clean feats, n=30, u=256)  ', NEW_FEATS,     30, 256, 250, 0.2, 0.003, True),
]


def main():
    print(f'STEPS_AHEAD={STEPS} (1 min)  decay={RECEIVE_DECAY_S}s  loadclip={LOADAVG_CLIP}\n')
    for label, feats, ns, u, ep, do, lr, es in CONFIGS:
        print(f'### {label}  n={ns} u={u} ep={ep} do={do} lr={lr} es={es}')
        maes = []
        for h in HOSTS:
            r = train_eval(h, feats, ns, u, ep, do, lr, es)
            if r is None:
                print(f'  {h}: no data'); continue
            maes.append(r['mae'])
            print(f'  {h}: MAE={r["mae"]:.2f}  persist={r["p_mae"]:.2f}  '
                  f'(Δ{r["p_mae"]-r["mae"]:+.2f})  RMSE={r["rmse"]:.2f}  '
                  f'bestval={r["best_val"]:.5f}  ntest={r["n_test"]}')
        if maes:
            print(f'  MEAN MAE={np.mean(maes):.2f}\n')
    # Also print persistence baseline directly
    print('### PERSISTENCE BASELINE (lag 2, all hosts)')
    for h in HOSTS:
        df = pd.read_csv(f'{MV_DIR}/{h}.csv')
        m = df['mem'].astype('float64').values
        pers = np.abs(m[:-STEPS] - m[STEPS:])
        print(f'  {h}: MAE={pers.mean():.2f} RMSE={np.sqrt((pers**2).mean()):.2f}')
    print()


if __name__ == '__main__':
    main()
