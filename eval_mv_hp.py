#coding: utf-8
"""Hyperparameter sweep for the cleaned-feature LSTM (single layer, kept arch).

Goal: pick the HP pool for predict_mv.py. Uses EarlyStopping + restore_best so
val_loss reflects the BEST epoch, not the last (matches how we will save models).
"""
import os, io, logging, warnings
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
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import keras.optimizers
    import eval_mv as E

STEPS_AHEAD = 6
SEED = 7


def run(hostname, n_steps, units, dropout, epochs, batch, lr, patience=25):
    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    df = pd.read_csv(f'{E.MV_DIR}/{hostname}.csv')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('time_stamp').sort_index()
    vals, feats = E.feature_fixed(df)
    dff = pd.DataFrame(vals, index=df.index, columns=feats)
    raw = pd.Series(df['mem'].astype('float64').values, index=dff.index)

    times = dff.index.to_series()
    grp = (times.diff().dt.total_seconds().fillna(0) > 240).cumsum()
    segs = [sub for _, sub in dff.groupby(grp) if len(sub) >= n_steps + STEPS_AHEAD]
    if not segs:
        return None
    mn, mx = E.minmax_fit(vals, feats)
    ti = feats.index(E.TARGET)

    Xs, ys, yr = [], [], []
    for seg in segs:
        sr = raw.loc[seg.index].values
        sv = E.scale(seg.values, mn, mx)
        for i in range(len(seg) - n_steps - STEPS_AHEAD + 1):
            Xs.append(sv[i:i + n_steps])
            t = i + n_steps + STEPS_AHEAD - 1
            ys.append(sv[t, ti]); yr.append(sr[t])
    X = array(Xs); ys = array(ys); yr = array(yr)
    split = len(X) * 8 // 10
    Xtr, Xte = X[:split], X[split:]
    ytr, yte_s = ys[:split], ys[split:]
    yr_te = yr[split:]

    m = Sequential()
    m.add(LSTM(units, input_shape=(n_steps, len(feats))))
    m.add(Dropout(dropout))
    m.add(Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, min_delta=1e-5)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)
    hist = m.fit(Xtr, ytr, epochs=epochs, batch_size=batch, verbose=0,
                 validation_split=0.1, callbacks=[es, rlr])
    best_val = min(hist.history['val_loss'])

    pred = m.predict(Xte, verbose=0).flatten()
    pred = pred * (mx[ti] - mn[ti]) + mn[ti]
    pred = np.clip(pred, 0, 100)
    mae = float(np.mean(np.abs(pred - yr_te)))
    rmse = float(np.sqrt(np.mean((pred - yr_te) ** 2)))
    persist = Xte[:, -1, ti] * (mx[ti] - mn[ti]) + mn[ti]
    p_mae = float(np.mean(np.abs(persist - yr_te)))
    return dict(mae=mae, rmse=rmse, persist_mae=p_mae, best_val=float(best_val),
                stopped=len(hist.history['val_loss']))


CONFIGS = [
    # n_steps, units, dropout, epochs, batch, lr
    (15, 128, 0.2, 250, 32, 0.003),
    (20, 128, 0.2, 250, 32, 0.003),
    (30, 128, 0.2, 250, 32, 0.003),
    (20, 128, 0.3, 250, 32, 0.003),
    (20,  64, 0.2, 250, 32, 0.003),
    (20, 128, 0.2, 250, 16, 0.003),
]


def main():
    hosts = ['compute1', 'compute2', 'compute3']
    print('cleaned-feature LSTM, EarlyStopping(restore_best)+ReduceLROnPlateau\n')
    best_mean = 1e9; best_cfg = None
    for cfg in CONFIGS:
        label = f'n={cfg[0]} u={cfg[1]} do={cfg[2]} ep={cfg[3]} bs={cfg[4]} lr={cfg[5]}'
        maes = []
        for h in hosts:
            r = run(h, *cfg)
            if r is None:
                continue
            maes.append(r['mae'])
            print(f'  {label} | {h}: MAE={r["mae"]:.2f} persist={r["persist_mae"]:.2f} '
                  f'(Δ{r["persist_mae"]-r["mae"]:+.2f}) stop@{r["stopped"]} bestval={r["best_val"]:.5f}')
        if maes:
            mean = np.mean(maes)
            print(f'  {label} | MEAN MAE={mean:.2f}\n')
            if mean < best_mean:
                best_mean = mean; best_cfg = cfg
    print(f'BEST config: {best_cfg}  MEAN MAE={best_mean:.2f}  (current as-is=8.31, feat-fix lstm=8.21)')


if __name__ == '__main__':
    main()
