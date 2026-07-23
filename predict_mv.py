#coding: utf-8
"""Multivariate LSTM pipeline (parallel to the univariate predict.py).

Reads workload_mv/{host}.csv, engineers cleaned features (decayed recency,
log-clipped loadavg, drop always-zero swap), trains an LSTM that predicts
`mem` steps_ahead steps ahead, and saves models under models_mv/{host}/.

Normalization (MinMax per feature) is applied because the features have very
different scales (mem 0-100, vms 0-27, recency 0-1, loadavg ~0-3 after log).
"""
import os
import io
import json
import logging
import warnings
import threading
import time
from contextlib import redirect_stderr
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('keras').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

_stderr = io.StringIO()
with redirect_stderr(_stderr):
    from random import choice
    from time import sleep
    from numpy import array, concatenate, where, exp, log1p, clip
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import keras.optimizers
    import pandas as pd

MV_DIR = 'workload_mv'
MODEL_DIR = 'models_mv'

# Raw CSV columns (as written by workload_mv.save)
RAW_COLS = ['mem', 'vms', 'secs_since_vm_created', 'secs_since_vm_deleted', 'swap', 'loadavg']

# Derived feature columns (after _engineer)
FEATURES = ['mem', 'vms', 'rec_create', 'rec_delete', 'loadavg']

TARGET = 'mem'
TRAIN_INTERVAL_S = 600  # retrain every 10 min

# Feature-engineering hyperparameters
RECENCY_DECAY_S = 300.0   # exp decay for rec_create/rec_delete (5 min → ~0.37)
LOADAVG_CLIP = 16.0       # clip loadavg; beyond ~2x cores is overload noise


def _engineer(df):
    """Transform raw CSV columns into cleaned features.

    - mem, vms: kept as-is (already in reasonable ranges).
    - swap: dropped (always 0 in this environment, no signal).
    - secs_since_vm_created/deleted: replaced by exp-decay recency (1=just
      happened, ~0=long ago; sentinel 9999 → 0).
    - loadavg: log1p with clipping to tame overload-spike outliers (82→92
      would otherwise wreck MinMax).
    """
    out = pd.DataFrame(index=df.index)
    out['mem'] = df['mem'].astype('float64')
    out['vms'] = df['vms'].astype('float64')
    cre = df['secs_since_vm_created'].astype('float64').values
    dele = df['secs_since_vm_deleted'].astype('float64').values
    out['rec_create'] = where(cre >= 9998, 0.0, exp(-cre / RECENCY_DECAY_S))
    out['rec_delete'] = where(dele >= 9998, 0.0, exp(-dele / RECENCY_DECAY_S))
    out['loadavg'] = log1p(clip(df['loadavg'].astype('float64').values, 0, LOADAVG_CLIP))
    return out


def _hp():
    """Random hyperparameters.

    n_steps [20, 30]: 10-15 min lookback; tests showed 30 wins for 3-min-ahead
        prediction because the trend needs a longer window. 10/15 removed (too
        short), and 30 added.
    units [128, 256]: 64 removed (worse val_loss on test sweep).
    epochs large w/ EarlyStopping: the callback stops early and restores best
        weights; high values here serve only as a generous ceiling.
    dropout [0.1, 0.2]: 0.3 removed (worse on test sweep).
    lr [0.002, 0.003]: ReduceLROnPlateau refines from the chosen rate.
    """
    return {
        'n_steps': choice([20, 30]),
        'lstm_units': choice([128, 256]),
        'epochs': choice([200, 250, 300]),
        'batch_size': choice([16, 32]),
        'dropout': choice([0.1, 0.2]),
        'learning_rate': choice([0.002, 0.003]),
    }


def _split_segments(df, gap_s=240):
    """Split a datetime-indexed df into continuous segments (gap > gap_s)."""
    df = df.sort_index()
    if len(df) < 2:
        return [df] if len(df) else []
    dt = df.index.to_series().diff().dt.total_seconds().fillna(0)
    group = (dt > gap_s).cumsum()
    return [seg for _, seg in df.groupby(group)]


def _minmax(fit_df):
    """Per-feature min/max from a dataframe (FEATURES)."""
    mn = []
    mx = []
    for col in FEATURES:
        if col in fit_df:
            mn.append(float(fit_df[col].min()))
            mx.append(float(fit_df[col].max()))
        else:
            mn.append(0.0)
            mx.append(1.0)
    # Avoid divide-by-zero: if min==max, set max=min+1
    mx = [m if m > mn[i] else mn[i] + 1.0 for i, m in enumerate(mx)]
    return mn, mx


def _scale(arr2d, mn, mx):
    """Scale a 2D array (samples x features) to [0,1]."""
    mn = array(mn)
    mx = array(mx)
    return (arr2d - mn) / (mx - mn)


def train_lstm_model_mv(hostname, steps_ahead=2):
    """Train one multivariate LSTM model for a host and save it."""
    os.makedirs(f'{MODEL_DIR}/{hostname}', exist_ok=True)
    try:
        df = pd.read_csv(f'{MV_DIR}/{hostname}.csv')
        df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('time_stamp').sort_index()
        # Check raw CSV columns before engineering
        for c in RAW_COLS:
            if c not in df:
                print(f'[MV TRAIN] {hostname}: coluna {c} ausente, abortando.')
                return None
        dfe = _engineer(df)[FEATURES].dropna()

        hp = _hp()
        n_steps = hp['n_steps']
        units = hp['lstm_units']
        epochs = hp['epochs']
        batch = hp['batch_size']
        dropout = hp['dropout']
        lr = hp['learning_rate']

        segments = [s for s in _split_segments(dfe) if len(s) >= n_steps + steps_ahead]
        if not segments:
            print(f'[MV TRAIN] {hostname}: sem segmentos suficientes ({len(dfe)} amostras).')
            return None

        # Fit min/max on all data (fit scaler before splitting windows)
        mn, mx = _minmax(dfe)

        Xs, ys = [], []
        for seg in segments:
            vals = _scale(seg[FEATURES].values.astype('float64'), mn, mx)
            target_idx = FEATURES.index(TARGET)
            for i in range(len(vals) - n_steps - steps_ahead + 1):
                Xs.append(vals[i:i + n_steps])
                ys.append(vals[i + n_steps + steps_ahead - 1, target_idx])
        if not Xs:
            print(f'[MV TRAIN] {hostname}: sem janelas após split.')
            return None
        X = array(Xs)
        y = array(ys)

        split = len(X) * 8 // 10
        Xtr, Xval = X[:split], X[split:]
        ytr, yval = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(units, input_shape=(n_steps, len(FEATURES))))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')

        # EarlyStopping: stop when val_loss stops improving; restore best weights
        # so the saved model is the peak, not the possibly-worse last epoch.
        # ReduceLROnPlateau: halve lr when valley plateaus (helps fine-tuning).
        es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, min_delta=1e-5)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)

        print(f'[MV TRAIN] {hostname}: X={X.shape} | hp={hp}')
        history = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch,
                            verbose=0, validation_data=(Xval, yval),
                            callbacks=[es, rlr])
        # Use the BEST validation loss (minimum), not the last epoch's value
        val_loss = float(min(history.history['val_loss']))
        print(f'[MV TRAIN] {hostname}: val_loss={val_loss:.6f} (best; stopped @ {len(history.history["val_loss"])})')

        loss_str = '.'.join([str(val_loss).split('.')[0].zfill(3), str(val_loss).split('.')[-1][:6]])
        ts = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        fname = f'{loss_str}_{ts}_{epochs}_{n_steps}_{units}_{steps_ahead}ahead.keras'
        model.save(f'{MODEL_DIR}/{hostname}/{fname}')
        # Save normalization parameters for DERIVED features (engineered scale)
        with open(f'{MODEL_DIR}/{hostname}/{fname}.norm.json', 'w') as f:
            json.dump({'min': mn, 'max': mx}, f)
        print(f'[MV MODEL SAVED] {hostname}: {fname}')
        return fname
    except Exception as e:
        print(f'[MV TRAIN ERROR] {hostname}: {e}')
        return None


def select_best_model_mv(hostname):
    """Return filename of the multivariate model with lowest val_loss, or None."""
    d = f'{MODEL_DIR}/{hostname}'
    if not os.path.isdir(d):
        return None
    best = None
    best_loss = float('inf')
    for f in os.listdir(d):
        if f.endswith('.keras'):
            try:
                loss = float(f.split('_')[0])
            except ValueError:
                continue
            if loss < best_loss:
                best_loss = loss
                best = f
    return best


def lstm_mv(hostname, steps_ahead=2):
    """Predict mem steps_ahead ahead using the best multivariate model.

    Returns the predicted mem (%), or None if no model / not enough data.
    Tries models in descending order of val_loss; skips any whose norm.json
    feature count doesn't match the current FEATURES (stale models from an
    older feature set).
    """
    d = f'{MODEL_DIR}/{hostname}'
    if not os.path.isdir(d):
        return None
    # Collect available models sorted by val_loss (best first)
    cand = []
    for f in os.listdir(d):
        if f.endswith('.keras'):
            try:
                cand.append((float(f.split('_')[0]), f))
            except ValueError:
                continue
    cand.sort(key=lambda x: x[0])
    df = pd.read_csv(f'{MV_DIR}/{hostname}.csv')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('time_stamp').sort_index()
    dfe = _engineer(df)[FEATURES].dropna()

    for _, best_file in cand:
        try:
            # Parse n_steps from filename: {loss}_{ts}_{epochs}_{n_steps}_{units}_{sa}ahead.keras
            parts = best_file.replace('.keras', '').split('_')
            n_steps = int(parts[3])
            if len(dfe) < n_steps:
                continue
            with open(f'{MODEL_DIR}/{hostname}/{best_file}.norm.json') as f:
                nm = json.load(f)
            mn, mx = nm['min'], nm['max']
            # Skip models trained with a different feature set
            if len(mn) != len(FEATURES):
                print(f'[MV PREDICT] skipping {best_file}: norm dim mismatch')
                continue
            window = _scale(dfe[FEATURES].values.astype('float64')[-n_steps:], mn, mx)
            x_input = window.reshape((1, n_steps, len(FEATURES)))
            model = load_model(f'{MODEL_DIR}/{hostname}/{best_file}', compile=False)
            pred_scaled = model.predict(x_input, verbose=0)[0][0]
            # Inverse-scale using the TARGET (mem) feature range
            target_idx = FEATURES.index(TARGET)
            pred = pred_scaled * (mx[target_idx] - mn[target_idx]) + mn[target_idx]
            pred = float(max(0.0, min(100.0, pred)))
            print(f'MV LSTM prediction for {hostname}: {pred:.2f} (model: {best_file})')
            return pred
        except Exception as e:
            print(f'[MV PREDICT] skipping {best_file}: {e}')
            continue
    return None


class MVLSTMTrainingManager:
    """Background trainer that keeps multivariate models fresh per host."""

    def __init__(self):
        self._hosts = {}

    def start_training(self, hostname):
        if hostname in self._hosts and self._hosts[hostname].is_alive():
            return
        t = threading.Thread(target=self._train_loop, args=(hostname,), daemon=True)
        t.start()
        self._hosts[hostname] = t

    def _train_loop(self, hostname):
        while True:
            try:
                train_lstm_model_mv(hostname, steps_ahead=2)
            except Exception as e:
                print(f'[MV TRAIN LOOP ERROR] {hostname}: {e}')
            sleep(TRAIN_INTERVAL_S)

    def stop_training(self):
        # Daemon threads: best-effort (they die with the process)
        pass


mv_manager = MVLSTMTrainingManager()
