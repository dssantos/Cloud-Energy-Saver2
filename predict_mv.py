#coding: utf-8
"""Multivariate LSTM pipeline (parallel to the univariate predict.py).

Reads workload_mv/{host}.csv, engineers cleaned features (decayed recency,
log-clipped loadavg, drop always-zero swap), trains an LSTM that predicts
the mem DELTA steps_ahead steps ahead, and saves models under models_mv/{host}/.

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
MODEL_CACHE_TTL_S = 600  # model cache expiry (reload from disk when stale)
MAX_MODELS_PER_HOST = 20  # prune to top-N by val_loss

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
    """Random hyperparameters (same search space as the univariate model)."""
    return {
        'n_steps': choice([10, 15, 20, 30, 50]),
        'lstm_units': choice([64, 128, 256]),
        'epochs': choice([100, 200, 300]),
        'batch_size': choice([8, 16, 32, 64]),
        'dropout': choice([0.1, 0.2, 0.5]),
        'learning_rate': choice([0.001, 0.002, 0.003, 0.005]),
    }


def _split_segments(df, gap_s=240):
    """Split a datetime-indexed df into continuous segments (gap > gap_s)."""
    df = df.sort_index()
    if len(df) < 2:
        return [df] if len(df) else []
    dt = df.index.to_series().diff().dt.total_seconds().fillna(0)
    group = (dt > gap_s).cumsum()
    return [seg for _, seg in df.groupby(group)]


def _fit_minmax(train_X):
    """Fit per-feature min/max on the TRAIN split only (no lookahead)."""
    mn = train_X.min(axis=0)
    mx = train_X.max(axis=0)
    mx = where(mx > mn, mx, mn + 1.0)
    return mn, mx


def _scale(arr2d, mn, mx):
    """Scale a 2D array (samples x features) to [0,1]."""
    return (arr2d - mn) / (mx - mn)


def _loss_to_str(val_loss):
    """Format a float loss into the zero-padded filename prefix (e.g. 000.012345)."""
    s = str(val_loss)
    head, _, tail = s.partition('.')
    return '.'.join([head.zfill(3), tail[:6]])


def train_lstm_model_mv(hostname, steps_ahead=2):
    """Train one multivariate LSTM (delta target) for a host and save it."""
    os.makedirs(f'{MODEL_DIR}/{hostname}', exist_ok=True)
    try:
        df = pd.read_csv(f'{MV_DIR}/{hostname}.csv')
        df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('time_stamp').sort_index()
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
        mem_idx = FEATURES.index(TARGET)

        segments = [s for s in _split_segments(dfe) if len(s) >= n_steps + steps_ahead]
        if not segments:
            print(f'[MV TRAIN] {hostname}: sem segmentos suficientes ({len(dfe)} amostras).')
            return None

        # Build raw windows (unscaled): X, mem at last input step, mem at target step
        raw_X, raw_mem_now, raw_mem_fut = [], [], []
        for seg in segments:
            vals = seg[FEATURES].values.astype('float64')
            for i in range(len(vals) - n_steps - steps_ahead + 1):
                raw_X.append(vals[i:i + n_steps])
                raw_mem_now.append(vals[i + n_steps - 1, mem_idx])
                raw_mem_fut.append(vals[i + n_steps + steps_ahead - 1, mem_idx])
        if not raw_X:
            print(f'[MV TRAIN] {hostname}: sem janelas após split.')
            return None

        # Temporal 80/20 split before scaling, so the scaler is fit on train only
        split = len(raw_X) * 8 // 10
        tr_X = array(raw_X[:split])
        tr_now = array(raw_mem_now[:split])
        tr_fut = array(raw_mem_fut[:split])
        va_X = array(raw_X[split:])
        va_now = array(raw_mem_now[split:])
        va_fut = array(raw_mem_fut[split:])

        mn, mx = _fit_minmax(tr_X.reshape(-1, len(FEATURES)))

        def _to_delta(Xs, nows, futs):
            Xs_s = _scale(Xs, mn, mx)
            now_s = (nows - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
            fut_s = (futs - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
            return Xs_s, fut_s - now_s  # delta target in scaled space

        Xtr, ytr = _to_delta(tr_X, tr_now, tr_fut)
        Xval, yval = _to_delta(va_X, va_now, va_fut)

        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(n_steps, len(FEATURES))))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, min_delta=1e-6)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)

        print(f'[MV TRAIN] {hostname}: X={Xtr.shape} | hp={hp} | target=delta')
        history = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch,
                            verbose=0, validation_data=(Xval, yval),
                            callbacks=[es, rlr])
        val_loss = float(min(history.history['val_loss']))
        print(f'[MV TRAIN] {hostname}: val_loss={val_loss:.6f} (best; stopped @ {len(history.history["val_loss"])})')

        fname = f'{_loss_to_str(val_loss)}_{datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")}_{epochs}_{n_steps}_{units}_{steps_ahead}ahead.keras'
        model.save(f'{MODEL_DIR}/{hostname}/{fname}')
        with open(f'{MODEL_DIR}/{hostname}/{fname}.norm.json', 'w') as f:
            json.dump({'min': mn.tolist(), 'max': mx.tolist(), 'target': 'delta'}, f)
        print(f'[MV MODEL SAVED] {hostname}: {fname}')
        return fname
    except Exception as e:
        print(f'[MV TRAIN ERROR] {hostname}: {e}')
        return None


def _list_models(hostname):
    """Return [(val_loss, filename)] for all .keras files, sorted ascending."""
    d = f'{MODEL_DIR}/{hostname}'
    if not os.path.isdir(d):
        return []
    cand = []
    for f in os.listdir(d):
        if f.endswith('.keras'):
            try:
                cand.append((float(f.split('_')[0]), f))
            except ValueError:
                continue
    cand.sort(key=lambda x: x[0])
    return cand


def select_best_model_mv(hostname):
    """Return filename of the model with lowest val_loss, pruning to the top-N."""
    cand = _list_models(hostname)
    if not cand:
        return None
    for _, worst in cand[MAX_MODELS_PER_HOST:]:
        for suffix in ('', '.norm.json'):
            try:
                os.remove(f'{MODEL_DIR}/{hostname}/{worst}{suffix}')
            except OSError:
                pass
    return cand[0][1]


def _load_model(hostname, filename):
    """Load a model + its normalization params + n_steps. Returns tuple or None."""
    try:
        parts = filename.replace('.keras', '').split('_')
        n_steps = int(parts[3])
        with open(f'{MODEL_DIR}/{hostname}/{filename}.norm.json') as f:
            nm = json.load(f)
        mn = array(nm['min'])
        mx = array(nm['max'])
        if len(mn) != len(FEATURES):
            return None
        model = load_model(f'{MODEL_DIR}/{hostname}/{filename}', compile=False)
        return model, mn, mx, n_steps
    except Exception:
        return None


def _predict_with(hostname, model, mn, mx, n_steps, filename):
    """Predict the next mem using a specific loaded model (delta + last mem)."""
    try:
        df = pd.read_csv(f'{MV_DIR}/{hostname}.csv')
        df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('time_stamp').sort_index()
        dfe = _engineer(df)[FEATURES].dropna()
        if len(dfe) < n_steps:
            return None
        mem_idx = FEATURES.index(TARGET)
        window_raw = dfe[FEATURES].values.astype('float64')[-n_steps:]
        window = _scale(window_raw, mn, mx)
        last_mem_scaled = (window_raw[-1, mem_idx] - mn[mem_idx]) / (mx[mem_idx] - mn[mem_idx])
        x_input = window.reshape((1, n_steps, len(FEATURES)))
        pred_delta_scaled = model.predict(x_input, verbose=0)[0][0]
        pred_mem_scaled = last_mem_scaled + pred_delta_scaled
        pred = pred_mem_scaled * (mx[mem_idx] - mn[mem_idx]) + mn[mem_idx]
        pred = float(max(0.0, min(100.0, pred)))
        print(f'MV LSTM prediction for {hostname}: {pred:.2f} (model: {filename})')
        return pred
    except Exception as e:
        print(f'[MV PREDICT ERROR] {hostname}: {e}')
        return None


def lstm_mv(hostname, steps_ahead=2):
    """Predict mem steps_ahead ahead using the best multivariate model.

    Returns the predicted mem (%), or None if no model / not enough data.
    Uses the manager's model cache to avoid reloading from disk each cycle.
    """
    loaded = mv_manager.get_model(hostname)
    if loaded is None:
        return None
    model, mn, mx, n_steps, filename = loaded
    return _predict_with(hostname, model, mn, mx, n_steps, filename)


class MVLSTMTrainingManager:
    """Background trainer that keeps multivariate models fresh per host."""

    def __init__(self):
        self._hosts = {}
        self._cache = {}          # hostname -> (model, mn, mx, n_steps, filename, timestamp)
        self._cache_lock = threading.Lock()
        self._stop_event = threading.Event()

    def start_training(self, hostname):
        if hostname in self._hosts and self._hosts[hostname].is_alive():
            return
        t = threading.Thread(target=self._training_loop, args=(hostname,), daemon=True)
        t.start()
        self._hosts[hostname] = t

    def get_model(self, hostname):
        """Return cached (model, mn, mx, n_steps, filename) or reload from disk."""
        with self._cache_lock:
            if hostname in self._cache:
                model, mn, mx, n_steps, filename, ts = self._cache[hostname]
                if time.time() - ts < MODEL_CACHE_TTL_S:
                    return model, mn, mx, n_steps, filename
                del self._cache[hostname]

        filename = select_best_model_mv(hostname)
        if not filename:
            return None
        loaded = _load_model(hostname, filename)
        if loaded is None:
            return None
        model, mn, mx, n_steps = loaded
        with self._cache_lock:
            self._cache[hostname] = (model, mn, mx, n_steps, filename, time.time())
        return model, mn, mx, n_steps, filename

    def _training_loop(self, hostname):
        while not self._stop_event.is_set():
            try:
                filename = train_lstm_model_mv(hostname, steps_ahead=2)
                if filename:
                    self._validate_and_update(hostname, filename)
            except Exception as e:
                print(f'[MV TRAIN LOOP ERROR] {hostname}: {e}')
            if self._stop_event.wait(TRAIN_INTERVAL_S):
                break

    def _validate_and_update(self, hostname, filename):
        """Validate the JUST-TRAINED model against the current RAM; if error
        < 5pp, pull its recorded loss toward the real error (0.7*old + 0.3*error)."""
        try:
            import ram_usage
            loaded = _load_model(hostname, filename)
            if loaded is None:
                return
            model, mn, mx, n_steps = loaded
            pred = _predict_with(hostname, model, mn, mx, n_steps, filename)
            actual = ram_usage.get(hostname)
            if pred is None or actual <= 0:
                return
            error = abs(pred - actual)
            if error < 5:
                old_loss = float(filename.split('_')[0])
                new_loss = old_loss * 0.7 + error * 0.3
                self._rename_loss(hostname, filename, new_loss)
            print(f'[MV PREDICT CHECK] {hostname}: pred={pred:.2f} actual={actual:.2f} error={error:.2f}')
        except Exception as e:
            print(f'[MV PREDICT CHECK ERROR] {hostname}: {e}')

    def _rename_loss(self, hostname, old_file, new_loss):
        parts = old_file.replace('.keras', '').split('_')
        new_file = f'{_loss_to_str(new_loss)}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}ahead.keras'
        for suffix in ('', '.norm.json'):
            try:
                os.rename(f'{MODEL_DIR}/{hostname}/{old_file}{suffix}',
                          f'{MODEL_DIR}/{hostname}/{new_file}{suffix}')
            except OSError:
                pass
        with self._cache_lock:
            self._cache.pop(hostname, None)
        print(f'[MV MODEL UPDATE] {hostname}: {old_file} -> {new_file} (loss {new_loss:.6f})')

    def stop_training(self):
        self._stop_event.set()
        for t in list(self._hosts.values()):
            if t.is_alive():
                t.join(timeout=5)
        self._hosts.clear()


mv_manager = MVLSTMTrainingManager()
