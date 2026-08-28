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
    from numpy import array, concatenate, where, exp, log1p, clip, sqrt, mean
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
FEATURES = ['mem', 'vms', 'rec_create', 'rec_delete', 'loadavg', 'mem_ma', 'mem_diff']

TARGET = 'mem'
TRAIN_INTERVAL_S = 600  # retrain every 10 min
MODEL_CACHE_TTL_S = 600  # model cache expiry (reload from disk when stale)
MAX_MODELS_PER_HOST = 20  # prune to top-N by val_loss
SCORE_WEIGHT = 0.1        # W in: effective = (1-W)*rmse + W*mean_error (blend toward runtime error)
SCOREBOARD_FILE = 'scoreboard.json'  # periodic dump of the runtime scoreboard
RE_EVAL_TOP_N = 5        # non-winner models to re-evaluate per cycle
RE_EVAL_WINDOWS = 30     # recent history windows used in re-evaluation

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
    - mem_ma: 2-step moving average of mem (short-term level).
    - mem_diff: 1-step first difference of mem (momentum/trend).
    """
    out = pd.DataFrame(index=df.index)
    mem = df['mem'].astype('float64')
    out['mem'] = mem
    out['vms'] = df['vms'].astype('float64')
    cre = df['secs_since_vm_created'].astype('float64').values
    dele = df['secs_since_vm_deleted'].astype('float64').values
    out['rec_create'] = where(cre >= 9998, 0.0, exp(-cre / RECENCY_DECAY_S))
    out['rec_delete'] = where(dele >= 9998, 0.0, exp(-dele / RECENCY_DECAY_S))
    out['loadavg'] = log1p(clip(df['loadavg'].astype('float64').values, 0, LOADAVG_CLIP))
    out['mem_ma'] = mem.rolling(window=2).mean()
    out['mem_diff'] = mem.diff(1)
    return out


def _hp():
    """Random hyperparameters (extended with stacked-LSTM support).

    num_layers/lstm_units2 come from the generic_predictor search space; a
    num_layers=1 draw reproduces the previous single-layer architecture.
    """
    return {
        'n_steps': choice([10, 15, 20, 30, 50]),
        'lstm_units': choice([64, 128, 256]),
        'lstm_units2': choice([64, 128, 256]),
        'num_layers': choice([1, 2]),
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
        units2 = hp['lstm_units2']
        num_layers = hp['num_layers']
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
        if num_layers == 2:
            model.add(LSTM(units, activation='relu', return_sequences=True,
                           input_shape=(n_steps, len(FEATURES))))
            model.add(Dropout(dropout))
            model.add(LSTM(units2, activation='relu'))
            model.add(Dropout(dropout))
        else:
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

        # Validation RMSE/MAE in mem% (unscaled absolute forecast). Used for
        # model selection and stored in the filename prefix.
        pred_delta = model.predict(Xval, verbose=0).flatten()
        pred_mem = va_now + pred_delta * (mx[mem_idx] - mn[mem_idx])
        val_rmse = float(sqrt(mean((pred_mem - va_fut) ** 2)))
        val_mae = float(mean(abs(pred_mem - va_fut)))
        print(f'[MV TRAIN] {hostname}: val RMSE={val_rmse:.3f} MAE={val_mae:.3f} (mem%)')

        fname = (f'{val_rmse:.3f}_'
                 f'{datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")}_'
                 f'{epochs}_{n_steps}_{units}_{units2}_{num_layers}_{steps_ahead}ahead.keras')
        model.save(f'{MODEL_DIR}/{hostname}/{fname}')
        with open(f'{MODEL_DIR}/{hostname}/{fname}.norm.json', 'w') as f:
            json.dump({'min': mn.tolist(), 'max': mx.tolist(), 'target': 'delta',
                       'hp': hp}, f)
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


def select_best_model_mv(hostname, effective_map=None):
    """Return filename of the best model by effective score, pruning to top-N by
    raw filename RMSE.

    effective_map: {filename: effective_score} (optional; falls back to raw RMSE).
    """
    cand = _list_models(hostname)
    if not cand:
        return None
    for _, worst in cand[MAX_MODELS_PER_HOST:]:
        for suffix in ('', '.norm.json'):
            try:
                os.remove(f'{MODEL_DIR}/{hostname}/{worst}{suffix}')
            except OSError:
                pass
    keep = cand[:MAX_MODELS_PER_HOST]
    if effective_map:
        keep = sorted(keep, key=lambda p: effective_map.get(p[1], p[0]))
    return keep[0][1]


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


def _recent_error(hostname, model, mn, mx, n_steps, steps_ahead=2, max_windows=30):
    """Walk-forward mean error of `model` over the most recent history windows.

    Uses the saved workload_mv history (real future values are known), so it can
    score a model's predictive ability without using it for the live decision.
    Returns (mean_error_mem%, n_windows) or None.
    """
    try:
        df = pd.read_csv(f'{MV_DIR}/{hostname}.csv')
        df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('time_stamp').sort_index()
        dfe = _engineer(df)[FEATURES].dropna()
        if len(dfe) < n_steps + steps_ahead:
            return None
        mem_idx = FEATURES.index(TARGET)
        vals = dfe[FEATURES].values.astype('float64')
        span = mx[mem_idx] - mn[mem_idx]
        errors = []
        total = len(vals)
        start = max(0, total - max_windows - n_steps - steps_ahead)
        for i in range(start, total - n_steps - steps_ahead + 1):
            win = vals[i:i + n_steps]
            last_mem = vals[i + n_steps - 1, mem_idx]
            fut = vals[i + n_steps + steps_ahead - 1, mem_idx]
            xs = _scale(win, mn, mx).reshape(1, n_steps, len(FEATURES))
            d = model.predict(xs, verbose=0)[0][0]
            pred = last_mem + d * span
            errors.append(abs(pred - fut))
        if not errors:
            return None
        return float(mean(errors)), len(errors)
    except Exception:
        return None


def lstm_mv(hostname, steps_ahead=2, actual=None):
    """Predict mem steps_ahead ahead using the best multivariate model.

    Returns the predicted mem (%), or None if no model / not enough data.
    Uses the manager's model cache to avoid reloading from disk each cycle.

    When `actual` (live mem%) is provided, records a runtime accuracy score for
    the chosen model; the scoreboard drives the next selection re-rank.
    """
    loaded = mv_manager.get_model(hostname)
    if loaded is None:
        return None
    model, mn, mx, n_steps, filename = loaded
    pred = _predict_with(hostname, model, mn, mx, n_steps, filename)
    if pred is not None and actual is not None and actual > 0:
        mv_manager.record_score(hostname, filename, abs(pred - actual))
    return pred


class MVLSTMTrainingManager:
    """Background trainer that keeps multivariate models fresh per host."""

    def __init__(self):
        self._hosts = {}
        self._cache = {}          # hostname -> (model, mn, mx, n_steps, filename, timestamp)
        self._cache_lock = threading.Lock()
        self._scores = {}         # hostname -> {filename: [sum_error, count]}  (live verifier predictions)
        self._re_eval = {}        # hostname -> {filename: recent_history_mean_error}
        self._scores_lock = threading.Lock()
        self._stop_event = threading.Event()

    def start_training(self, hostname):
        if hostname in self._hosts and self._hosts[hostname].is_alive():
            return
        t = threading.Thread(target=self._training_loop, args=(hostname,), daemon=True)
        t.start()
        self._hosts[hostname] = t

    def record_score(self, hostname, filename, error):
        """Accumulate one prediction error for (hostname, filename)."""
        with self._scores_lock:
            e = self._scores.setdefault(hostname, {}).setdefault(filename, [0.0, 0])
            e[0] += float(error)
            e[1] += 1
            mean = e[0] / e[1]
            n = e[1]
        print(f'[MV SCORE] {hostname}: {filename} err={error:.2f} mean={mean:.2f} (n={n})')
        self.dump_scoreboard()

    def reset_scores(self, hostname=None):
        """Clear the scoreboard (all hosts, or one host). Called at run start."""
        with self._scores_lock:
            if hostname is None:
                self._scores.clear()
                self._re_eval.clear()
            else:
                self._scores.pop(hostname, None)
                self._re_eval.pop(hostname, None)
        self.dump_scoreboard()

    def _effective_for(self, hostname, rmse, filename):
        """Ranking score for a model: live mean if it has verifier predictions,
        else its recent re-eval mean, else its raw RMSE (unscored)."""
        with self._scores_lock:
            entry = self._scores.get(hostname, {}).get(filename)
            recent = self._re_eval.get(hostname, {}).get(filename)
        if entry and entry[1] > 0:
            return (1.0 - SCORE_WEIGHT) * rmse + SCORE_WEIGHT * (entry[0] / entry[1])
        if recent is not None:
            return (1.0 - SCORE_WEIGHT) * rmse + SCORE_WEIGHT * recent
        return rmse

    def _effective_map(self, hostname):
        """{filename: effective_score} for all models of a host."""
        return {fn: self._effective_for(hostname, rmse, fn)
                for rmse, fn in _list_models(hostname)}

    def dump_scoreboard(self):
        """Write the current scoreboard (models + runtime scores) to SCOREBOARD_FILE."""
        try:
            hosts = set()
            if os.path.isdir(MODEL_DIR):
                hosts.update(d for d in os.listdir(MODEL_DIR)
                             if os.path.isdir(f'{MODEL_DIR}/{d}'))
            with self._scores_lock:
                hosts.update(self._scores.keys())
                hosts.update(self._re_eval.keys())
            payload = {'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       'score_weight': SCORE_WEIGHT,
                       'hosts': {}}
            for host in sorted(hosts):
                entries = []
                for rmse, filename in _list_models(host):
                    with self._scores_lock:
                        entry = self._scores.get(host, {}).get(filename)
                        recent = self._re_eval.get(host, {}).get(filename)
                    mean_e = entry[0] / entry[1] if entry and entry[1] > 0 else 0.0
                    n = entry[1] if entry else 0
                    entries.append({'rmse': round(rmse, 3),
                                    'mean_err': round(mean_e, 2),
                                    'n': n,
                                    'recent_err': round(recent, 2) if recent is not None else None,
                                    'effective': round(self._effective_for(host, rmse, filename), 3),
                                    'filename': filename})
                entries.sort(key=lambda x: x['effective'])
                payload['hosts'][host] = entries
            with open(SCOREBOARD_FILE, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def record_eval(self, hostname, filename, mean_error, n_windows):
        """Store a model's recent-history accuracy (does NOT touch the live n)."""
        with self._scores_lock:
            self._re_eval.setdefault(hostname, {})[filename] = float(mean_error)
        print(f'[MV RE-EVAL] {hostname}: {filename} recent_err={mean_error:.2f} '
              f'(n={n_windows}w)')
        self.dump_scoreboard()

    def re_evaluate_competitors(self, hostname):
        """Periodically re-score non-winner models against recent history so their
        scoreboard stays fresh even when they are not the live prediction model."""
        try:
            cand = _list_models(hostname)
            if not cand:
                return
            eff = self._effective_map(hostname)
            # rank by effective ascending; skip the winner
            ranked = sorted(cand, key=lambda p: eff.get(p[1], p[0]))
            competitors = ranked[1:1 + RE_EVAL_TOP_N]
            for rmse, filename in competitors:
                loaded = _load_model(hostname, filename)
                if loaded is None:
                    continue
                model, mn, mx, n_steps = loaded
                r = _recent_error(hostname, model, mn, mx, n_steps,
                                  steps_ahead=2, max_windows=RE_EVAL_WINDOWS)
                if r is not None:
                    self.record_eval(hostname, filename, r[0], r[1])
        except Exception as e:
            print(f'[MV RE-EVAL ERROR] {hostname}: {e}')

    def get_model(self, hostname):
        """Return cached (model, mn, mx, n_steps, filename) or reload from disk.

        Re-ranks using the runtime scoreboard on every call, but only reloads the
        model file when the winner changes (avoids re-loading every cycle).
        """
        filename = select_best_model_mv(hostname, self._effective_map(hostname))
        if not filename:
            return None
        with self._cache_lock:
            cached = self._cache.get(hostname)
            if cached and cached[4] == filename:
                return cached[:5]
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
                    with self._cache_lock:
                        self._cache.pop(hostname, None)  # new model competes next cycle
            except Exception as e:
                print(f'[MV TRAIN LOOP ERROR] {hostname}: {e}')
            try:
                self.re_evaluate_competitors(hostname)
            except Exception as e:
                print(f'[MV RE-EVAL LOOP ERROR] {hostname}: {e}')
            if self._stop_event.wait(TRAIN_INTERVAL_S):
                break

    def stop_training(self):
        self._stop_event.set()
        for t in list(self._hosts.values()):
            if t.is_alive():
                t.join(timeout=5)
        self._hosts.clear()


mv_manager = MVLSTMTrainingManager()
