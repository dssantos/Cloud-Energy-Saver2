#coding: utf-8
"""Offline evaluation harness for the multivariate LSTM.

Compares the CURRENT production approach (predict_mv.py) against candidate
improvements, measuring error in REAL units (mem %) on a temporal holdout.
Nothing here touches production code; it only reads workload_mv/{host}.csv.

Metric: predict mem 6 steps (3 min) ahead. Report test MAE/RMSE in % points,
always compared against the persistence baseline (predict = current mem).
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
    from numpy import array
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
    import keras.optimizers

MV_DIR = 'workload_mv'
TARGET = 'mem'
STEPS_AHEAD = 6
SEED = 7
RECEIVE_DECAY_S = 300.0   # exp decay for recency features (5 min)
LOADAVG_CLIP = 16.0       # clip loadavg (beyond ~2x cores is overload noise)

# Fixed, sensible hyperparameters so differences reflect the APPROACH, not luck.
HP = dict(n_steps=20, units=128, epochs=120, batch=32, dropout=0.2, lr=0.003)


def split_segments(df, gap_s=240):
    df = df.sort_index()
    if len(df) < 2:
        return [df] if len(df) else []
    dt = df.index.to_series().diff().dt.total_seconds().fillna(0)
    group = (dt > gap_s).cumsum()
    return [seg for _, seg in df.groupby(group)]


def feature_current(df):
    """CURRENT production feature set (as-is, incl. swap + sentinel + raw loadavg)."""
    feats = ['mem', 'vms', 'secs_since_vm_created', 'secs_since_vm_deleted', 'swap', 'loadavg']
    return df[feats].astype('float64').values, feats


def feature_fixed(df):
    """Cleaned features: drop swap, decayed recency, log loadavg."""
    mem = df['mem'].astype('float64').values
    vms = df['vms'].astype('float64').values
    cre = df['secs_since_vm_created'].astype('float64').values
    dele = df['secs_since_vm_deleted'].astype('float64').values
    rec_c = np.where(cre >= 9998, 0.0, np.exp(-cre / RECEIVE_DECAY_S))   # 1.0 just after, ~0 later
    rec_d = np.where(dele >= 9998, 0.0, np.exp(-dele / RECEIVE_DECAY_S))
    load = np.log1p(df['loadavg'].astype('float64').clip(upper=LOADAVG_CLIP).values)
    feats = ['mem', 'vms', 'rec_create', 'rec_delete', 'loadavg']
    vals = np.column_stack([mem, vms, rec_c, rec_d, load])
    return vals, feats


def minmax_fit(vals, feats):
    mn = np.array([vals[:, i].min() for i in range(len(feats))])
    mx = np.array([vals[:, i].max() for i in range(len(feats))])
    mx = np.where(mx > mn, mx, mn + 1.0)
    return mn, mx


def scale(arr2d, mn, mx):
    return (arr2d - mn) / (mx - mn)


def build_windows(segments, n_steps, target_idx, mn, mx):
    Xs, ys = [], []
    for seg in segments:
        vals = scale(seg, mn, mx)
        for i in range(len(vals) - n_steps - STEPS_AHEAD + 1):
            Xs.append(vals[i:i + n_steps])
            ys.append(vals[i + n_steps + STEPS_AHEAD - 1, target_idx])
    return array(Xs), array(ys)


def make_model(feat_n, arch, n_steps):
    m = Sequential()
    if arch == 'lstm1':
        m.add(LSTM(HP['units'], input_shape=(n_steps, feat_n)))
    elif arch == 'bilstm':
        m.add(Bidirectional(LSTM(HP['units']), input_shape=(n_steps, feat_n)))
    elif arch == 'lstm2':
        m.add(LSTM(HP['units'], return_sequences=True, input_shape=(n_steps, feat_n)))
        m.add(Dropout(HP['dropout']))
        m.add(LSTM(HP['units'] // 2))
    elif arch == 'gru1':
        m.add(GRU(HP['units'], input_shape=(n_steps, feat_n)))
    m.add(Dropout(HP['dropout']))
    m.add(Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=HP['lr']), loss='mse')
    return m


def evaluate_host(hostname, feat_fn, arch, delta_target=False):
    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)

    df = pd.read_csv(f'{MV_DIR}/{hostname}.csv')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('time_stamp').sort_index()
    vals, feats = feat_fn(df)
    dff = pd.DataFrame(vals, index=df.index, columns=feats)
    # store raw mem for delta inverse + real-unit eval
    raw_mem = df['mem'].astype('float64').values

    segments_raw = [s for s in split_segments(dff) if len(s) >= HP['n_steps'] + STEPS_AHEAD]
    if not segments_raw:
        return None

    mn, mx = minmax_fit(vals, feats)
    target_idx = feats.index(TARGET)

    # Build windows keeping a parallel raw-mem target for real-unit scoring
    Xs, ys_scaled, ys_raw = [], [], []
    seg_list = []
    for seg in split_segments(dff):
        if len(seg) < HP['n_steps'] + STEPS_AHEAD:
            continue
        seg_vals = scale(seg.values, mn, mx)
        # align raw_mem slice to this segment's positions
        # (split_segments preserves order; rebuild raw index mapping)
        seg_list.append(seg)
    # Simpler: rebuild windows directly with positional raw_mem via global index
    # Use full continuous arrays (segments already split); iterate per segment:
    # map each segment back to its integer positions in df
    pos = 0
    seg_pos = []
    idx = dff.index
    # compute segment boundaries by index membership
    bounds = []
    cur = []
    prev = None
    # rebuild segments on the dff index to get integer ranges
    times = dff.index.to_series()
    dt = times.diff().dt.total_seconds().fillna(0)
    grp = (dt > 240).cumsum()
    for _, sub in dff.groupby(grp):
        bounds.append((sub.index[0], sub.index[-1], sub.index))

    Xs, ys_scaled, ys_raw = [], [], []
    raw_series = pd.Series(raw_mem, index=dff.index)
    for s_start, s_end, sub_idx in bounds:
        seg = dff.loc[sub_idx]
        seg_raw = raw_series.loc[sub_idx].values
        seg_vals = scale(seg.values, mn, mx)
        n = HP['n_steps']
        if len(seg) < n + STEPS_AHEAD:
            continue
        for i in range(len(seg) - n - STEPS_AHEAD + 1):
            Xs.append(seg_vals[i:i + n])
            t = i + n + STEPS_AHEAD - 1
            ys_scaled.append(seg_vals[t, target_idx])
            ys_raw.append(seg_raw[t])
    X = array(Xs); ys = array(ys_scaled); yr = array(ys_raw)

    # Temporal split: last 20% of windows = test (later in time)
    split = len(X) * 8 // 10
    Xtr, Xte = X[:split], X[split:]
    ytr_s, yte_s = ys[:split], ys[split:]
    yte_raw = yr[split:]

    # For delta target: transform train target to delta vs last input mem (raw units),
    # but keep evaluation in absolute mem via inverse on (scaled mem + delta).
    # Here we keep absolute target for all configs except explicit delta below.
    model = make_model(len(feats), arch, HP['n_steps'])
    hist = model.fit(Xtr, ytr_s, epochs=HP['epochs'], batch_size=HP['batch'],
                     verbose=0, validation_split=0.1)

    pred_s = model.predict(Xte, verbose=0).flatten()
    # inverse-scale to mem % using target feature range
    pred = pred_s * (mx[target_idx] - mn[target_idx]) + mn[target_idx]
    pred = np.clip(pred, 0.0, 100.0)

    mae = np.mean(np.abs(pred - yte_raw))
    rmse = np.sqrt(np.mean((pred - yte_raw) ** 2))
    # persistence baseline on the SAME test windows
    # persistence = mem at the last input timestep (absolute). Reconstruct from X (scaled mem col).
    last_mem_scaled = Xte[:, -1, target_idx]
    persist = last_mem_scaled * (mx[target_idx] - mn[target_idx]) + mn[target_idx]
    p_mae = np.mean(np.abs(persist - yte_raw))
    p_rmse = np.sqrt(np.mean((persist - yte_raw) ** 2))

    return dict(n_train=len(Xtr), n_test=len(Xte), mae=float(mae), rmse=float(rmse),
                persist_mae=float(p_mae), persist_rmse=float(p_rmse),
                last_val_loss=float(hist.history['val_loss'][-1]))


CONFIGS = [
    ('current      ', 'feature_current', 'lstm1'),
    ('feat-fix     ', 'feature_fixed', 'lstm1'),
    ('feat-fix+bilstm', 'feature_fixed', 'bilstm'),
    ('feat-fix+lstm2 ', 'feature_fixed', 'lstm2'),
    ('feat-fix+gru  ', 'feature_fixed', 'gru1'),
]

FEAT_FNS = {'feature_current': feature_current, 'feature_fixed': feature_fixed}


def main():
    hosts = ['compute1', 'compute2', 'compute3']
    print(f'HP={HP}  STEPS_AHEAD={STEPS_AHEAD}  decay={RECEIVE_DECAY_S}s  loadclip={LOADAVG_CLIP}\n')
    results = {}
    for label, ffn, arch in CONFIGS:
        print(f'### {label}  (feat={ffn}, arch={arch})')
        agg = {}
        per_host = {}
        for h in hosts:
            r = evaluate_host(h, FEAT_FNS[ffn], arch)
            per_host[h] = r
            if r is None:
                print(f'  {h}: no data')
                continue
            beat = r['persist_mae'] - r['mae']
            print(f'  {h}: MAE={r["mae"]:.2f}  persist={r["persist_mae"]:.2f}  '
                  f'(Δ={beat:+.2f})  RMSE={r["rmse"]:.2f}  n_test={r["n_test"]}')
            for k in ('mae', 'rmse', 'persist_mae'):
                agg.setdefault(k, []).append(r[k])
        if agg:
            print(f'  MEAN: MAE={np.mean(agg["mae"]):.2f}  persist={np.mean(agg["persist_mae"]):.2f}  '
                  f'(Δ={np.mean(agg["persist_mae"]) - np.mean(agg["mae"]):+.2f})  RMSE={np.mean(agg["rmse"]):.2f}')
        print()
        results[label.strip()] = per_host
    with open('eval_mv_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('Saved eval_mv_results.json')


if __name__ == '__main__':
    main()
