#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstroi, offline, as previsoes de RAM por host do modo LSTM do run 20260903.

Para cada host, carrega o melhor modelo em servico segundo o placar (scoreboard.json)
e o aplica ao workload registrado do proprio host (workload/workload_<host>.csv),
repetindo o mesmo caminho de predicao usado em execucao (janela deslizante, escalonamento
MinMax por feature, alvo delta somado a ultima memoria, resultado clipado em 0-100).

Roda a partir da raiz do Cloud-Energy-Saver2, com o ambiente virtual do projeto:

    .venv/bin/python experiments/20260903/predictions/predict_from_workload.py

Saida: predictions_<host>.csv, com colunas
    ts_emissao  instante da emissao (fim da janela de entrada)
    ts_alvo     instante futuro previsto (emissao + 3 min)
    previsto    RAM prevista (%)
    mem_real    RAM real observada em ts_alvo (pareamento nearest, tolerancia 30 s)
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(RUN_DIR, "..", ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)

import predict_mv as pm  # noqa: E402  (carrega TensorFlow e as constantes do CES2)

RUN_START = pd.Timestamp("2026-09-01 16:35:00")   # inicio do modo LSTM
RUN_END = pd.Timestamp("2026-09-02 06:40:00")     # fim do modo LSTM
GAP_S = pm.config.GAP_S
HORIZON_S = 180                                   # steps_ahead=6 x 30 s

# melhores modelos em servico por host, na ordem do placar (scoreboard.json)
BEST = {
    "compute1": "4.369_20260901223906_300_50_256_256_1_6ahead.keras",
    "compute2": "3.979_20260901021354_300_20_64_64_1_6ahead.keras",
    "compute3": "4.312_20260901035837_200_30_64_128_1_6ahead.keras",
}


def predictions_for(host, filename):
    """Aplica o modelo ao workload do host e devolve o DataFrame de previsoes."""
    pm.MODEL_DIR = os.path.join(RUN_DIR, "models")
    loaded = pm._load_model(host, filename)
    if loaded is None:
        raise SystemExit(f"falha ao carregar o modelo de {host}: {filename}")
    model, mn, mx, n_steps = loaded

    df = pd.read_csv(os.path.join(RUN_DIR, "workload", f"workload_{host}.csv"))
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.set_index("time_stamp").sort_index()

    dfe = pm._engineer(df)[pm.FEATURES].dropna()
    mem_idx = pm.FEATURES.index(pm.TARGET)
    vals = dfe[pm.FEATURES].values.astype("float64")
    ts = dfe.index

    # janelas contiguas (mesmo criterio de gap do preditor em execucao)
    idxs, X = [], []
    for i in range(n_steps - 1, len(dfe)):
        difs = np.diff(ts[i - n_steps + 1:i + 1].values).astype("timedelta64[s]").astype(np.int64)
        if (difs <= GAP_S).all():
            idxs.append(i)
            X.append(vals[i - n_steps + 1:i + 1])
    if not X:
        raise SystemExit(f"nenhuma janela contigua para {host}")
    X = np.array(X)

    Xs = (X - mn) / (mx - mn)
    last_mem_scaled = Xs[:, -1, mem_idx]
    pred_delta = model.predict(Xs, verbose=0).ravel()
    pred = (last_mem_scaled + pred_delta) * (mx[mem_idx] - mn[mem_idx]) + mn[mem_idx]
    pred = np.clip(pred, 0.0, 100.0)

    ts_emissao = ts[idxs]
    ts_alvo = ts_emissao + pd.Timedelta(seconds=HORIZON_S)
    serie = pd.Series(dfe[pm.TARGET].values, index=ts)
    real = serie.reindex(ts_alvo, method="nearest", tolerance=pd.Timedelta(seconds=30))

    out = pd.DataFrame({"ts_emissao": ts_emissao, "ts_alvo": ts_alvo,
                        "previsto": pred.round(3), "mem_real": real.round(3)})
    return out.dropna(subset=["mem_real"]), n_steps


def main():
    for host, filename in BEST.items():
        out, n_steps = predictions_for(host, filename)
        caminho = os.path.join(HERE, f"predictions_{host}.csv")
        out.to_csv(caminho, index=False)
        err = out["previsto"] - out["mem_real"]
        rmse = float(np.sqrt((err ** 2).mean()))
        print(f"{host}: {filename}")
        print(f"  n_steps={n_steps} | {len(out)} previsoes | "
              f"RMSE={rmse:.3f} p.p. | vies={err.mean():+.2f} p.p. | -> {os.path.basename(caminho)}")


if __name__ == "__main__":
    main()
