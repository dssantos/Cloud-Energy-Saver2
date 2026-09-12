#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera predicted_vs_real.png: RAM real de cada host do modo LSTM contra a RAM
prevista com 3 minutos de antecedencia (predictions/predictions_<host>.csv).

A linha real usa a serie completa de workload registrada enquanto o host esteve
ligado (cadencia de 30 s). A linha prevista existe apenas onde ha previsao, ou
seja, onde a janela de entrada estava contigua.

    python3 experiments/20260903/plot_predictions.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.grid": True, "grid.color": "#d9d8d2",
                     "grid.linewidth": 0.6, "grid.alpha": 0.8,
                     "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})

INK = "#0b0b0b"       # RAM real
PRED = "#2a78d6"      # RAM prevista
LIM_MAX = 70.0        # limiar superior de sobrecarga (lim_max)
HOSTS = ["compute1", "compute2", "compute3"]

fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
for ax, host in zip(axes, HOSTS):
    w = pd.read_csv(os.path.join(HERE, "workload", f"workload_{host}.csv"),
                    parse_dates=["time_stamp"]).sort_values("time_stamp")
    t0 = w["time_stamp"].min()
    mins = (w["time_stamp"] - t0).dt.total_seconds() / 60
    # quebra a linha real apenas onde nao ha coleta (host desligado)
    real = w["mem"].where(w["time_stamp"].diff() <= pd.Timedelta(seconds=90))
    ax.plot(mins, real, color=INK, linewidth=1.1, label="RAM real do host", zorder=3)

    p = pd.read_csv(os.path.join(HERE, "predictions", f"predictions_{host}.csv"),
                    parse_dates=["ts_alvo"]).sort_values("ts_alvo")
    mins_p = (p["ts_alvo"] - t0).dt.total_seconds() / 60
    prev = p["previsto"].where(p["ts_alvo"].diff() <= pd.Timedelta(minutes=5))
    ax.plot(mins_p, prev, color=PRED, linewidth=1.1, linestyle="--",
            label="RAM prevista (LSTM, 3 min a frente)", zorder=4)

    ax.axhline(LIM_MAX, color="#c0392b", linewidth=1.0, linestyle=":")
    ax.set_ylim(0, 100)
    ax.set_ylabel("RAM (%)")
    ax.set_title(f"{host} (melhor modelo em servico do placar)", color=PRED,
                 fontsize=10, loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

axes[0].text(0.99, 0.94, "Previsao 3 min a frente",
             transform=axes[0].transAxes, ha="right", va="top", fontsize=9, color=INK,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cfcec8"))
axes[-1].set_xlabel("Minutos desde o inicio do modo LSTM")
fig.suptitle("RAM prevista versus real por host - modo LSTM (run 20260903)",
             fontsize=12, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(HERE, "predicted_vs_real.png"), dpi=300, bbox_inches="tight")
print("ok predicted_vs_real.png")
