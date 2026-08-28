#coding: utf-8
"""Reconstrói o placar em tempo de execução do LSTM multivariado.

O placar (erro médio acumulado por modelo) vive em memória no processo do
orchestrator e não é persistido. Este módulo reconstrói o estado atual a partir
de duas fontes legíveis de fora do processo:

  - models_mv/{host}/*.keras  -> RMSE de validação (prefixo do nome do arquivo);
  - experiment_run.log        -> linhas '[MV SCORE] ...' gravadas a cada
                                 verificação (erro |pred - real| acumulado).

A seleção efetiva usa effective = RMSE + SCORE_WEIGHT * erro_médio (menor vence).
"""

import os
import re
import json

MODEL_DIR = 'models_mv'
LOG_FILE = 'experiment_run.log'
SCOREBOARD_FILE = 'scoreboard.json'
SCORE_WEIGHT = 0.2
TOP_N = 5  # quantos modelos listar por host no placar

_SCORE_RE = re.compile(r'\[MV SCORE\] (\S+): (\S+) err=[\d.]+ mean=([\d.]+) \(n=(\d+)\)')


def _list_models(host):
    """[(rmse, filename)] ordenado por RMSE, lendo o prefixo do nome do arquivo."""
    d = os.path.join(MODEL_DIR, host)
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if f.endswith('.keras'):
            try:
                out.append((float(f.split('_')[0]), f))
            except ValueError:
                continue
    out.sort()
    return out


def total_models():
    """Retorna {host: total de modelos .keras salvos em disco}."""
    out = {}
    if os.path.isdir(MODEL_DIR):
        for d in os.listdir(MODEL_DIR):
            if os.path.isdir(os.path.join(MODEL_DIR, d)):
                out[d] = len(_list_models(d))
    return out


def _parse_scores():
    """{(host, filename): (mean_err, n)} lendo as linhas [MV SCORE] do log atual."""
    scores = {}
    if not os.path.isfile(LOG_FILE):
        return scores
    with open(LOG_FILE, errors='ignore') as fh:
        for line in fh:
            m = _SCORE_RE.search(line)
            if m:
                host, fn, mean, n = m.groups()
                scores[(host, fn)] = (float(mean), int(n))
    return scores


def get(top_n=TOP_N):
    """Retorna {host: [ {rmse, mean_err, n, effective, filename}, ... ]}.

    Cada host lista apenas os `top_n` melhores (por `effective` crescente).
    Prefere o scoreboard.json gravado pelo orchestrator (dump autoritativo do
    estado em memória); se ele não existir, reconstrói a partir do log.
    """
    if os.path.isfile(SCOREBOARD_FILE):
        try:
            with open(SCOREBOARD_FILE) as f:
                hosts = json.load(f).get('hosts')
            if hosts is not None:
                return {h: es[:top_n] for h, es in hosts.items()}
        except Exception:
            pass
    return _get_from_log(top_n)


def _get_from_log(top_n=TOP_N):
    """Reconstrói o placar a partir do log (fallback quando não há dump)."""
    scores = _parse_scores()

    hosts = set()
    if os.path.isdir(MODEL_DIR):
        hosts.update(d for d in os.listdir(MODEL_DIR)
                     if os.path.isdir(os.path.join(MODEL_DIR, d)))
    hosts.update(h for h, _ in scores.keys())

    out = {}
    for host in sorted(hosts):
        entries = []
        for rmse, fn in _list_models(host):
            mean, n = scores.get((host, fn), (0.0, 0))
            eff = rmse if n <= 0 else (1.0 - SCORE_WEIGHT) * rmse + SCORE_WEIGHT * mean
            entries.append({
                'rmse': rmse,
                'mean_err': mean,
                'n': n,
                'recent_err': None,
                'effective': round(eff, 3),
                'filename': fn,
            })
        entries.sort(key=lambda e: e['effective'])
        out[host] = entries[:top_n]
    return out
