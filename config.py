#coding: utf-8
"""
Configurações centralizadas do CES2.

Constantes de decisão, coleta e análise. Centralizadas aqui para remover
números mágicos espalhados pelo código (verifier.py, orchestrator.py,
analyze_metrics.py) e servir de fonte única para a análise do TCC.
"""

import ast


def _load_n_hosts(default=3):
    """Lê registered.txt dinamicamente para obter o número de hosts."""
    try:
        with open("registered.txt", "r") as f:
            return len(ast.literal_eval(f.read()))
    except Exception:
        return default


# --- Ambiente ---
N_HOSTS = _load_n_hosts()

# --- Potência dos hosts (estimativa; fonte bibliográfica a definir) ---
P_IDLE = 65.0   # W - host ligado, sem carga
P_LOAD = 120.0  # W - host ligado, com carga
P_AVG = (P_IDLE + P_LOAD) / 2.0  # ~92.5 W (continuidade com o valor usado em analyze_events.py)

# --- Limiares de RAM (%) ---
# Defaults; CLI --lim-max / --lim-med sobrescrevem em tempo de execução.
LIM_MAX = 70.0
LIM_MED = 30.0

# --- Predição ---
STEPS_AHEAD = 6  # 3 min à frente (já é o default em predict.py / verifier.py)

# --- SLA ---
SLA_TIMEOUT = 120  # segundos (também definido em instances.py)

# --- Análise / coleta ---
VALIDATION_WINDOW_MIN = 15        # janela para validar se um wake foi necessário
CLUSTER_SAMPLE_INTERVAL_S = 10    # cadência do logger de cluster_workload (era 20; antecipação ~0.8 min pede mais resolução)

# --- Anti-flapping ---
EMERGENCY_COOLDOWN_SECONDS = 300   # após um wake emergencial, não desligar o host (também definido em verifier.py)
SHUTDOWN_COOLDOWN_SECONDS = 300    # após um shutdown, não re-acordar o host (evita religamento/flapping)
SHUTDOWN_FLAP_BLOCK_S = 60         # janela anti-flap: mesmo em emergency, não re-acordar host desligado há menos disso

# --- SLA ---
SLA_RAM_MARGIN_PCT = 10            # SLA #1 (ram_over_threshold): host > lim_max * (1 + this/100). Ex.: lim_max 80 -> 88%
