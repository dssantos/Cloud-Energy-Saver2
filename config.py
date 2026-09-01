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
STEPS_AHEAD = 6  # 3 min à frente (cadência de coleta de 30s; era 2 = 1 min)
MAX_VMS_PER_HOST = 10  # host "cheio" com >= N VMs; usado na recuperação por falha de alocação

# --- SLA ---
SLA_TIMEOUT = 120  # segundos (também definido em instances.py)

# --- Análise / coleta ---
VALIDATION_WINDOW_MIN = 15        # janela para validar se um wake foi necessário
CLUSTER_SAMPLE_INTERVAL_S = 10    # cadência do logger de cluster_workload (era 20; antecipação ~0.8 min pede mais resolução)

# --- Ociosidade ---
IDLE_DELETE_RECENCY_S = 180          # se VMs foram deletadas do host há menos disso, é ocioso genuíno -> pode desligar

# --- Tendência do nº de VMs (fase create/delete do instantiator) ---
VM_TREND_WINDOW_S = 300              # janela para medir se o nº de VMs está subindo/descendo
VM_TREND_MIN_DELTA = 1               # variação mínima (em VMs) para considerar tendência de subida

# --- SLA ---
SLA_RAM_MARGIN_PCT = 10            # SLA #1 (ram_over_threshold): host > lim_max * (1 + this/100). Ex.: lim_max 80 -> 88%
