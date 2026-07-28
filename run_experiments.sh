#!/usr/bin/env bash
# Experimentos longos alternados para reduzir viés de deriva do ambiente.
# Sequência padrão: default -> lstm -> default -> lstm (intercalado).
#
# Configurável por variáveis de ambiente:
#   DUR_HOURS : horas por experimento (default 6; use 12 para runs mais longos)
#   MODELS    : sequência de modelos (default "default lstm default lstm")
#   NUM_VMS   : VMs por ciclo no instantiator (default 27)
#
# Limpa VMs entre runs (deleta, aguarda ~10s/instância, re-confere, repete).
# Safety: timeout -s INT (DUR_HOURS*3600 + 900s) caso o corte gracoso do --duration trave.
#
# Exemplo:
#   DUR_HOURS=12 bash run_experiments.sh
set -u
cd /home/danilo/dev/python/Cloud-Energy-Saver2
PY=.venv/bin/python

DUR_HOURS="${DUR_HOURS:-6}"
MODELS="${MODELS:-baseline default lstm}"
NUM_VMS="${NUM_VMS:-27}"
LIM_MAX="${LIM_MAX:-70}"
LIM_MED="${LIM_MED:-50}"
DUR_S=$($PY -c "print(int($DUR_HOURS * 3600))")
SAFETY_S=$($PY -c "print(int($DUR_S + 900))")   # 15 min de folga sobre --duration

# Timestamped log filename + symlink para convenience (tail -f experiment_run.log funciona)
TS=$(date +%Y%m%d_%H%M%S)
LOG="experiment_run_${TS}.log"
ln -sf "$LOG" experiment_run.log
# Redireciona toda a saída para o arquivo timestampado
exec > "$LOG" 2>&1

cleanup_vms() {
  echo "[cleanup] removendo VMs remanescentes..."
  $PY -c "
import time, instances
for attempt in range(1, 6):  # ate 5 passagens
    try:
        instances.get()
        n = instances.length
    except Exception as e:
        print('  error listando VMs:', e); break
    if n == 0:
        print(f'  pass {attempt}: nenhuma VM restante.'); break
    print(f'  pass {attempt}: {n} VMs restantes -> deletando...')
    try:
        instances.off(n)
    except Exception as e:
        print('  off error:', e)
    wait = 10 * n  # ~10s por instancia (delete assincrono)
    print(f'  aguardando {wait}s para o delete concluir...')
    time.sleep(wait)
print('[cleanup] concluido.')
" 2>&1 | grep -v "absl\|cuda\|cudart\|InitializeLog"
}

echo "=== START $(date) ==="
echo "DUR_HOURS=$DUR_HOURS  |  MODELS='$MODELS'  |  NUM_VMS=$NUM_VMS  |  LIM_MAX=$LIM_MAX LIM_MED=$LIM_MED"
echo "python: $($PY --version 2>&1)"

for MODEL in $MODELS; do
  echo ""
  echo "########## cleanup antes de MODEL=$MODEL ($(date)) ##########"
  cleanup_vms
  sleep 10
  echo ""
  echo "########## MODEL=$MODEL  --duration $DUR_HOURS ($DUR_S s) ($(date)) ##########"
  timeout -s INT "$SAFETY_S" $PY orchestrator.py \
      --model "$MODEL" --lim-max "$LIM_MAX" --lim-med "$LIM_MED" \
      --num-vms "$NUM_VMS" --duration "$DUR_HOURS"
  echo "########## MODEL=$MODEL finished exit=$? at $(date) ##########"
  sleep 15
done

echo ""
echo "=== ANALYZE (3-way: baseline vs default vs lstm) $(date) ==="
BEvents=$(ls -t events_baseline_*.json 2>/dev/null | head -1)
BCsv=$(ls -t cluster_workload_baseline_*.csv 2>/dev/null | head -1)
REvents=$(ls -t events_default_*.json 2>/dev/null | head -1)
RCsv=$(ls -t cluster_workload_default_*.csv 2>/dev/null | head -1)
LEvents=$(ls -t events_lstm_*.json 2>/dev/null | head -1)
LCsv=$(ls -t cluster_workload_lstm_*.csv 2>/dev/null | head -1)
echo "baseline: $BEvents  /  $BCsv"
echo "reactive: $REvents  /  $RCsv"
echo "lstm    : $LEvents  /  $LCsv"
echo ""
if [[ -n "$BEvents" && -n "$BCsv" && -n "$REvents" && -n "$RCsv" && -n "$LEvents" && -n "$LCsv" ]]; then
  $PY analyze_metrics.py \
    --baseline-events "$BEvents" --baseline-csv "$BCsv" \
    --reactive-events "$REvents" --reactive-csv "$RCsv" \
    --lstm-events "$LEvents" --lstm-csv "$LCsv" \
    --lim-max "$LIM_MAX" --lim-med "$LIM_MED" 2>&1 | grep -v "absl\|cuda\|cudart\|InitializeLog"
else
  echo "ERROR: faltam arquivos de coleta para a análise."
fi
echo "=== END $(date) ==="
