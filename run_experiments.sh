#!/usr/bin/env bash
# Roda default e lstm por ~1h cada (--duration 1, agora funcional) e ao final analisa.
# Limpa VMs entre runs (loop: deleta, aguarda ~10s/instancia, re-verifica, repete).
# Safety: timeout -s INT 4500 caso o corte gracioso trave.
set -u
cd /home/danilo/dev/python/Cloud-Energy-Saver2
PY=.venv/bin/python

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
echo "python: $($PY --version 2>&1)"

for MODEL in default lstm; do
  echo ""
  echo "########## cleanup antes de MODEL=$MODEL ($(date)) ##########"
  cleanup_vms
  sleep 10
  echo ""
  echo "########## MODEL=$MODEL  ($(date))  --duration 1 (3600s) ##########"
  timeout -s INT 4500 $PY orchestrator.py --model "$MODEL" --lim-max 70 --lim-med 50 --num-vms 27 --duration 1
  echo "########## MODEL=$MODEL finished exit=$? at $(date) ##########"
  sleep 15
done

echo ""
echo "=== ANALYZE $(date) ==="
REvents=$(ls -t events_default_*.json 2>/dev/null | head -1)
RCsv=$(ls -t cluster_workload_default_*.csv 2>/dev/null | head -1)
LEvents=$(ls -t events_lstm_*.json 2>/dev/null | head -1)
LCsv=$(ls -t cluster_workload_lstm_*.csv 2>/dev/null | head -1)
echo "reactive: $REvents  /  $RCsv"
echo "lstm    : $LEvents  /  $LCsv"
if [[ -n "$REvents" && -n "$RCsv" && -n "$LEvents" && -n "$LCsv" ]]; then
  $PY analyze_metrics.py \
    --reactive-events "$REvents" --reactive-csv "$RCsv" \
    --lstm-events "$LEvents" --lstm-csv "$LCsv" \
    --lim-max 70 --lim-med 50 2>&1 | grep -v "absl\|cuda\|cudart\|InitializeLog"
else
  echo "ERROR: faltam arquivos de coleta para a análise."
fi
echo "=== END $(date) ==="
