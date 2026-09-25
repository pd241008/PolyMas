#!/usr/bin/env bash
# Guarded System B (Mamba) trainer.
#
# The host GPU (RTX 4050 laptop, WSL2/dxg) resets under sustained load, which
# kills long training runs mid-epoch. This wrapper:
#   1. waits for the GPU to be visible before each attempt (never falls back
#      to CPU — a 25-hour CPU epoch silently resuming is worse than waiting);
#   2. reruns train_system_b.py with --resume until it completes, so each
#      driver reset costs at most one epoch (checkpoints are per-epoch);
#   3. optionally starts fresh by parking the previous checkpoint/report.
#
# Usage: scripts/train_system_b_guarded.sh
# Env overrides: EPOCHS, BATCH, DATA, OUT, DISEASES, MAX_ATTEMPTS, MAX_GPU_WAITS,
#                POLYMAS_RESULTS_DIR (results root; default stash/results — set
#                it to a fresh folder to never overwrite previous runs)
set -u
cd "$(dirname "$0")/.."

PY=services/ml-engine-python/.venv/bin/python
LOG=${LOG:-/tmp/system_b_guarded.log}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-16}
DATA=${DATA:-kmer5000_gwas}
OUT=${OUT:-kmer5000_gwas_out}
DISEASES=${DISEASES:-"RA SLE SJOGRENS AITD T1D VITILIGO MS"}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
MAX_GPU_WAITS=${MAX_GPU_WAITS:-30}
RESULTS_DIR=${POLYMAS_RESULTS_DIR:-stash/results}
export POLYMAS_RESULTS_DIR="$RESULTS_DIR"
REPORT="$RESULTS_DIR/sequence/$OUT/smoke_test_report.json"

# Fresh start: park any previous checkpoint so --resume cannot pick it up.
if [ "${FRESH:-1}" = "1" ]; then
  parked="$RESULTS_DIR/sequence/$OUT/superseded_$(date +%m%d_%H%M)"
  mkdir -p "$parked"
  for f in checkpoint.pt best_model.pt smoke_test_report.json; do
    [ -f "$RESULTS_DIR/sequence/$OUT/$f" ] && mv "$RESULTS_DIR/sequence/$OUT/$f" "$parked/"
  done
  echo "[guard] previous artifacts parked in $parked"
fi

: > "$LOG"
attempt=1
gpu_waits=0
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  if ! nvidia-smi >/dev/null 2>&1; then
    gpu_waits=$((gpu_waits + 1))
    if [ "$gpu_waits" -gt "$MAX_GPU_WAITS" ]; then
      echo "[guard] GPU unavailable for $MAX_GPU_WAITS waits — giving up."
      exit 2
    fi
    echo "[guard] $(date +%H:%M:%S) GPU not visible (wait $gpu_waits/$MAX_GPU_WAITS); retrying in 60s"
    sleep 60
    continue
  fi

  echo "[guard] $(date +%H:%M:%S) attempt $attempt/$MAX_ATTEMPTS (batch=$BATCH)"
  PYTHONPATH=services/ml-engine-python "$PY" scripts/train_system_b.py \
    --data-dir "$DATA" --out-dir "$OUT" --epochs "$EPOCHS" --batch-size "$BATCH" \
    --diseases $DISEASES --resume --require-gpu >> "$LOG" 2>&1
  rc=$?

  if grep -q "Smoke test done" "$LOG"; then
    echo "[guard] training complete after $attempt attempt(s). Report: $REPORT"
    exit 0
  fi
  echo "[guard] attempt $attempt exited rc=$rc without completing; resuming in 30s"
  sleep 30
  attempt=$((attempt + 1))
done

echo "[guard] giving up after $MAX_ATTEMPTS attempts — inspect $LOG"
exit 1
