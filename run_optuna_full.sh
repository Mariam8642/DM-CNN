#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/hep/ma2922/Project/DM-CNN
CONTAINER=/vols/sbn/uboone/darkTridents/container/larcv2.sif


#arguments
EPOCHS="${1:-15}"
EARLY_STOPPING_PATIENCE="${2:-3}"
MIN_DELTA="${3:-0.001}"

export QUICK=0
export EPOCHS
export EARLY_STOPPING_PATIENCE
export MIN_DELTA


cd "$PROJECT_DIR"

echo "hostname: $(hostname)"
echo "Start time: $(date)"
echo "Running with EPOCHS=${EPOCHS}, EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}, MIN_DELTA=${MIN_DELTA}"

command -v apptainer
test -d "$PROJECT_DIR"
test -f "$CONTAINER"

exec apptainer exec --nv \
  --bind /vols/sbn/uboone/darkTridents:/vols/sbn/uboone/darkTridents \
  --bind "$PROJECT_DIR":"$PROJECT_DIR" \
  "$CONTAINER" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/home/hep/ma2922/Project/DM-CNN:${PYTHONPATH:-}
    export QUICK=0
    export EPOCHS=$EPOCHS
    export EARLY_STOPPING_PATIENCE=$EARLY_STOPPING_PATIENCE
    export MIN_DELTA=$MIN_DELTA
    cd /home/hep/ma2922/Project/DM-CNN
    exec python -u optuna_DM-CNN.py
  '


