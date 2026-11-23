#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/hep/ma2922/Project/DM-CNN
CONTAINER=/vols/sbn/uboone/darkTridents/container/larcv2.sif

cd "$PROJECT_DIR"

echo "Host: $(hostname)"
echo "Start time: $(date)"

command -v apptainer
test -d "$PROJECT_DIR"
test -f "$CONTAINER"

# Full-run configuration.
export QUICK=0
export N_TRIALS=1
export EPOCHS=50
export EARLY_STOPPING_PATIENCE=5
export MIN_DELTA=0.001

echo "Starting CNN training inside Apptainer: $(date)"

exec apptainer exec --nv \
  --bind /vols/sbn/uboone/darkTridents:/vols/sbn/uboone/darkTridents \
  --bind "$PROJECT_DIR":"$PROJECT_DIR" \
  "$CONTAINER" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH=/home/hep/ma2922/Project/DM-CNN:${PYTHONPATH:-}
    export QUICK=0
    export N_TRIALS=1
    export EPOCHS=50
    export EARLY_STOPPING_PATIENCE=5
    export MIN_DELTA=0.001
    cd /home/hep/ma2922/Project/DM-CNN
    exec python -u optuna_DM-CNN.py
  '


