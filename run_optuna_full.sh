#!/bin/bash
# set -e
set -euxo pipefail

cd /home/hep/ma2922/Project/DM-CNN

echo "Host: $(hostname)"
echo "Start time: $(date)"

export PYTHONPATH=/home/hep/ma2922/Project/DM-CNN:${PYTHONPATH:-}
export EARLY_STOPPING_PATIENCE=5
export MIN_DELTA=0.001
export EPOCHS=50

python train_optuna_early_stopping.py

command -v apptainer
test -d /home/hep/ma2922/Project/DM-CNN
test -f /vols/sbn/uboone/darkTridents/container/larcv2.sif

echo "Starting CNN training: $(date)"

apptainer exec --nv \
  --bind /vols/sbn/uboone/darkTridents:/vols/sbn/uboone/darkTridents \
  --bind /home/hep/ma2922/Project/DM-CNN:/home/hep/ma2922/Project/DM-CNN \
  /vols/sbn/uboone/darkTridents/container/larcv2.sif \
  bash -c '
    set -eux
    export PYTHONPATH=/home/hep/ma2922/Project/DM-CNN:${PYTHONPATH:-}
    cd /home/hep/ma2922/Project/DM-CNN
    python -u optuna_DM-CNN.py
  '

echo "Finished: $(date)"

