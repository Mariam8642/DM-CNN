#!/bin/bash
set -e

cd /home/hep/ma2922/Project/DM-CNN

echo "Starting full Optuna hyperparameter search"
echo "Host: $(hostname)"
date

export PYTHONUNBUFFERED=1

apptainer exec --nv \
--bind /vols/sbn/uboone/darkTridents:/vols/sbn/uboone/darkTridents \
/vols/sbn/uboone/darkTridents/container/larcv2.sif \
python -u optuna_DM-CNN.py

echo "Full Optuna run completed"
date
