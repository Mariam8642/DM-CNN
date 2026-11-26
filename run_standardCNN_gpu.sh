#!/bin/bash
set -euxo pipefail

cd /home/hep/ma2922/Project/DM-CNN

echo "Host: $(hostname)"
echo "Start time: $(date)"

export PYTHONPATH=/home/hep/ma2922/Project/DM-CNN:${PYTHONPATH:-}

command -v apptainer
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

    python -c "
import torch
print(\"CUDA:\", torch.cuda.is_available())
print(\"GPUs:\", torch.cuda.device_count())
print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\")
"

    python -u standardCNN.py
  '

echo "Finished: $(date)"