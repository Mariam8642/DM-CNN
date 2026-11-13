#!/bin/bash
set -e

cd /home/hep/ma2922/Project/DM-CNN

echo "Host: $(hostname)"
echo "Start time: $(date)"

# Use this only if required by the cluster CUDA setup
# source /vols/software/cuda/setup.sh 10.2

export PYTHONPATH=/home/hep/ma2922/Project/DM-CNN:${PYTHONPATH:-}

echo "Checking GPU:"
nvidia-smi

echo "Checking PyTorch CUDA:"
apptainer exec --nv \
  /vols/sbn/uboone/darkTridents/container/larcv2.sif \
  python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

echo "Starting CNN training: $(date)"

apptainer exec --nv \
  --bind /vols/sbn/uboone/darkTridents:/vols/sbn/uboone/darkTridents \
  --bind /home/hep/ma2922/Project/DM-CNN:/home/hep/ma2922/Project/DM-CNN \
  /vols/sbn/uboone/darkTridents/container/larcv2.sif \
  bash -c '
    export PYTHONPATH=/home/hep/ma2922/Project/DM-CNN:${PYTHONPATH:-}
    cd /home/hep/ma2922/Project/DM-CNN
    python -u uboone/standardCNN.py
  '

echo "Finished: $(date)"
