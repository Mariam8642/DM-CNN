#!/bin/bash
set -e

cd /home/hep/ma2922/Project/DM-CNN

echo "Starting CNN smoke test"
echo "Host: $(hostname)"
date

apptainer exec --nv \
--bind /vols/sbn/uboone/darkTridents:/vols/sbn/uboone/darkTridents \
/vols/sbn/uboone/darkTridents/container/larcv2.sif \
python uboone/train_DM-CNN_smoketest.py

echo "Smoke test completed"
date
