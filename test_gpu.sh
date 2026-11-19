#!/bin/bash
set -euxo pipefail

apptainer exec --nv \
  --bind /vols:/vols \
  /vols/sbn/uboone/darkTridents/container/larcv2.sif \
  python -u -c '
import socket
import torch
print("Host:", socket.gethostname())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'
