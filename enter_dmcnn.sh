#!/bin/bash

# Enter the DM-CNN project directory
cd "$HOME/Project/DM-CNN" || exit 1

# Configure ROOT and LArCV2
source /app/root/bin/thisroot.sh
source /app/larcv2/configure.sh

# Load project-specific settings
source setup.sh

# Display the Python environment
echo "DM-CNN environment loaded"
python -c "import torch; print('PyTorch:', torch.__version__)"

