#!/bin/bash
# Run ONCE on a bwUniCluster login node to create the conda env in a workspace.
set -euo pipefail

WS=$(ws_find whisper 2>/dev/null || ws_allocate whisper 60)
echo "workspace: $WS"

# --- miniforge in the workspace (home quota is too small for envs) ---
if [ ! -d "$WS/miniforge3" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$WS/miniforge3"
fi
source "$WS/miniforge3/etc/profile.d/conda.sh"

conda create -y -n whisper python=3.11
conda activate whisper

# torch matching cluster CUDA (12.x). Adjust cu121/cu124 to the loaded module.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo
echo ">>> Done. Now authenticate for the gated Common Voice dataset:"
echo ">>>   huggingface-cli login"
echo ">>> (accept terms once at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 )"
