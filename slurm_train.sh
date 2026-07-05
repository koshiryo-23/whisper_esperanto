#!/bin/bash
#SBATCH --job-name=whisper-eo
#SBATCH --partition=gpu_a100_il       # full runs. Quick test: dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00               # dev queue max is short (~30min) -> lower this there
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Verify partitions/time limits on the cluster with:  sinfo -s   /   scontrol show partition
set -euo pipefail

# --- workspace (fast scratch; home quota is tiny) ---
WS=$(ws_find whisper 2>/dev/null || ws_allocate whisper 60)
echo "workspace: $WS"

# put ALL caches on the workspace, not home
export HF_HOME="$WS/hf"
export HF_DATASETS_CACHE="$WS/hf/datasets"
export TRANSFORMERS_CACHE="$WS/hf/transformers"
mkdir -p logs "$HF_HOME"

# --- environment ---
module load devel/cuda/12.4 2>/dev/null || module load devel/cuda || true
source "$WS/miniforge3/etc/profile.d/conda.sh"
conda activate whisper

# --- run ---
# Whisper Small + LoRA, Polish proxy token. Edit flags for full FT / other model.
srun python train_whisper.py \
    --model openai/whisper-small \
    --language polish \
    --use_lora \
    --output_dir "$WS/whisper-eo-small-lora" \
    --max_train_samples 10000 \
    --max_steps 1500 \
    --batch_size 16 \
    --eval_steps 500
