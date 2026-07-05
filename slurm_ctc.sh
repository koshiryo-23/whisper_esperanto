#!/bin/bash
#SBATCH --job-name=ctc-eo
#SBATCH --partition=gpu_a100_il       # quick test: dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

WS=$(ws_find whisper 2>/dev/null || ws_allocate whisper 60)
echo "workspace: $WS"

export HF_HOME="$WS/hf"
export HF_DATASETS_CACHE="$WS/hf/datasets"
export TRANSFORMERS_CACHE="$WS/hf/transformers"
mkdir -p logs "$HF_HOME"

module load devel/cuda/12.4 2>/dev/null || module load devel/cuda || true
source "$WS/miniforge3/etc/profile.d/conda.sh"
conda activate whisper

srun python train_ctc.py \
    --model facebook/wav2vec2-xls-r-300m \
    --output_dir "$WS/wav2vec2-eo" \
    --max_train_samples 10000 \
    --max_steps 2000 \
    --batch_size 16 \
    --eval_steps 500
