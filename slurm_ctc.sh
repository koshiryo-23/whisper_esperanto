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

# big downloads -> workspace; keep HF token in its default home path (see slurm_train.sh)
export HF_HUB_CACHE="$WS/hf/hub"
export HF_DATASETS_CACHE="$WS/hf/datasets"
mkdir -p logs "$WS/hf"

module load devel/cuda/12.4 2>/dev/null || module load devel/cuda || true
source "$WS/miniforge3/etc/profile.d/conda.sh"
conda activate whisper

# --- fail fast if not authenticated (Common Voice is gated) ---
python - <<'PY' || { echo "ERROR: no Hugging Face token. On a login node run:  huggingface-cli login"; echo "then accept terms at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"; exit 1; }
from huggingface_hub import whoami
print("HF auth OK:", whoami()["name"])
PY

srun python train_ctc.py \
    --model facebook/wav2vec2-xls-r-300m \
    --output_dir "$WS/wav2vec2-eo" \
    --max_train_samples 10000 \
    --max_steps 2000 \
    --batch_size 16 \
    --eval_steps 500
