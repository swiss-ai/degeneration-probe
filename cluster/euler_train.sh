#!/usr/bin/env bash
# =============================================================================
# SLURM job: train a degeneration probe on a GPU node
#
# Submit (default config):
#   sbatch cluster/euler_train.sh
#
# Override Hydra parameters:
#   HYDRA_OVERRIDES="model=apertus num_epochs=20" sbatch cluster/euler_train.sh
#
# Override training data:
#   HYDRA_OVERRIDES="train_data=/path/to/data.jsonl model=apertus" sbatch cluster/euler_train.sh
# =============================================================================
#SBATCH --job-name=degen-train
#SBATCH --output=/cluster/home/%u/degeneration-probe/logs/train_%j.out
#SBATCH --error=/cluster/home/%u/degeneration-probe/logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000        # 32 GB total
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g         # 40 GB VRAM for 8B model + activations

set -euo pipefail

module load eth_proxy             # internet access on compute nodes
module load stack/2024-06 gcc/12.2.0 python/3.11.6 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

PROJ_DIR="/cluster/home/${USER}/degeneration-probe"
cd "${PROJ_DIR}"

export HF_HOME="/cluster/scratch/${USER}/hf_cache"
mkdir -p "${PROJ_DIR}/logs"

# Hydra overrides (pass via environment variable at submit time)
HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"

echo "[job] HF_HOME  = ${HF_HOME}"
echo "[job] overrides = ${HYDRA_OVERRIDES}"
echo "[job] GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python scripts/train.py ${HYDRA_OVERRIDES}

echo "[job] Done."
