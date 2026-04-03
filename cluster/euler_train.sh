#!/usr/bin/env bash
# =============================================================================
# SLURM job: train a degeneration probe on a GPU node
#
# Submit (default config):
#   sbatch cluster/euler_train.sh
#
# Override config file:
#   TRAIN_CONFIG=configs/train/my_config.yaml sbatch cluster/euler_train.sh
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

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/default.yaml}"

echo "[job] HF_HOME  = ${HF_HOME}"
echo "[job] config   = ${TRAIN_CONFIG}"
echo "[job] GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python -m degeneration_probe train --config "${TRAIN_CONFIG}"

echo "[job] Done."
