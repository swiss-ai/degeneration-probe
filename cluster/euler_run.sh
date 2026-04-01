#!/usr/bin/env bash
# =============================================================================
# SLURM job: full pipeline (fetch → generate → plot) on a GPU node
#
# Submit:
#   CONFIG_FILE=alpaca_qwen05b.yaml sbatch cluster/euler_run.sh
#
# The CONFIG_FILE must exist under configs/ in the project directory.
# Default runs the small Alpaca/Qwen-0.5B config.
# =============================================================================
#SBATCH --job-name=degen-probe
#SBATCH --output=/cluster/home/%u/degeneration-probe/logs/run_%j.out
#SBATCH --error=/cluster/home/%u/degeneration-probe/logs/run_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000        # 32 GB total — increase for larger models
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g         # request ≥20 GB VRAM; remove for any GPU

set -euo pipefail

module load eth_proxy             # internet access on compute nodes
module load stack/2024-06 gcc/12.2.0 python/3.11.6 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

PROJ_DIR="/cluster/home/${USER}/degeneration-probe"
cd "${PROJ_DIR}"

export HF_HOME="/cluster/scratch/${USER}/hf_cache"
mkdir -p "${PROJ_DIR}/logs"

# Config file to use (override at submit time with CONFIG_FILE=foo.yaml sbatch ...)
CONFIG_FILE="${CONFIG_FILE:-alpaca_qwen05b.yaml}"

echo "[job] config   = ${CONFIG_FILE}"
echo "[job] HF_HOME  = ${HF_HOME}"
echo "[job] GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python run.py --config "configs/${CONFIG_FILE}"

echo "[job] Done."
