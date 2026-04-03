#!/usr/bin/env bash
# =============================================================================
# SLURM job: generate labelled data on a GPU node
#
# Submit (default config):
#   sbatch cluster/euler_generate.sh
#
# Override config file:
#   GEN_CONFIG=configs/generate/aime_apertus8b.yaml sbatch cluster/euler_generate.sh
# =============================================================================
#SBATCH --job-name=degen-generate
#SBATCH --output=/cluster/home/%u/degeneration-probe/logs/generate_%j.out
#SBATCH --error=/cluster/home/%u/degeneration-probe/logs/generate_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000        # 32 GB total — increase for larger models
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g         # 40 GB VRAM for 8B models

set -euo pipefail

module load eth_proxy             # internet access on compute nodes
module load stack/2024-06 gcc/12.2.0 python/3.11.6 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

PROJ_DIR="/cluster/home/${USER}/degeneration-probe"
cd "${PROJ_DIR}"

export HF_HOME="/cluster/scratch/${USER}/hf_cache"
mkdir -p "${PROJ_DIR}/logs"

# Config file to use (override at submit time with GEN_CONFIG=... sbatch ...)
GEN_CONFIG="${GEN_CONFIG:-configs/generate/alpaca_qwen05b.yaml}"

echo "[job] config   = ${GEN_CONFIG}"
echo "[job] HF_HOME  = ${HF_HOME}"
echo "[job] GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python -m degeneration_probe generate --config "${GEN_CONFIG}"

echo "[job] Done."
