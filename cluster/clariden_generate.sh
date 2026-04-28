#!/usr/bin/env bash
# =============================================================================
# SLURM job: generate threshold data on CSCS Clariden
#
# Submit all three datasets:
#   sbatch cluster/clariden_generate.sh configs/generate/threshold_alpaca_apertus8b.yaml
#   sbatch cluster/clariden_generate.sh configs/generate/threshold_hermes_apertus8b.yaml
#   sbatch cluster/clariden_generate.sh configs/generate/threshold_aime_apertus8b.yaml
# =============================================================================
#SBATCH --job-name=degen-threshold
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --environment=my_env
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/degen_threshold_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/degen_threshold_%j.err

set -eo pipefail

PROJ_DIR="$HOME/degeneration-probe"
cd "${PROJ_DIR}"

# Create log dir
mkdir -p /iopsstor/scratch/cscs/${USER}/logs

# Load secrets (HF_TOKEN, etc.) from .env if present
if [ -f "${PROJ_DIR}/.env" ]; then
    set -a; source "${PROJ_DIR}/.env"; set +a
fi

# Use scratch for HF cache
export HF_HOME="/iopsstor/scratch/cscs/${USER}/hf_cache"
mkdir -p "${HF_HOME}"

# Config file passed as first argument
GEN_CONFIG="${1:?Usage: sbatch cluster/clariden_generate.sh <config.yaml>}"

echo "[job] config   = ${GEN_CONFIG}"
echo "[job] HF_HOME  = ${HF_HOME}"
echo "[job] node     = $(hostname)"
echo "[job] GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Install deps if needed
pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || true

python -m degeneration_probe generate --config "${GEN_CONFIG}"

echo "[job] Done."
