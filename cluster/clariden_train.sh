#!/usr/bin/env bash
# =============================================================================
# SLURM job: train the degeneration probe on CSCS Clariden.
#
# Usage:
#   sbatch cluster/clariden_train.sh <train-config.yaml>
#
# The config's `train_data:` field MUST point at a real generations.jsonl
# (produced by an earlier clariden_generate.sh run).
# =============================================================================
#SBATCH --job-name=degen-train
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --environment=my_env
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/degen_train_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/degen_train_%j.err

set -eo pipefail

PROJ_DIR="$HOME/degeneration-probe"
cd "${PROJ_DIR}"

mkdir -p /iopsstor/scratch/cscs/${USER}/logs

export HF_HOME="/iopsstor/scratch/cscs/${USER}/hf_cache"
export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/users/${USER}/hf_models"
mkdir -p "${HF_HOME}"

TRAIN_CONFIG="${1:?Usage: sbatch cluster/clariden_train.sh <train-config.yaml>}"

echo "[job] config   = ${TRAIN_CONFIG}"
echo "[job] HF_HOME  = ${HF_HOME}"
echo "[job] node     = $(hostname)"
echo "[job] GPU      = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Initialise / update the hallucination_probes fork submodule so the training
# code (which lives in third_party/hallucination_probes/degeneration/) is
# present and at the pinned commit.
git submodule update --init --recursive third_party/hallucination_probes

# Install our package (drags in peft, jaxtyping, termcolor, etc. via deps).
pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || true

python -m degeneration_probe train --config "${TRAIN_CONFIG}"

echo "[job] Done."
