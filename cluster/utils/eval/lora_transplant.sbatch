#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --job-name=degeneration-probe-lora-transplant
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
mkdir -p $REPO/cluster/log
cd $REPO

RUN_DIR="${RUN_DIR:?a source LoRA run directory is required}"
CHECKPOINT="${CHECKPOINT:?a checkpoint name is required}"
DATASET_CONFIG="${DATASET_CONFIG:?a target training-side dataset config is required}"

echo "Commit: $(git rev-parse HEAD)"
echo "Run dir: $RUN_DIR  Checkpoint: $CHECKPOINT  Target: $DATASET_CONFIG"

export HF_TOKEN="$(cat ~/keys/.hf_token)"

srun --environment=$REPO/cluster/env.toml \
    python scripts/evaluate_lora_transplant.py \
        --run-dir "$RUN_DIR" --checkpoint "$CHECKPOINT" \
        --dataset-config "$DATASET_CONFIG"

echo "LoRA transplant scoring done."
