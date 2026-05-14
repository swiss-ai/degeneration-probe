#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=lorenzo-train
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe/feature-probes

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"

export HF_TOKEN="$(cat ~/keys/.hf_token)"
export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

# Per-run output dir: /capstor/.../feature-probes/output/<timestamp>/<probe_id>/
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export PROBE_OUTPUT_DIR="/capstor/scratch/cscs/$USER/feature-probes/output/$RUN_TIMESTAMP"
mkdir -p "$PROBE_OUTPUT_DIR"
echo "PROBE_OUTPUT_DIR=$PROBE_OUTPUT_DIR"

srun --environment=$REPO/cluster/env.toml \
    python scripts/train_probe.py \
        model=apertus \
        training=apertus_stability_fp32 \
        dataset=repetition_instruct_token_level_balanced
