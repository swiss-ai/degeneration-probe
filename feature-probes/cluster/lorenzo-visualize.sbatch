#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --job-name=lorenzo-visualize
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
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

PROBE_DIR="${PROBE_DIR:-/capstor/scratch/cscs/$USER/feature-probes/output/20260513_140153/apertus_apertus_stability_repetition_balanced_layer30/epoch_2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/capstor/scratch/cscs/$USER/feature-probes/visualizations/${RUN_TAG}}"
SPLIT="${SPLIT:-test}"
HF_REPO="${HF_REPO:-lorenzo0312/degeneration-probe-instruct-token-level-balanced}"

mkdir -p "$OUT_DIR"
echo "PROBE_DIR=$PROBE_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "SPLIT=$SPLIT"
echo "HF_REPO=$HF_REPO"

srun --environment=$REPO/cluster/env.toml \
    python scripts/visualize_probe_tokens.py \
        --probe-dir "$PROBE_DIR" \
        --output-dir "$OUT_DIR" \
        --hf-repo "$HF_REPO" \
        --split "$SPLIT" \
        --sample-ids \
            "nvidia_Llama-Nemotron-Post-Training-Dataset-chat-1710" \
            "zwhe99_DeepMath-103K-train-8289"
