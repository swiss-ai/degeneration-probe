#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --job-name=lorenzo-visualize-layers
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

# Sweep produced by submit_layer_sweep.sh on 2026-05-15
SWEEP_DIR="${SWEEP_DIR:-/capstor/scratch/cscs/$USER/feature-probes/output/20260515_180522}"
LAYERS=(10 14 18 22 24 26 28 29 30 31)

PROBE_DIRS=()
for L in "${LAYERS[@]}"; do
    PROBE_DIRS+=( "$SWEEP_DIR/apertus_apertus_stability_repetition_balanced_layer_sweep_layer$L" )
done

RUN_TAG="${RUN_TAG:-layers_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/capstor/scratch/cscs/$USER/feature-probes/visualizations/${RUN_TAG}}"
SPLIT="${SPLIT:-validation}"
HF_REPO="${HF_REPO:-lorenzo0312/degeneration-probe-instruct-token-level-balanced}"

mkdir -p "$OUT_DIR"
echo "SWEEP_DIR=$SWEEP_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "SPLIT=$SPLIT"
echo "HF_REPO=$HF_REPO"
echo "PROBE_DIRS=${PROBE_DIRS[*]}"

srun --environment=$REPO/cluster/env.toml \
    python scripts/visualize_probe_tokens_layers.py \
        --probe-dirs "${PROBE_DIRS[@]}" \
        --output-dir "$OUT_DIR" \
        --hf-repo "$HF_REPO" \
        --split "$SPLIT" \
        --sample-ids \
            "nvidia_Llama-Nemotron-Post-Training-Dataset-chat-1710" \
            "zwhe99_DeepMath-103K-train-8289"
