#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --job-name=analyze-probe-errors
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe/feature-probes
mkdir -p $REPO/cluster/log

export HF_TOKEN="$(cat ~/keys/.hf_token)"
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

# Override these with: --export=PROBE_ID=...,EPOCH=...,DATASET_SPLIT=...
PROBE_ID="${PROBE_ID:-apertus_apertus_stability_repetition_balanced_layer30}"
EPOCH="${EPOCH:-1}"
MODEL="${MODEL:-apertus}"
LAYER="${LAYER:-30}"
PROBE_DTYPE="${PROBE_DTYPE:-float32}"
NORM="${NORM:-layernorm}"
LORA_LAYERS="${LORA_LAYERS:-all}"
TOP_K="${TOP_K:-50}"
DATASET_SPLIT="${DATASET_SPLIT:-validation}"
DATASET_ID="${DATASET_ID:-repetition_${DATASET_SPLIT}}"
OUTPUT_DIR_ARG=""
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR_ARG="--output-dir $OUTPUT_DIR"
fi

srun --environment=$REPO/cluster/env.toml \
    python scripts/analyze_probe_errors.py \
        --probe-id "$PROBE_ID" \
        --model "$MODEL" \
        --layer "$LAYER" \
        --probe-dtype "$PROBE_DTYPE" \
        --norm "$NORM" \
        --lora-layers "$LORA_LAYERS" \
        --epoch "$EPOCH" \
        --top-k "$TOP_K" \
        --dataset-id "$DATASET_ID" \
        --dataset-split "$DATASET_SPLIT" \
        $OUTPUT_DIR_ARG
