#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=lorenzo-resume
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

# Resume parameters (overridable via --export=...)
PROBE_ID="${PROBE_ID:-apertus_apertus_stability_repetition_balanced_layer30}"
RESUME_FROM="${RESUME_FROM:-/capstor/scratch/cscs/$USER/feature-probes/output/20260513_140153/$PROBE_ID/epoch_2}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
PROBE_HEAD_LR="${PROBE_HEAD_LR:-5e-5}"
LORA_LR="${LORA_LR:-5e-5}"

if [[ ! -d "$RESUME_FROM" ]]; then
    echo "ERROR: resume checkpoint not found: $RESUME_FROM" >&2
    exit 1
fi

# Per-run output dir, seeded with the resume checkpoint files.
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export PROBE_OUTPUT_DIR="/capstor/scratch/cscs/$USER/feature-probes/output/${RUN_TIMESTAMP}_resume_from_epoch2"
NEW_PROBE_PATH="$PROBE_OUTPUT_DIR/$PROBE_ID"
mkdir -p "$NEW_PROBE_PATH"
echo "PROBE_OUTPUT_DIR=$PROBE_OUTPUT_DIR"
echo "Seeding probe_path from: $RESUME_FROM"
cp -r "$RESUME_FROM"/. "$NEW_PROBE_PATH/"
echo "Files in new probe_path:"
ls "$NEW_PROBE_PATH"

srun --environment=$REPO/cluster/env.toml \
    python scripts/train_probe.py \
        model=apertus \
        training=apertus_stability_fp32 \
        dataset=repetition_instruct_token_level_balanced \
        probe_config.load_from=disk \
        num_train_epochs="$NUM_EPOCHS" \
        probe_head_lr="$PROBE_HEAD_LR" \
        lora_lr="$LORA_LR" \
        +wandb_tags='[resume,warm_start_epoch2]'
