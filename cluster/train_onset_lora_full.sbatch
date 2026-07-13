#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=11:00:00
#SBATCH --job-name=onset-lora-full
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Full-scale LoRA training for one onset-probe design -- multi-horizon (all 5 horizons)
# or a single probe_N horizon -- see scripts/train_onset_lora_pilot.py's module
# docstring. Horizons are passed as the sbatch script's first positional arg, a
# space-separated list, e.g.:
#   sbatch --job-name=onset-lora-multi cluster/train_onset_lora_full.sbatch "5 20 50 100 200"
#   sbatch --job-name=onset-lora-probe5 cluster/train_onset_lora_full.sbatch "5"

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
HORIZONS="${1:?Usage: sbatch train_onset_lora_full.sbatch \"<space-separated horizons>\"}"

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Horizons: $HORIZONS"

export HF_TOKEN="$(cat ~/keys/.hf_token)"
export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"

srun --environment=$REPO/cluster/env.toml \
    bash -c "PYTHONPATH=$REPO:\$PYTHONPATH python scripts/train_onset_lora_pilot.py \
        --horizons $HORIZONS \
        --gradient-accumulation-steps 16 \
        --eval-steps 100 \
        --logging-steps 10 \
        --report-to wandb"
