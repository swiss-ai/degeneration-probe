#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --job-name=onset-lora-pilot-smoketest
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"

export HF_TOKEN="$(cat ~/keys/.hf_token)"
export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"

# See cluster/layer_sweep.sbatch's comment for why PYTHONPATH must be set explicitly
# (the container's baked degeneration_probe install is a separate, possibly-stale git
# checkout, not this mounted $REPO -- this script's new code isn't in that image at all).
srun --environment=$REPO/cluster/env.toml \
    bash -c "PYTHONPATH=$REPO:\$PYTHONPATH python scripts/train_onset_lora_pilot.py \
        --smoke-test-n 3 \
        --gradient-accumulation-steps 2 \
        --eval-steps 2 \
        --logging-steps 1 \
        --report-to none"
