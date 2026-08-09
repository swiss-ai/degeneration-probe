#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --job-name=degeneration-probe-train
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

# An assignment takes the exit status of the assignment, not of the command
# substitution inside it, so a missing key file would otherwise export an empty
# credential and fail much later with a confusing error.
read_key() {
    local file=$1
    if [[ ! -f $file ]]; then
        echo "Missing credential file: $file" >&2
        return 1
    fi
    cat "$file"
}

HF_TOKEN="$(read_key ~/keys/.hf_token)"
export HF_TOKEN
if [[ -f ~/keys/.wandb_key ]]; then
    WANDB_API_KEY="$(read_key ~/keys/.wandb_key)"
    export WANDB_API_KEY
else
    echo "No ~/keys/.wandb_key: falling back to the credentials in ~/.netrc" >&2
fi
if [[ -f ~/keys/.hf_token_write ]]; then
    HF_WRITE_TOKEN="$(read_key ~/keys/.hf_token_write)"
    export HF_WRITE_TOKEN
fi

# Hydra overrides pass straight through, so one job script covers every
# recipe: sbatch cluster/train.sbatch training.probe.layer=20 ...
srun --environment=$REPO/cluster/env.toml \
    python scripts/train_probe.py "$@"
