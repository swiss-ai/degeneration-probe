#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --job-name=degeneration-probe-lora-linearity
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# One teacher-forced pass per condition (frozen / lora) over the 30 rollouts
# the frozen-vs-LoRA linearity notebook compares, at every layer:
#
#   sbatch cluster/extract_lora_linearity.sbatch \
#       --frozen-run-dir outputs/<frozen_run>/<attempt> \
#       --lora-checkpoint outputs/<lora_run>/<attempt>/checkpoint-<n> \
#       --out-dir outputs/lora_linearity_probe/activations
#
# Arguments pass straight through to scripts/extract_lora_linearity_activations.py.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe

mkdir -p $REPO/cluster/log
cd $REPO

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"

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

srun --environment=$REPO/cluster/env.toml \
    python scripts/extract_lora_linearity_activations.py "$@"
