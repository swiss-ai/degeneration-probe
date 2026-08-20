#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --job-name=degeneration-probe-actsim
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# The training-free activation self-similarity baseline, read straight off the
# cached hidden states:
#   sbatch cluster/activation_similarity.sbatch --split val
# Arguments pass straight through to the scoring script.
#
# Validation is roughly 120 GB of strided reads for six layers and finishes well
# inside the walltime; the arithmetic afterwards is minor. Narrow the layer or lag
# list rather than raising the limit, because a shorter job starts sooner.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
BUILD=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/degeneration-dataset-apertus-8b-instruct

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"

srun --environment=$REPO/cluster/env.toml \
    python scripts/score_activation_similarity.py --build-root $BUILD "$@"
