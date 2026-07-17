#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --job-name=onset-layer-sweep
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Dirty files: $(git -C $REPO status --porcelain | wc -l)"

# CPU-only (reads cached activations off disk, no model forward pass) --
# no --gpus-per-task, just reuses the same container environment as training.
#
# The container's `degeneration_probe` package is an editable install baked in at
# image-build time from a git clone under /workspace (see cluster/Dockerfile) --
# NOT the same checkout as this mounted $REPO. env.toml mounts $REPO only as the
# working directory, so without this, code that isn't yet in the baked image
# (e.g. this session's new onset_labels.py / onset_dataset.py) resolves to the
# stale baked copy instead and fails with ModuleNotFoundError. Prepending $REPO to
# PYTHONPATH makes the live, mounted checkout take priority.
srun --environment=$REPO/cluster/env.toml \
    bash -c "PYTHONPATH=$REPO:\$PYTHONPATH python scripts/layer_sweep.py"
