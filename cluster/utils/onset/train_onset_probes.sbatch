#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --job-name=onset-probe-train
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-32%6

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
LAYER=$SLURM_ARRAY_TASK_ID

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Layer: $LAYER"

# CPU-only (materializes cached activations into memory, trains tiny linear/MLP
# heads -- no GPU needed). See cluster/utils/onset/layer_sweep.sbatch's comment for why
# PYTHONPATH must be set explicitly (the container's baked `degeneration_probe`
# install is a separate, possibly-stale git checkout, not this mounted $REPO).
srun --environment=$REPO/cluster/env.toml \
    bash -c "PYTHONPATH=$REPO:\$PYTHONPATH python scripts/train_onset_probes.py --layer $LAYER"
