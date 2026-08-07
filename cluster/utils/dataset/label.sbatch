#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --job-name=degeneration-probe-label
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
mkdir -p $REPO/cluster/log
cd $REPO

CONFIG="${CONFIG:-configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml}"

echo "Commit: $(git rev-parse HEAD)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Job: $SLURM_JOB_ID  CPUs: $SLURM_CPUS_PER_TASK  Config: $CONFIG"

# label.py is CPU-only (numpy/pandas, no torch) -- runs directly against the
# repo's .venv on the compute node, no container needed (same aarch64 arch
# as the login node this venv was built on).
source .venv/bin/activate

python -m degeneration_probe.dataset_gen.label \
    --config "$CONFIG" \
    --window-size 256 \
    --ttr-ngram 2 \
    --lrs-min-length 10 \
    --workers "$SLURM_CPUS_PER_TASK"

echo "Labeling done."
