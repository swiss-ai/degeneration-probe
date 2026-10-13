#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --job-name=degeneration-probe-response-pool
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Response-level AUC and best accuracy of every evaluated depth and baseline:
#
#   sbatch cluster/response_metrics_pool.sbatch [split ...]    # default val test_indomain
#
# Reads only the rollout tables that changed since the last run.
set -euo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
cd $REPO
echo "Commit: $(git rev-parse HEAD)"
SPLITS=${*:-val test_indomain}
.venv/bin/python scripts/response_metrics_pool.py --workers "$SLURM_CPUS_PER_TASK" --splits $SPLITS
