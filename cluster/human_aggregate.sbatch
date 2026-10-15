#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=00:45:00
#SBATCH --job-name=degeneration-probe-human-aggregate
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# The human-verified versions of selected_metrics.csv and the best-accuracy
# pool, from the evaluation_human folders written by cluster/human_eval_all.sbatch.
#
#   sbatch cluster/human_aggregate.sbatch

set -euo pipefail
cd /iopsstor/scratch/cscs/$USER/degeneration-probe
echo "Commit: $(git rev-parse HEAD)"
.venv/bin/python scripts/aggregate_selected.py --human
.venv/bin/python scripts/response_metrics_pool.py --human --workers 64 --splits val test_indomain test_heldout_domains
echo "Done at $(date)"
