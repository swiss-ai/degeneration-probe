#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --job-name=degeneration-probe-baselines
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Score the heuristic baselines and push them through the evaluation protocol,
# the same one the probes go through:
#   sbatch cluster/baselines.sbatch
# The persistence sweep is what fixes m for every later experiment.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
BUILD_ROOT=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/degeneration-dataset-apertus-8b-instruct

mkdir -p $REPO/cluster/log
cd $REPO

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"

srun --environment=$REPO/cluster/env.toml bash -c "
    set -euo pipefail
    python scripts/score_baselines.py \
        --build-root $BUILD_ROOT \
        --out-dir outputs/baselines \
        --baselines repetition entropy lrs \
        --splits val test_indomain test_heldout_domains
    for name in repetition entropy lrs; do
        python scripts/evaluate_scores.py \
            --run-dir outputs/baselines/\$name \
            --compare-persistence 1 4 8 16 32
    done
"
