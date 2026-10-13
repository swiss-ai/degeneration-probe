#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --job-name=degeneration-probe-baselines
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Score the heuristic baselines and push them through the evaluation protocol,
# the same one the probes go through:
#
#   sbatch cluster/baselines.sbatch [dataset] [baseline ...]
#
# The dataset defaults to apertus-8b-instruct and the baselines to every causal
# one that needs no extra artifact; neglogp needs <build>/token_logprobs first
# (cluster/token_logprobs.sbatch).
#
# None of the baselines reads the model, so this is CPU work over the dataset
# build. Each baseline gets its own directory under
# <outputs>/baselines/<dataset>/<baseline>/{scores,evaluation}, in the layout
# every probe depth uses, so the same readers serve both.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
STORE=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe
DATASET=${1:-apertus-8b-instruct}
BUILD_ROOT=$STORE/degeneration-dataset-$DATASET
OUT_DIR=$STORE/outputs/baselines/$DATASET
shift || true
BASELINES="${*:-repetition repetition_trailing entropy lrs rep_l ngram_8 ngram_16 ngram_30}"

mkdir -p $REPO/cluster/log $OUT_DIR
cd $REPO
echo "Commit: $(git rev-parse HEAD)   Dataset: $DATASET"

.venv/bin/python scripts/score_baselines.py \
    --build-root $BUILD_ROOT \
    --out-dir $OUT_DIR \
    --baselines $BASELINES \
    --splits val test_indomain test_heldout_domains

evaluate_baseline() {
    local name=$1
    .venv/bin/python scripts/evaluate_scores.py --run-dir "$OUT_DIR/$name" --splits val \
        --onset-labels "$BUILD_ROOT/onset_labels/onset_labels.parquet" \
        && .venv/bin/python scripts/evaluate_scores.py --run-dir "$OUT_DIR/$name" \
            --splits test_indomain test_heldout_domains --reuse-thresholds \
            --onset-labels "$BUILD_ROOT/onset_labels/onset_labels.parquet" \
        && echo "ok     $name" || { echo "FAILED $name"; return 1; }
}
export -f evaluate_baseline
export OUT_DIR BUILD_ROOT
printf '%s\n' $BASELINES | xargs -P 8 -I{} bash -c 'evaluate_baseline "$1"' _ {}
