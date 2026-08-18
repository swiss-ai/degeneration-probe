#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=degeneration-probe-score-heldout
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Report one checkpoint of one depth on a split it has never been read on:
#
#   sbatch cluster/score_heldout.sbatch <run_dir> <layer> <checkpoint> [split]
#
# The job scores validation first, freezes the thresholds from it, and only
# then scores the held-out split and reports it against those frozen numbers.
# Everything lands under sweep/layer_NN/<checkpoint>/, beside the run's other
# checkpoints rather than on top of them.
#
# Three details are the reason this is a script rather than a command.
#
# A checkpoint has to be named. A run's `final` weights are whichever step won
# on rollout recall, which saturates within the first few evaluations, so
# `final` is typically an early checkpoint and not the one the depth is
# reported at. Naming the checkpoint keeps the scorer and the numbers it is
# compared against the same weights.
#
# Validation is scored in the same job as the held-out split. The frozen
# thresholds have to come from this exact checkpoint, and a threshold frozen
# from some other checkpoint's scores describes a different scorer.
#
# The evaluator is called with --reuse-thresholds for the held-out split.
# Without it, it re-derives the thresholds from validation on every call, which
# is the wrong default once a test split is in play: the point of freezing is
# that the file was written before the split was read.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
RUN_DIR=${1:?a run directory is required}
LAYER=${2:?a layer is required}
CHECKPOINT=${3:?a checkpoint is required, for example checkpoint-1000}
SPLIT=${4:-test_heldout_domains}
printf -v PADDED '%02d' "$LAYER"
OUT_DIR=$RUN_DIR/sweep/layer_$PADDED/$CHECKPOINT

mkdir -p "$REPO/cluster/log"
cd $REPO

if [[ ! -d $RUN_DIR/$CHECKPOINT ]]; then
    echo "No $CHECKPOINT in $RUN_DIR." >&2
    exit 1
fi

echo "Commit:     $(git -C $REPO rev-parse HEAD)"
echo "Run:        $RUN_DIR"
echo "Layer:      $LAYER"
echo "Checkpoint: $CHECKPOINT"
echo "Split:      $SPLIT"
echo "Output:     $OUT_DIR"

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

srun --environment=$REPO/cluster/env.toml bash -c "
    set -euo pipefail

    printf '\n===== validation, to freeze the thresholds =====\n'
    python scripts/score_rollouts.py \
        --run-dir '$RUN_DIR' --checkpoint '$CHECKPOINT' \
        --layer '$LAYER' --splits val --output-dir '$OUT_DIR'
    python scripts/evaluate_scores.py --run-dir '$OUT_DIR' --splits val

    printf '\n===== %s, against those frozen thresholds =====\n' '$SPLIT'
    python scripts/score_rollouts.py \
        --run-dir '$RUN_DIR' --checkpoint '$CHECKPOINT' \
        --layer '$LAYER' --splits '$SPLIT' --output-dir '$OUT_DIR'
    python scripts/evaluate_scores.py \
        --run-dir '$OUT_DIR' --splits '$SPLIT' --reuse-thresholds
"
