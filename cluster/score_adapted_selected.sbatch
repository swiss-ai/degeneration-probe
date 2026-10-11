#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=degeneration-probe-score-adapted
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Score an adapted (LoRA) run at the checkpoint the rule selects for its one
# depth, on validation and both test splits, then evaluate it with thresholds
# frozen on validation:
#
#   sbatch cluster/score_adapted_selected.sbatch <run_dir> [subdir]
#   sbatch --dependency=afterok:<train job> cluster/score_adapted_selected.sbatch <train job> [subdir]
#
# Given a training job id instead of a directory, the run directory is read from
# that job's log, which lets the scoring be queued behind a training job whose
# attempt directory does not exist yet.
#
# An adapted run cannot be replayed, so the selection reads the records its
# monitoring pass wrote during training (selection_history.parquet), under the
# same rule and fallback as the frozen sweep. Results land in
# <run_dir>/<subdir>/layer_NN/{scores,evaluation}/, the layout the frozen runs use,
# so one aggregation reads both regimes.
set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
TARGET=${1:?a run directory or a training job id is required}
SUBDIR=${2:-selected}
mkdir -p $REPO/cluster/log
cd $REPO
if [[ $TARGET =~ ^[0-9]+$ ]]; then
    TRAIN_LOG=$REPO/cluster/log/degeneration-probe-train-$TARGET.log
    TARGET=$(grep -m1 "^Run directory:" "$TRAIN_LOG" | awk '{print $3}')
    [[ -n $TARGET ]] || { echo "No run directory in $TRAIN_LOG" >&2; exit 1; }
fi
RUN_DIR=$(realpath "$TARGET")
[[ -f $RUN_DIR/run_info.json ]] || { echo "$RUN_DIR is not a run directory" >&2; exit 1; }
STATUS=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" "$RUN_DIR/run_info.json")
[[ $STATUS == finished ]] || { echo "run status is '$STATUS', not finished; refusing to score" >&2; exit 1; }
echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Run:    $RUN_DIR"

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

if [[ ! -f $RUN_DIR/checkpoint_selection.json ]]; then
    .venv/bin/python scripts/select_checkpoints.py --run-dir "$RUN_DIR" --source history
fi
read -r LAYER STEP < <(.venv/bin/python -c "
import json, sys
depths = json.load(open(sys.argv[1]))['depths']
assert len(depths) == 1, f'expected one depth, found {len(depths)}'
print(depths[0]['layer'], depths[0]['selected_step'])" "$RUN_DIR/checkpoint_selection.json")
printf -v PADDED '%02d' "$LAYER"
OUT_DIR=$RUN_DIR/$SUBDIR/layer_$PADDED
echo "Depth:  layer $LAYER at checkpoint-$STEP -> $OUT_DIR"
[[ -d $RUN_DIR/checkpoint-$STEP ]] || { echo "No checkpoint-$STEP in $RUN_DIR" >&2; exit 1; }

srun --environment=$REPO/cluster/env.toml bash -c "
    set -euo pipefail
    cd $REPO
    start=\$SECONDS
    python scripts/score_rollouts.py --run-dir '$RUN_DIR' --checkpoint checkpoint-$STEP \
        --splits val test_indomain test_heldout_domains --output-dir '$OUT_DIR'
    echo \"ELAPSED scoring: \$((SECONDS - start)) s\"
    python scripts/evaluate_scores.py --run-dir '$OUT_DIR' --splits val
    python scripts/evaluate_scores.py --run-dir '$OUT_DIR' --splits test_indomain test_heldout_domains --reuse-thresholds
"
