#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=degeneration-probe-score-selected
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Score every depth of one run at its selected checkpoint, on validation and
# both test splits, then evaluate each depth with thresholds frozen on
# validation:
#
#   sbatch cluster/score_selected_all.sbatch <run_dir> [subdir]
#
# Needs <run_dir>/checkpoint_selection.json (scripts/select_checkpoints.py).
# The three splits are read once each for all depths; the per-depth results land
# in <run_dir>/<subdir>/layer_NN/{scores,evaluation}/.
set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
RUN_DIR=$(realpath "${1:?a run directory is required}")
SUBDIR=${2:-selected}
mkdir -p $REPO/cluster/log
cd $REPO
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

# A run queued behind its replay has no selection yet; decide it here, from the
# replay, so the chain needs no separate step.
if [[ ! -f $RUN_DIR/checkpoint_selection.json ]]; then
    .venv/bin/python scripts/select_checkpoints.py --run-dir "$RUN_DIR" --source replay
fi

srun --environment=$REPO/cluster/env.toml bash -c "
    set -euo pipefail
    cd $REPO
    start=\$SECONDS
    python scripts/score_selected_depths.py --run-dir '$RUN_DIR' --subdir '$SUBDIR' \
        --splits val test_indomain test_heldout_domains
    echo \"ELAPSED scoring: \$((SECONDS - start)) s\"
    start=\$SECONDS
    for depth in '$RUN_DIR'/'$SUBDIR'/layer_*/; do
        printf '\n===== %s =====\n' \"\$depth\"
        python scripts/evaluate_scores.py --run-dir \"\$depth\" --splits val
        python scripts/evaluate_scores.py --run-dir \"\$depth\" --splits test_indomain test_heldout_domains --reuse-thresholds
    done
    echo \"ELAPSED evaluation: \$((SECONDS - start)) s\"
"
