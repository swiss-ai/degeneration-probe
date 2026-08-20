#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --job-name=degeneration-probe-score-selected
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Score several depths of one run, each at the checkpoint its own rule selected,
# rather than all of them at the final step:
#
#   sbatch cluster/score_selected_depths.sbatch <run_dir> "4:700 8:500 15:1000" [split]
#
# The pairs are layer:step. Each lands in <run_dir>/sweep/layer_NN/step_<step>/,
# which is where a checkpoint other than the reported one belongs, so nothing
# already in layers/ is touched. Every destination gets its own provenance file,
# so the checkpoint behind each number can be named afterwards.
#
# The walltime fits about eight pairs: the container and the model cost roughly
# eight minutes once, and each pair about three after that. Keep it tight, since a
# short job is backfilled into a gap that a long one waits behind, and override
# with --time for a longer list rather than padding the default for everyone.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
RUN_DIR=${1:?a run directory is required}
LAYER_STEPS=${2:?a quoted list of layer:step pairs is required}
SPLIT=${3:-val}

mkdir -p $REPO/cluster/log
cd $REPO

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Run:    $RUN_DIR"
echo "Pairs:  $LAYER_STEPS   Split: $SPLIT"

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
    for pair in $LAYER_STEPS; do
        layer=\${pair%%:*}
        step=\${pair##*:}
        printf -v padded '%02d' \"\$layer\"
        dest='$RUN_DIR'/sweep/layer_\$padded/step_\$step
        printf '\n===== layer %s at step %s -> %s =====\n' \"\$layer\" \"\$step\" \"\$dest\"
        python scripts/score_rollouts.py \
            --run-dir '$RUN_DIR' --checkpoint checkpoint-\$step \
            --layer \"\$layer\" --splits $SPLIT --output-dir \"\$dest\"
        python scripts/evaluate_scores.py --run-dir \"\$dest\" --splits $SPLIT
    done
"
