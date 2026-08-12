#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --job-name=degeneration-probe-score-layers
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Score several depths of one multi-layer run, then put each through the
# evaluator. One job rather than one per depth, because the queue is the
# constraint and a depth costs only a few minutes:
#
#   sbatch cluster/score_layers.sbatch <run_dir> "4 8 12 15 20 25 30" [split]
#
# Each depth writes to <run_dir>/layers/layer_NN/, which is an ordinary run
# directory as far as the evaluator is concerned.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
RUN_DIR=${1:?a run directory is required}
LAYERS=${2:?a quoted list of layers is required}
SPLIT=${3:-val}

mkdir -p $REPO/cluster/log
cd $REPO

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Run:    $RUN_DIR"
echo "Layers: $LAYERS   Split: $SPLIT"

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
    for layer in $LAYERS; do
        printf '\n===== layer %s =====\n' \"\$layer\"
        python scripts/score_rollouts.py \
            --run-dir '$RUN_DIR' --checkpoint final \
            --layer \"\$layer\" --splits $SPLIT
        printf -v padded '%02d' \"\$layer\"
        python scripts/evaluate_scores.py \
            --run-dir '$RUN_DIR'/layers/layer_\$padded --splits $SPLIT
    done
"
