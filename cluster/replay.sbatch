#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --job-name=degeneration-probe-replay
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Put every saved checkpoint of one run through the measurements the stopping
# rule reads, so a run that trained to the step cap can be told where it would
# have stopped:
#
#   sbatch cluster/replay.sbatch outputs/<run>/latest [step spacing]
#
# The evaluation population is read once and every checkpoint applied to it
# together, so this costs about one pass over the rollouts rather than one pass
# per checkpoint. Result goes to <run_dir>/checkpoint_replay.parquet.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
RUN_DIR=${1:?a run directory is required}
EVERY=${2:-50}

mkdir -p $REPO/cluster/log
cd $REPO

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Run:    $RUN_DIR"
echo "Every:  $EVERY steps"

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
    cd $REPO
    start=\$SECONDS
    python scripts/replay_checkpoints.py --run-dir '$(realpath "$RUN_DIR")' --every $EVERY
    echo \"ELAPSED replay: \$((SECONDS - start)) s\"
"
