#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --job-name=degeneration-probe-score-extra
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16

# Add the rollouts each listed run's test scores are missing (capped answers the
# judge ruled healthy), then re-evaluate the test splits of every depth against
# the thresholds frozen on validation:
#
#   sbatch cluster/score_extra.sbatch <file with one run directory per line>
set -uo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
LIST=${1:?a file listing run directories is required}
cd $REPO
echo "Commit: $(git rev-parse HEAD)   List: $LIST ($(wc -l < "$LIST") runs)"
HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_TOKEN
failed=0
while read -r run; do
    [[ -n $run ]] || continue
    if srun --environment=$REPO/cluster/env.toml bash -c "cd $REPO && python scripts/score_extra_rollouts.py --run-dir '$run'" < /dev/null; then
        ls -d "$run"/selected/layer_*/ | xargs -P "$SLURM_CPUS_PER_TASK" -I{} bash -c \
            '.venv/bin/python scripts/evaluate_scores.py --run-dir "$1" --splits test_indomain test_heldout_domains --reuse-thresholds > "$1/evaluation_extra.log" 2>&1 || echo "FAILED evaluation $1"' _ {}
    else
        echo "FAILED scoring $run"; failed=$((failed + 1))
    fi
done < "$LIST"
echo "done, $failed runs failed"
