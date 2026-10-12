#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=degeneration-probe-reevaluate
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Recompute the evaluation views of every scored depth of the listed runs from
# the per-token scores already on disk:
#
#   sbatch cluster/reevaluate_selected.sbatch <file with one run directory per line> [subdir]
#
# Scores are not touched. Validation is evaluated first, which freezes the
# thresholds again (the same data gives the same thresholds), then the test
# splits are read against them. Depths run in parallel, several per node,
# since each is a few seconds of CPU over one table.
set -uo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
LIST=${1:?a file listing run directories is required}
SUBDIR=${2:-selected}
cd $REPO
echo "Commit: $(git rev-parse HEAD)   List: $LIST ($(wc -l < "$LIST") runs)"

evaluate_depth() {
    local depth=$1
    [[ -f $depth/scores/val.parquet ]] || { echo "no val scores in $depth"; return 1; }
    .venv/bin/python scripts/evaluate_scores.py --run-dir "$depth" --splits val > "$depth/evaluation.log" 2>&1 \
        && .venv/bin/python scripts/evaluate_scores.py --run-dir "$depth" \
            --splits test_indomain test_heldout_domains --reuse-thresholds >> "$depth/evaluation.log" 2>&1 \
        && echo "ok     $depth" || { echo "FAILED $depth"; return 1; }
}
export -f evaluate_depth

failed=0
while read -r run; do
    [[ -n $run ]] || continue
    printf '\n===== %s =====\n' "$run"
    ls -d "$run/$SUBDIR"/layer_*/ | xargs -P "$SLURM_CPUS_PER_TASK" -I{} bash -c 'evaluate_depth "$1"' _ {} \
        || failed=$((failed + 1))
done < "$LIST"
echo "reevaluation done, $failed runs with failures"
exit $(( failed > 0 ))
