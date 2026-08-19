#!/bin/bash
# Score the checkpoint the rule chose, for every adapted run that has finished.
#
# A run measures enough while it trains to decide which of its checkpoints is
# worth keeping, and no more than that: the reported views need one score per
# token of every answer, which costs a pass over the data and is spent once, on
# the checkpoint the rule named.
#
# Reads that checkpoint out of each run's own verdict rather than taking the
# weights left in final/, which are wherever training happened to stop and are
# usually well past the peak.
#
#   ./cluster/score_selected.sh              submit the jobs
#   DRY_RUN=1 ./cluster/score_selected.sh    print what would be submitted
#
# Scores are written inside each run's own directory, so they land wherever the
# run does.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/iopsstor/scratch/cscs/mdenegri/degeneration-probe/outputs}
# The held-out domains are read once, at the end, for the probe that is actually
# reported. Scoring them for every run would spend the one measurement that is
# supposed to stay untouched.
SPLITS=${SPLITS:-"val test_indomain"}
DRY_RUN=${DRY_RUN:-0}

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

submitted=0 skipped=0
for verdict in "$OUTPUT_ROOT"/*/2*/selection_outcomes.json; do
    [[ -e $verdict ]] || continue
    run=${verdict%/*}
    name=$(basename "$(dirname "$run")")

    [[ $(jq -r '.training.features.regime' "$run/resolved_config.json") == adapted ]] || continue

    status=$(jq -r '.status' "$run/run_info.json")
    if [[ $status != finished ]]; then
        echo "skip   $name  (still $status)"; skipped=$((skipped + 1)); continue
    fi

    step=$(jq -r '.depths[0].selected_step // empty' "$verdict")
    if [[ -z $step ]]; then
        # The depth never cleared the in-loop coverage floor, so no checkpoint of
        # it qualifies. That is a result to report, not a run to score.
        echo "skip   $name  (no checkpoint qualified)"; skipped=$((skipped + 1)); continue
    fi
    checkpoint="checkpoint-${step%.*}"

    if [[ ! -d $run/$checkpoint ]]; then
        echo "skip   $name  ($checkpoint is missing)"; skipped=$((skipped + 1)); continue
    fi

    # A directory of scores is read as one scorer, since the thresholds frozen in
    # it describe whatever filled it.
    previous=$(jq -r '.checkpoint // empty' "$run/scoring_provenance.json" 2>/dev/null || true)
    if [[ $previous == "$checkpoint" ]]; then
        echo "done   $name  ($checkpoint already scored)"; skipped=$((skipped + 1)); continue
    fi

    if [[ $DRY_RUN == 1 ]]; then
        echo "would  $name  $checkpoint  [$SPLITS]"
    else
        sbatch cluster/score.sbatch --run-dir "$run" --checkpoint "$checkpoint" --splits $SPLITS >/dev/null
        echo "queued $name  $checkpoint  [$SPLITS]"
    fi
    submitted=$((submitted + 1))
done

echo
echo "$submitted to score, $skipped skipped"
