#!/bin/bash
# Submit cluster/score_selected_all.sbatch for every run that has a checkpoint
# selection and no scores yet under <run>/selected/.
#
#   ./cluster/score_selected_launch.sh            submit
#   DRY_RUN=1 ./cluster/score_selected_launch.sh  list only
set -euo pipefail
cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe
OUTPUT_ROOT=${OUTPUT_ROOT:-/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs}
SUBDIR=${SUBDIR:-selected}
DRY_RUN=${DRY_RUN:-0}

submitted=0 skipped=0
for selection in "$OUTPUT_ROOT"/*/2*/checkpoint_selection.json; do
    [[ -e $selection ]] || continue
    run=${selection%/*}
    name=$(basename "$(dirname "$run")")/$(basename "$run")
    if ls "$run/$SUBDIR"/layer_*/scores/test_heldout_domains.parquet >/dev/null 2>&1; then
        echo "done   $name"; skipped=$((skipped + 1)); continue
    fi
    if [[ $DRY_RUN == 1 ]]; then
        echo "would  $name"
    else
        echo "submit $name: $(sbatch --parsable cluster/score_selected_all.sbatch "$run" "$SUBDIR")"
    fi
    submitted=$((submitted + 1))
done
echo "$submitted to score, $skipped skipped"
