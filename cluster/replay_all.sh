#!/bin/bash
# Replay every finished run of the sweep that has no replay yet: each saved
# checkpoint of each depth measured on the whole validation split, which is what
# scripts/select_checkpoints.py reads.
#
#   ./cluster/replay_all.sh            submit
#   DRY_RUN=1 ./cluster/replay_all.sh  list only
set -euo pipefail
cd /iopsstor/scratch/cscs/mdenegri/degeneration-probe
OUTPUT_ROOT=${OUTPUT_ROOT:-/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs}
EVERY=${EVERY:-100}
DRY_RUN=${DRY_RUN:-0}

submitted=0 skipped=0
for info in "$OUTPUT_ROOT"/*/2*/run_info.json; do
    [[ -e $info ]] || continue
    run=${info%/*}
    name=$(basename "$(dirname "$run")")/$(basename "$run")
    status=$(jq -r '.status' "$info")
    if [[ $status != finished ]]; then
        echo "skip   $name  (status: $status)"; skipped=$((skipped + 1)); continue
    fi
    if [[ -f $run/checkpoint_replay.parquet ]]; then
        echo "done   $name"; skipped=$((skipped + 1)); continue
    fi
    if [[ $DRY_RUN == 1 ]]; then
        echo "would  $name"
    else
        echo "submit $name: $(sbatch --parsable cluster/replay.sbatch "$run" "$EVERY")"
    fi
    submitted=$((submitted + 1))
done
echo "$submitted to replay, $skipped skipped"
