#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --job-name=degeneration-probe-human-eval-all
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# Read every scored depth and every baseline on the human-verified test splits:
#
#   sbatch cluster/human_eval_all.sbatch
#
# Needs the label tables in <store>/human_review/labels/ (written by
# cluster/human_eval.sbatch). Each depth gets an evaluation_human/ folder next
# to its evaluation/ folder; afterwards
#
#   python scripts/aggregate_selected.py --human
#   python scripts/response_metrics_pool.py --human --splits val test_indomain test_heldout_domains
#
# rebuild the human-verified versions of selected_metrics.csv and the pool.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
STORE=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe
LABELS=$STORE/human_review/labels
WORK=$STORE/human_review/eval_all/$(date +%Y%m%d_%H%M)
mkdir -p $REPO/cluster/log $WORK
cd $REPO
echo "Commit: $(git rev-parse HEAD)   Out: $WORK"

# Every depth aggregate_selected.py reads, and every baseline folder, grouped by build.
.venv/bin/python - "$WORK" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
work = Path(sys.argv[1])
outputs = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs")
datasets = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}
table = pd.read_csv(outputs / "analysis" / "selected_metrics.csv", usecols=["run_name", "attempt", "model", "layer"])
table = table.drop_duplicates(["run_name", "attempt", "layer"])
chunks = 48
for model, dataset in datasets.items():
    dirs = [str(outputs / r.run_name / str(r.attempt) / "selected" / f"layer_{int(r.layer):02d}")
            for r in table[table["model"] == model].itertuples()]
    dirs += [str(p) for root in ("baselines", "baselines_sweep") for p in sorted((outputs / root / dataset).glob("*"))
             if (p / "decision_thresholds.json").is_file()]
    share = max(1, round(chunks * len(dirs) / len(table)))
    for i in range(share):
        part = dirs[i::share]
        if part:
            (work / f"{dataset}__{i:02d}.txt").write_text("\n".join(part) + "\n")
    print(dataset, len(dirs), "folders in", share, "chunks")
EOF

run_chunk() {
    local list=$1
    local name=$(basename $list .txt)
    local dataset=${name%%__*}
    .venv/bin/python scripts/evaluate_human_labels.py \
        --dataset-config configs/dataset/builds/degeneration-dataset-$dataset.yaml \
        --labels-in $LABELS/labels_$dataset.csv \
        --run-dirs-file $list --populations hybrid --write-evaluation evaluation_human \
        --out $WORK/$name.csv > $WORK/$name.log 2>&1 \
        && echo "ok     $name $(wc -l < $list)" || { echo "FAILED $name"; return 1; }
}
export -f run_chunk
export LABELS WORK
ls $WORK/*.txt | xargs -P 48 -I{} bash -c 'run_chunk "$1"' _ {}
echo "Done at $(date)"
