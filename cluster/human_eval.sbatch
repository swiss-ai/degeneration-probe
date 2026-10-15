#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --job-name=degeneration-probe-human-eval
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Read every scorer of the comparison tables on the human-verified test splits:
#
#   sbatch cluster/human_eval.sbatch
#
# For each model, the baselines at their settings chosen on validation and the
# three reported probes (three seeds each) are relabelled with the returned
# reviews and evaluated at their frozen thresholds. Each scorer gets an
# evaluation_human/ folder next to its evaluation/ folder, the human label
# table goes to <store>/human_review/labels/, and a summary of the four
# populations per scorer goes to <store>/human_review/eval/<timestamp>/.
# Afterwards, on the login node:
#
#   python scripts/q3_comparison.py --sweep --human
#   python scripts/baseline_sweep_tables.py --human --out <dir>

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
STORE=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe
HR=$STORE/human_review
EVAL=$HR/eval/$(date +%Y%m%d_%H%M)
mkdir -p $REPO/cluster/log $EVAL $HR/labels
cd $REPO
echo "Commit: $(git rev-parse HEAD)   Out: $EVAL"

# One line per scorer folder: <dataset> <scorer> <folder holding scores/ and decision_thresholds.json>.
.venv/bin/python - > $EVAL/scorers.txt <<'EOF'
import pandas as pd
from pathlib import Path
import scripts.q3_comparison as q3
probes = pd.read_csv(q3.OUTPUTS / "analysis" / "selected_metrics.csv")
probes = probes[probes["recipe"].str.startswith(q3.RECIPE) & (probes["split"] == "val")]
for model, dataset in q3.MODELS.items():
    for folder, label in q3.baseline_dirs(model, dataset, sweep=True):
        print(dataset, folder.parent.name, folder.parent)
    frozen, own = q3.LAYERS[model]
    part = probes[probes["model"] == model]
    for lora, layer, name in [("none", frozen, f"frozen_L{frozen}"), ("all", frozen, f"adapted_L{frozen}"),
                              ("all", own, f"adapted_L{own}")]:
        runs = part[(part["lora"] == lora) & (part["layer"] == layer)].drop_duplicates(["run_name", "attempt"])
        for _, run in runs.iterrows():
            folder = q3.OUTPUTS / run["run_name"] / str(run["attempt"]) / "selected" / f"layer_{layer:02d}"
            print(dataset, f"{name}_s{run['seed']}", folder)
EOF
cat $EVAL/scorers.txt

# The label tables, once per model.
for DATASET in $(cut -d' ' -f1 $EVAL/scorers.txt | sort -u); do
    FIRST=$(awk -v d=$DATASET '$1==d {print $3; exit}' $EVAL/scorers.txt)
    .venv/bin/python scripts/evaluate_human_labels.py \
        --dataset-config configs/dataset/builds/degeneration-dataset-$DATASET.yaml \
        --reviews $HR/round2_test_splits/returned $HR/round1_apertus_heldout \
        --adjudications $HR/round2_test_splits/returned/adjudications.csv \
        --run-dirs $FIRST --splits test_indomain --populations judge \
        --labels-out $HR/labels/labels_$DATASET.csv --out $EVAL/.labels_$DATASET.csv > $EVAL/labels_$DATASET.log 2>&1
    cp $HR/labels/labels_$DATASET.csv $EVAL/
done

evaluate_scorer() {
    local dataset=$1 name=$2 folder=$3
    .venv/bin/python scripts/evaluate_human_labels.py \
        --dataset-config configs/dataset/builds/degeneration-dataset-$dataset.yaml \
        --reviews $HR/round2_test_splits/returned $HR/round1_apertus_heldout \
        --adjudications $HR/round2_test_splits/returned/adjudications.csv \
        --run-dirs $folder --populations hybrid reviewed_judge reviewed_human \
        --write-evaluation evaluation_human \
        --out $EVAL/${dataset}__$name.csv > $EVAL/${dataset}__$name.log 2>&1 \
        && echo "ok     $dataset $name" || { echo "FAILED $dataset $name"; return 1; }
}
export -f evaluate_scorer
export HR EVAL
xargs -P 24 -L 1 bash -c 'evaluate_scorer "$@"' _ < $EVAL/scorers.txt
echo "Done at $(date)"
