#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --job-name=degeneration-probe-reader-remap
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Put a scored text-only reader run back on the generator's tokens and evaluate
# it against the generator's own onset labels:
#
#   sbatch cluster/reader_remap.sbatch <reader run dir | training job id> <generator dataset>
#
# Reads <run>/selected/layer_NN/scores, writes <run>/selected_generator/layer_NN/
# {scores,evaluation}, the layout aggregate_selected.py reads with --subdir.
set -euo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
STORE=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe
TARGET=${1:?a run directory or a training job id is required}
DATASET=${2:?the generator dataset is required}
cd $REPO
if [[ $TARGET =~ ^[0-9]+$ ]]; then
    TARGET=$(grep -m1 "^Run directory:" cluster/log/degeneration-probe-train-$TARGET.log | awk '{print $3}')
fi
RUN_DIR=$(realpath "$TARGET")
SOURCE=$STORE/degeneration-dataset-$DATASET
READER=$STORE/degeneration-dataset-$DATASET-reader-qwen2p5-1p5b
for depth in "$RUN_DIR"/selected/layer_*; do
    OUT=$RUN_DIR/selected_generator/$(basename "$depth")
    .venv/bin/python scripts/reader_scores_to_generator.py --scores-dir "$depth/scores" \
        --reader-build-root $READER --source-build-root $SOURCE --out-dir "$OUT/scores"
    .venv/bin/python scripts/evaluate_scores.py --run-dir "$OUT" --splits val \
        --onset-labels $SOURCE/onset_labels/onset_labels.parquet
    .venv/bin/python scripts/evaluate_scores.py --run-dir "$OUT" --splits test_indomain test_heldout_domains \
        --reuse-thresholds --onset-labels $SOURCE/onset_labels/onset_labels.parquet
done
