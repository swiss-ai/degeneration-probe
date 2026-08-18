#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --job-name=degeneration-probe-backfill-bands
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Fill in coverage of the run-up at every reported width, for runs that were
# replayed when fewer widths were being recorded:
#
#   sbatch cluster/backfill_bands.sbatch outputs/<run>/latest [more run dirs...]
#
# The threshold each checkpoint was measured at is recovered from the healthy
# answers whose peaks the replay kept, so only the degenerate answers are read
# again. That is a hundred-odd answers against three and a half thousand, which
# is why this costs minutes per run where the replay cost hours. Several runs
# fit comfortably in one job.

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
mkdir -p $REPO/cluster/log
cd $REPO

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Runs:   $#"

HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_TOKEN

START=$SECONDS
for RUN_DIR in "$@"; do
    echo "=== $RUN_DIR"
    srun --environment=$REPO/cluster/env.toml \
        python scripts/backfill_warning_bands.py --run-dir "$RUN_DIR"
done
echo "ELAPSED backfill: $((SECONDS - START)) s"
