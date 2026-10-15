#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --job-name=degeneration-probe-human-position
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Flag share by distance from the onset and by position (Figure 4), on the
# human-verified in-domain test split.
#
#   sbatch cluster/human_position.sbatch

set -euo pipefail
cd /iopsstor/scratch/cscs/$USER/degeneration-probe
echo "Commit: $(git rev-parse HEAD)"
.venv/bin/python scripts/position_analysis.py --human \
    --out-dir /capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe/outputs/analysis/position_analysis_human
echo "Done at $(date)"
