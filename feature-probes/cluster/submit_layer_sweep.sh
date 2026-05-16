#!/bin/bash
# Wrapper that generates a single SWEEP_TIMESTAMP and submits the array job.
# All 3 array tasks (10 probes total) will share the same output dir:
#   /capstor/scratch/cscs/$USER/feature-probes/output/<SWEEP_TIMESTAMP>/
#
# Usage:  bash cluster/submit_layer_sweep.sh
set -euo pipefail

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe/feature-probes
SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "Submitting layer sweep with SWEEP_TIMESTAMP=$SWEEP_TIMESTAMP"
echo "Layers: [10, 14, 18, 22, 24, 26, 28, 29, 30, 31]"
echo "Output will be under: /capstor/scratch/cscs/$USER/feature-probes/output/$SWEEP_TIMESTAMP/"

sbatch --export=ALL,SWEEP_TIMESTAMP="$SWEEP_TIMESTAMP" \
       "$REPO/cluster/lorenzo-layer-sweep.sbatch"
