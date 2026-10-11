#!/bin/bash
# Build the container image from the local working tree (no GitHub access needed):
#   srun --account=infra01 --partition=normal --time=01:30:00 bash cluster/build_local.sh [tag-suffix]
# Writes $SCRATCH/ce-images/degeneration-probe+25.06-<tag-suffix>.sqsh, the path env.toml reads.
set -euo pipefail
trap 'echo "Build failed at $(date)" >&2' ERR

SUFFIX=${1:-dataset-v2-pilot}
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
TAG="degeneration-probe:${SUFFIX}"
CE_IMAGE_DIR=${SCRATCH}/ce-images
OUT=${CE_IMAGE_DIR}/degeneration-probe+25.06-${SUFFIX}.sqsh

echo "Host:    $(hostname)   Commit: $(git -C $REPO rev-parse --short HEAD)"
echo "Image:   $OUT"
mkdir -p ${CE_IMAGE_DIR}

podman build --no-cache -f $REPO/cluster/Dockerfile.local -t $TAG $REPO

rm -f $OUT
enroot import -o $OUT podman://${TAG}
[[ -s $OUT ]] && echo "Done: $OUT  ($(du -h $OUT | cut -f1))" || { echo "ERROR: sqsh missing or empty" >&2; exit 1; }
