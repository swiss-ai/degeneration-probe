#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=degeneration-probe-generate
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

if [[ -z "${DOMAIN:-}" ]]; then
    echo "DOMAIN env var must be set (e.g. sbatch --export=DOMAIN=deepmath_103k cluster/utils/dataset/generate.sbatch)" >&2
    exit 1
fi

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe

CONFIG="${CONFIG:-configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml}"

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Job: $SLURM_JOB_ID  Domain: $DOMAIN  Config: $CONFIG"
echo "(part of a per-domain wave chain -- see cluster/utils/dataset/submit_generation.sh)"

export HF_TOKEN="$(cat ~/keys/.hf_token)"

srun --environment=$REPO/cluster/env.toml \
    python -m degeneration_probe.dataset_gen.generate \
        --config "$CONFIG" \
        --domains $DOMAIN \
        --batch-size 16

echo "Domain $DOMAIN wave done."
