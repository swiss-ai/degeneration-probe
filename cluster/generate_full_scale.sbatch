#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=degeneration-probe-generate-fullscale
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

if [[ -z "${DOMAIN:-}" ]]; then
    echo "DOMAIN env var must be set (e.g. sbatch --export=DOMAIN=deepmath_103k generate_full_scale.sbatch)" >&2
    exit 1
fi

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Job: $SLURM_JOB_ID  Domain: $DOMAIN"
echo "(part of a per-domain wave chain -- see submit_full_scale_generation.sh)"

export HF_TOKEN="$(cat ~/keys/.hf_token)"

srun --environment=$REPO/cluster/env.toml \
    python -m degeneration_probe.dataset_gen.generate \
        --config configs/dataset_gen/full_scale_v1.yaml \
        --domains $DOMAIN \
        --batch-size 16

echo "Domain $DOMAIN wave done."
