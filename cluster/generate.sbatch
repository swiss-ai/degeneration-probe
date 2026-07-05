#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --job-name=degeneration-probe-generate
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --array=0-5

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Array task: $SLURM_ARRAY_TASK_ID"

export HF_TOKEN="$(cat ~/keys/.hf_token)"

# Pilot v1's 6 sources (4 in-domain, 2 held-out) -- see configs/dataset_gen/pilot_v1.yaml.
# One array task per domain so all 6 generate in parallel on separate GPUs;
# generate.py's resumability makes each task independently safe to retry.
DOMAINS=(deepmath_103k numinamath_1_5 if_sft_data_verified llama_nemotron medical_o1 aime_2025)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
echo "Domain: $DOMAIN"

srun --environment=$REPO/cluster/env.toml \
    python -m degeneration_probe.dataset_gen.generate \
        --config configs/dataset_gen/pilot_v1.yaml \
        --domains $DOMAIN \
        --batch-size 16

echo "Array task $SLURM_ARRAY_TASK_ID ($DOMAIN) done."
