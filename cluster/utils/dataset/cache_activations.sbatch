#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=degeneration-probe-cache-activations-fullscale
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
mkdir -p $REPO/cluster/log
cd $REPO

echo "Commit: $(git rev-parse HEAD)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Job: $SLURM_JOB_ID"

export HF_TOKEN="$(cat ~/keys/.hf_token)"

# All 6 domains run sequentially in ONE process (no --domains filter) --
# activations/manifest.parquet is a single file shared across all domains
# (unlike generate.py/label.py's per-domain shards), so running domains as
# separate concurrent SLURM jobs would race on read-modify-write of that
# file and silently lose rows. Resumable: a rerun skips any (prompt_id,
# rollout_idx) already recorded with status "ok".
srun --environment=$REPO/cluster/env.toml \
    python -m degeneration_probe.dataset_gen.cache_activations \
        --config configs/dataset/full_scale.yaml

echo "Activation caching done."
