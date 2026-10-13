#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --job-name=degeneration-probe-token-logprobs
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1

# The generator's log-probability of each token it sampled, for the windowed
# negative log-probability baseline:
#
#   sbatch cluster/token_logprobs.sbatch <dataset> <hf model id>
#
# Four shards run in parallel, one per GPU, and write
# <build>/token_logprobs/shard_<i>_of_4.parquet.
set -euo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
STORE=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe
DATASET=${1:?a dataset name is required}
MODEL=${2:?a model id is required}
BUILD_ROOT=$STORE/degeneration-dataset-$DATASET
cd $REPO
echo "Commit: $(git rev-parse HEAD)   Dataset: $DATASET   Model: $MODEL"
HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_TOKEN
srun --environment=$REPO/cluster/env.toml bash -c "
    cd $REPO
    python scripts/compute_token_logprobs.py --build-root $BUILD_ROOT --model-name $MODEL \
        --shard \$SLURM_PROCID --num-shards \$SLURM_NTASKS
"
