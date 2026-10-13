#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=degeneration-probe-reader-build
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# Re-tokenize one dataset build for the text-only reader:
#
#   sbatch cluster/build_reader_dataset.sbatch <dataset> <generator tokenizer> [reader tokenizer]
set -euo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
STORE=/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe
DATASET=${1:?a dataset name is required}
GENERATOR=${2:?the generator tokenizer is required}
READER=${3:-Qwen/Qwen2.5-1.5B}
cd $REPO
echo "Commit: $(git rev-parse HEAD)   Dataset: $DATASET   Reader: $READER"
HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_TOKEN
.venv/bin/python scripts/build_reader_dataset.py \
    --source-build-root $STORE/degeneration-dataset-$DATASET \
    --out-build-root $STORE/degeneration-dataset-$DATASET-reader-qwen2p5-1p5b \
    --generator-tokenizer $GENERATOR --reader-tokenizer $READER --workers $SLURM_CPUS_PER_TASK
