#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:15:00
#SBATCH --job-name=degeneration-probe-pca-states
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G

# Frozen and adapted hidden states at the frozen probe's layer of each model,
# for the PCA of degenerate and healthy text, one GPU per model:
#
#   sbatch cluster/extract_pca_states.sbatch
set -uo pipefail
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
cd $REPO
echo "Commit: $(git rev-parse HEAD)  start $(date)"
HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_TOKEN
for spec in ${SPECS:-apertus:15 llama3p1:15 mistral:5}; do
    srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --mem=120G --exact --environment=$REPO/cluster/env.toml \
        bash -c "cd $REPO && python scripts/extract_pca_states.py --model ${spec%%:*} --layer ${spec##*:} ${EXTRA:-}" \
        > cluster/log/pca-states-${spec%%:*}-$SLURM_JOB_ID.log 2>&1 &
done
wait
echo "done $(date)"
