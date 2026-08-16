#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --job-name=degeneration-probe-onset-labels
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
mkdir -p $REPO/cluster/log
cd $REPO

CONFIG="${CONFIG:-configs/dataset/builds/degeneration-dataset-apertus-8b-instruct.yaml}"

echo "Commit: $(git rev-parse HEAD)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Job: $SLURM_JOB_ID  Config: $CONFIG"

# onset_quotes.py/onset_labels.py are CPU-only (locate the judge's onset_quote
# in the token stream, then materialize the resolved onset per rollout) --
# runs directly against the repo's .venv, same as label.sbatch. Needs the
# tokenizer, so needs a token when model_name/tokenizer_name is a gated repo.
source .venv/bin/activate
export HF_TOKEN="$(cat ~/keys/.hf_token)"

python -m degeneration_probe.dataset_gen.onset_quotes --config "$CONFIG"
python -m degeneration_probe.dataset_gen.onset_labels --config "$CONFIG"

echo "Onset materialization done."
