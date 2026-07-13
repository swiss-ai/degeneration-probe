#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --job-name=onset-lora-eval
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Scores one LoRA onset-probe checkpoint (subtask I/J) on test_indomain +
# test_heldout_domains -- the two splits train_onset_lora_pilot.py never
# evaluates on (it only scores val). Probe id is the sbatch script's first
# positional arg, e.g.:
#   sbatch cluster/evaluate_onset_lora_probes.sbatch onset_multi_horizon_lora_pilot_layer16
#   sbatch cluster/evaluate_onset_lora_probes.sbatch onset_probe_5_lora_pilot_layer16

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
PROBE_ID="${1:?Usage: sbatch evaluate_onset_lora_probes.sbatch <probe_id>}"

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
echo "Probe id: $PROBE_ID"

export HF_TOKEN="$(cat ~/keys/.hf_token)"

srun --environment=$REPO/cluster/env.toml \
    bash -c "PYTHONPATH=$REPO:\$PYTHONPATH python scripts/evaluate_onset_lora_probes.py --probe-id $PROBE_ID"
