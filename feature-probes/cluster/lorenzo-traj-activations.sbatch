#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --job-name=lorenzo-traj-activations
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/degeneration-probe/feature-probes/cluster/log/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe/feature-probes
mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"

export HF_TOKEN="$(cat ~/keys/.hf_token)"
[[ -f ~/keys/.hf_token_write ]] && export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="/capstor/scratch/cscs/$USER/feature-probes/output/activation_traj_${RUN_TIMESTAMP}_job${SLURM_JOB_ID}"
mkdir -p "$OUT_BASE"
echo "OUT_BASE=$OUT_BASE"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

ROW=${ROW:-612}
# LoRA adapter trained jointly with the layer-30 probe (LoRA hits ALL decoder layers).
LORA_PATH=${LORA_PATH:-/capstor/scratch/cscs/lbaggi/feature-probes/output/20260515_180522/apertus_apertus_stability_repetition_balanced_layer_sweep_layer30}

echo "=== 1/3 Extract activations: BASE model ==="
srun --environment=$REPO/cluster/env.toml \
    python scripts/extract_traj_activations.py \
        --row "$ROW" \
        --out-dir "$OUT_BASE" \
        --out-name "row${ROW}_base"

echo "=== 2/3 Extract activations: BASE + LoRA ==="
srun --environment=$REPO/cluster/env.toml \
    python scripts/extract_traj_activations.py \
        --row "$ROW" \
        --lora-path "$LORA_PATH" \
        --out-dir "$OUT_BASE" \
        --out-name "row${ROW}_lora"

echo "=== 3/3 Plot 2D projections (PCA / UMAP / t-SNE) ==="
srun --environment=$REPO/cluster/env.toml \
    python scripts/plot_traj_activations.py \
        --inputs "$OUT_BASE/row${ROW}_base.npz" "$OUT_BASE/row${ROW}_lora.npz" \
        --out-dir "$OUT_BASE/plots"

echo "All done. Outputs in $OUT_BASE"
ls -la "$OUT_BASE" "$OUT_BASE/plots"
