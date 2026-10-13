#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --job-name=degeneration-probe-paper-figure-states
#SBATCH --output=/iopsstor/scratch/cscs/%u/degeneration-probe/cluster/log/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

# Frozen and adapted hidden states, at every layer, of the rollout the paper's
# PCA and parallel-coordinates figures show. The adapted model is the
# layer-15 S4 LoRA probe whose seed has the best validation warning recall at
# 256 tokens among the selectable seeds, read at its selected checkpoint:
#
#   sbatch --dependency=afterany:<score-adapted jobs> cluster/extract_paper_figure_states.sbatch
#
# Files land where notebooks/paper_figures.py looks for them.
set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR
REPO=/iopsstor/scratch/cscs/$USER/degeneration-probe
OUTPUTS=/capstor/store/cscs/swissai/infra01/users/$USER/degeneration-probe/outputs
OUT_DIR=${1:-$OUTPUTS/paper_figures/activations}
ROLLOUT=deepmath_103k_00196:8
mkdir -p $REPO/cluster/log
cd $REPO
echo "Commit: $(git -C $REPO rev-parse HEAD)"

read -r FROZEN_RUN LORA_CHECKPOINT < <(.venv/bin/python - "$OUTPUTS" <<'EOF'
import json, sys
from pathlib import Path

outputs = Path(sys.argv[1])
recipe = dict(model="apertus", selection="frontier_window", window=512, anchor="centered",
              label="frontier_hard", horizon=256, loss="bce")

def attempts(lora):
    for info_path in sorted(outputs.glob("*/latest/run_info.json")):
        info = json.loads(info_path.read_text())
        axes = info.get("axes", {})
        if info.get("status") == "finished" and axes.get("lora") == lora and all(axes.get(k) == v for k, v in recipe.items()):
            yield info_path.parent.resolve(), axes

candidates = []
for run_dir, axes in attempts("all"):
    selection_path = run_dir / "checkpoint_selection.json"
    if axes.get("layer") != 15 or not selection_path.is_file():
        continue
    depth = json.loads(selection_path.read_text())["depths"][0]
    print(f"{run_dir.parent.name}: step {depth['selected_step']}, selectable {depth['selectable']}, "
          f"warning_recall_256 {depth['warning_recall_256']:.4f}", file=sys.stderr)
    candidates.append((bool(depth["selectable"]), float(depth["warning_recall_256"]), run_dir, depth, axes["seed"]))
if not candidates:
    sys.exit("no finished layer-15 S4 LoRA run with a checkpoint selection")
_, _, run_dir, depth, seed = max(candidates, key=lambda c: (c[0], c[1]))
checkpoint = run_dir / f"checkpoint-{depth['selected_step']}"
if not (checkpoint / "adapter_model.safetensors").is_file():
    sys.exit(f"{checkpoint} holds no LoRA adapter")

frozen = sorted(attempts("none"), key=lambda a: a[1].get("seed") != seed)
if not frozen:
    sys.exit("no finished frozen S4 run to read the dataset settings from")
print(frozen[0][0], checkpoint)
EOF
)
echo "Frozen run:      $FROZEN_RUN"
echo "LoRA checkpoint: $LORA_CHECKPOINT"
echo "Output:          $OUT_DIR"

HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_TOKEN

srun --environment=$REPO/cluster/env.toml \
    python scripts/extract_lora_linearity_activations.py \
        --frozen-run-dir "$FROZEN_RUN" \
        --lora-checkpoint "$LORA_CHECKPOINT" \
        --out-dir "$OUT_DIR" \
        --rollouts "$ROLLOUT" \
        --sanity-rollouts 1
