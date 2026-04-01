#!/usr/bin/env bash
# =============================================================================
# SLURM job: full pipeline (fetch → generate → plot) on a GPU node
#
# Submit:
#   sbatch cluster/euler_main.sh
#
# Override any parameter at submission time, e.g.:
#   MODEL=Qwen/Qwen2.5-7B-Instruct MAX_SAMPLES=500 sbatch cluster/euler_main.sh
# =============================================================================
#SBATCH --job-name=degen-main
#SBATCH --output=/cluster/home/%u/degeneration-probe/logs/main_%j.out
#SBATCH --error=/cluster/home/%u/degeneration-probe/logs/main_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000        # 32 GB total — increase for larger models
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g         # request ≥20 GB VRAM; remove for any GPU

set -euo pipefail

module load eth_proxy             # internet access on compute nodes
module load stack/2024-06 gcc/12.2.0 python/3.11.6 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

PROJ_DIR="/cluster/home/${USER}/degeneration-probe"
cd "${PROJ_DIR}"

export HF_HOME="/cluster/scratch/${USER}/hf_cache"
mkdir -p "${PROJ_DIR}/logs"

# ----- parameters (override via env vars at sbatch time) ---------------------
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET="${DATASET:-tatsu-lab/alpaca}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-25}"
MAX_PROMPTS="${MAX_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"

echo "[job] model        = ${MODEL}"
echo "[job] dataset      = ${DATASET}/${SPLIT}  samples=${MAX_SAMPLES}  prompts=${MAX_PROMPTS}"
echo "[job] max_new_tok  = ${MAX_NEW_TOKENS}  dtype=${DTYPE}"
echo "[job] HF_HOME      = ${HF_HOME}"
echo "[job] GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python scripts/main.py \
  --dataset        "${DATASET}" \
  --split          "${SPLIT}" \
  --max_samples    "${MAX_SAMPLES}" \
  --max_prompts    "${MAX_PROMPTS}" \
  --model_name     "${MODEL}" \
  --device         cuda \
  --torch_dtype    "${DTYPE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --token_path     "${HOME}/.hf_token"

echo "[job] Done."
