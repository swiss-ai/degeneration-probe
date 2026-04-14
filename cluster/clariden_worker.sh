#!/usr/bin/env bash
# =============================================================================
# SLURM job: inference worker on CSCS Clariden
#
# Submit:
#   sbatch cluster/clariden_worker.sh
#
# Override model/probe at submission time:
#   MODEL=Qwen/Qwen2.5-7B-Instruct sbatch cluster/clariden_worker.sh
#   PROBE=/path/to/checkpoint sbatch cluster/clariden_worker.sh
# =============================================================================
#SBATCH --job-name=degen-worker
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --environment=my_env
#SBATCH --output=/iopsstor/scratch/cscs/%u/logs/worker_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/logs/worker_%j.err

set -eo pipefail

PROJ_DIR="$HOME/degeneration-probe"
cd "${PROJ_DIR}"

mkdir -p /iopsstor/scratch/cscs/${USER}/logs

export HF_HOME="/iopsstor/scratch/cscs/${USER}/hf_cache"
export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/users/${USER}/hf_models"
mkdir -p "${HF_HOME}"


MODEL="${MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"
PROBE="${PROBE:-}"
DTYPE="${DTYPE:-bfloat16}"
PORT="${PORT:-9000}"

echo "============================================="
echo "[worker] model = ${MODEL}"
echo "[worker] probe = ${PROBE:-<none>}"
echo "[worker] dtype = ${DTYPE}"
echo "[worker] port  = ${PORT}"
echo "[worker] node  = $(hostname)"
echo "[worker] GPU   = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "To connect from your laptop:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} clariden"
echo "============================================="

pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || true


WORKER_ARGS=(
    --model "${MODEL}"
    --host 0.0.0.0
    --port "${PORT}"
    --dtype "${DTYPE}"
)

if [[ -n "${PROBE}" ]]; then
    WORKER_ARGS+=(--probe "${PROBE}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_WORKER:-0}" python -m degeneration_probe worker "${WORKER_ARGS[@]}"
