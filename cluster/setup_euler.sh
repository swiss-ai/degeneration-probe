#!/usr/bin/env bash
# =============================================================================
# One-time setup — run interactively on an Euler login node
# =============================================================================
set -euo pipefail

# Internet access on Euler
module load eth_proxy 2>/dev/null || true

# Python (adjust version if the cluster has updated its stack)
module load stack/2024-06 gcc/12.2.0 python/3.11.6 2>/dev/null || true

# Install uv into ~/.local/bin if not already present
if ! command -v uv &>/dev/null; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "[setup] uv $(uv --version)"

# Point HuggingFace cache to scratch (models can be many GB)
HF_CACHE="/cluster/scratch/${USER}/hf_cache"
mkdir -p "${HF_CACHE}"
grep -qxF "export HF_HOME=${HF_CACHE}" ~/.bashrc \
  || echo "export HF_HOME=${HF_CACHE}" >> ~/.bashrc
export HF_HOME="${HF_CACHE}"
echo "[setup] HF cache → ${HF_CACHE}"

# Remind user to place the HF token
TOKEN="${HOME}/.hf_token"
if [[ ! -f "${TOKEN}" ]]; then
  echo ""
  echo "[setup] HF token not found. Create it with:"
  echo "  printf '%s' 'hf_...' > ${TOKEN} && chmod 600 ${TOKEN}"
else
  echo "[setup] HF token found at ${TOKEN}"
fi

# Install Python dependencies
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "[setup] Syncing env in ${PROJ_DIR}..."
cd "${PROJ_DIR}"
uv sync --frozen

echo ""
echo "[setup] Done. Submit a job with:  CONFIG_FILE=alpaca_qwen05b.yaml sbatch cluster/euler_run.sh"
