#!/usr/bin/env bash
# Start the full stack: Clariden worker + SSH tunnel + local backend + local UI.
#
# Defaults target the `debug` partition (fast queue, 1h cap) and the Apertus-8B
# base model. Override via env vars:
#   MODEL=Qwen/Qwen2.5-7B-Instruct PARTITION=normal TIME=04:00:00 ./cluster/start.sh
#
# Requires: ssh clariden working (cscs-key loaded in your agent) and `uv` locally.
# Runtime state (PIDs, jobid, logs) is written to .run/ — used by stop.sh.
set -euo pipefail

MODEL="${MODEL:-swiss-ai/Apertus-8B-2509}"
PARTITION="${PARTITION:-debug}"
TIME="${TIME:-01:00:00}"
WORKER_PORT="${WORKER_PORT:-9000}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
UI_PORT="${UI_PORT:-7860}"
SSH_HOST="${SSH_HOST:-clariden}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${REPO_DIR}/.run"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Refuse to double-start.
if [[ -s "${RUN_DIR}/jobid" || -s "${RUN_DIR}/tunnel.pid" || -s "${RUN_DIR}/backend.pid" || -s "${RUN_DIR}/ui.pid" ]]; then
    echo "error: runtime state exists in ${RUN_DIR}. Run ./cluster/stop.sh first." >&2
    exit 1
fi

echo "[start] verifying ssh to ${SSH_HOST}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${SSH_HOST}" true 2>/dev/null; then
    echo "error: ssh ${SSH_HOST} failed. Load your cscs-key into the agent:" >&2
    echo "  ssh-add -t 1d ~/.ssh/cscs-key" >&2
    exit 1
fi

echo "[start] syncing repo on ${SSH_HOST} to origin/serving..."
ssh "${SSH_HOST}" 'cd ~/degeneration-probe && git fetch --quiet origin serving && git checkout --quiet serving && git reset --quiet --hard origin/serving && git log -1 --oneline'

echo "[start] submitting worker job (model=${MODEL}, partition=${PARTITION}, time=${TIME})..."
JOBID=$(ssh "${SSH_HOST}" "cd ~/degeneration-probe && MODEL='${MODEL}' PORT='${WORKER_PORT}' sbatch --parsable --partition='${PARTITION}' --time='${TIME}' cluster/clariden_worker.sh")
if [[ -z "${JOBID}" ]]; then
    echo "error: sbatch returned empty jobid" >&2
    exit 1
fi
echo "${JOBID}" > "${RUN_DIR}/jobid"
echo "[start] submitted job ${JOBID}"

echo "[start] waiting for job to start running..."
for _ in $(seq 1 60); do
    state=$(ssh "${SSH_HOST}" "squeue -h -j ${JOBID} -o %T" 2>/dev/null || true)
    [[ "${state}" == "RUNNING" ]] && break
    sleep 5
done
if [[ "${state}" != "RUNNING" ]]; then
    echo "error: job ${JOBID} did not enter RUNNING within 5 min (state=${state})" >&2
    echo "check: ssh ${SSH_HOST} 'squeue -j ${JOBID}; scontrol show job ${JOBID}'" >&2
    exit 1
fi

NODE=$(ssh "${SSH_HOST}" "squeue -h -j ${JOBID} -o %N")
echo "${NODE}" > "${RUN_DIR}/node"
echo "[start] job running on node ${NODE}"

echo "[start] waiting for worker to listen on ws://${NODE}:${WORKER_PORT} (model download + load)..."
for _ in $(seq 1 60); do
    if ssh "${SSH_HOST}" "grep -q 'Inference worker listening' /iopsstor/scratch/cscs/\$USER/logs/worker_${JOBID}.err 2>/dev/null"; then
        break
    fi
    sleep 10
done
if ! ssh "${SSH_HOST}" "grep -q 'Inference worker listening' /iopsstor/scratch/cscs/\$USER/logs/worker_${JOBID}.err 2>/dev/null"; then
    echo "error: worker did not come up within 10 min" >&2
    echo "check: ssh ${SSH_HOST} 'tail -50 /iopsstor/scratch/cscs/\$USER/logs/worker_${JOBID}.err'" >&2
    exit 1
fi
echo "[start] worker listening"

echo "[start] opening SSH tunnel localhost:${WORKER_PORT} -> ${NODE}:${WORKER_PORT}..."
# -f: fork to background after auth; keepalive flags prevent idle timeouts
# killing the tunnel mid-session.
ssh -f -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L "${WORKER_PORT}:${NODE}:${WORKER_PORT}" \
    "${SSH_HOST}"
pgrep -f "ssh -f -N.*-L ${WORKER_PORT}:${NODE}:${WORKER_PORT}" > "${RUN_DIR}/tunnel.pid"

echo "[start] starting local backend on :${BACKEND_PORT}..."
cd "${REPO_DIR}"
uv run python -m degeneration_probe serve >"${LOG_DIR}/backend.log" 2>&1 &
echo $! > "${RUN_DIR}/backend.pid"

echo "[start] starting local UI on :${UI_PORT}..."
uv run python -m degeneration_probe ui >"${LOG_DIR}/ui.log" 2>&1 &
echo $! > "${RUN_DIR}/ui.pid"

echo ""
echo "=============================================="
echo " stack is up"
echo "   UI:      http://localhost:${UI_PORT}"
echo "   Backend: http://localhost:${BACKEND_PORT}"
echo "   Worker:  ws://localhost:${WORKER_PORT}  (-> ${NODE}:${WORKER_PORT})"
echo "   Job:     ${JOBID} on ${NODE}"
echo "   Logs:    ${LOG_DIR}/{tunnel,backend,ui}.log"
echo "   Stop:    ./cluster/stop.sh"
echo "=============================================="
