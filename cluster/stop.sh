#!/usr/bin/env bash
# Tear down the full stack started by cluster/start.sh.
set -uo pipefail

SSH_HOST="${SSH_HOST:-clariden}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${REPO_DIR}/.run"

kill_pidfile() {
    local name="$1" file="${RUN_DIR}/$2"
    if [[ -s "${file}" ]]; then
        local pid
        pid=$(cat "${file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[stop] killing ${name} (pid ${pid})"
            kill "${pid}" 2>/dev/null || true
            sleep 1
            kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${file}"
    fi
}

kill_pidfile ui ui.pid
kill_pidfile backend backend.pid
kill_pidfile tunnel tunnel.pid

if [[ -s "${RUN_DIR}/jobid" ]]; then
    JOBID=$(cat "${RUN_DIR}/jobid")
    echo "[stop] cancelling SLURM job ${JOBID} on ${SSH_HOST}"
    ssh -o BatchMode=yes "${SSH_HOST}" "scancel ${JOBID}" 2>/dev/null || echo "[stop] warning: scancel failed (ssh unreachable?)"
    rm -f "${RUN_DIR}/jobid" "${RUN_DIR}/node"
fi

echo "[stop] done"
