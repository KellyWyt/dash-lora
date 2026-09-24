#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SESSION_NAME="${GPU_USE_SESSION:-gpu-use}"
DURATION="${1:-7h}"
GPU_IDS="${2:-0,1,2,3}"
BUSY_PERCENT="${3:-100}"
MATRIX_SIZE="${4:-8192}"
MEMORY_MIB="${5:-23000}"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}.log"
PYTHON_BIN="/nfs1/miniconda3/envs/moka/bin/python"

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed or unavailable in PATH." >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: moka Python not found at ${PYTHON_BIN}" >&2
    exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Error: tmux session '${SESSION_NAME}' already exists." >&2
    echo "Attach: tmux attach -t ${SESSION_NAME}" >&2
    echo "Stop:   tmux kill-session -t ${SESSION_NAME}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
printf -v RUN_COMMAND \
    'cd %q && exec %q %q --duration %q --gpus %q --busy-percent %q --matrix-size %q --memory-mib %q 2>&1 | tee -a %q' \
    "${PROJECT_DIR}" "${PYTHON_BIN}" "${SCRIPT_DIR}/keep_gpus_busy.py" \
    "${DURATION}" "${GPU_IDS}" "${BUSY_PERCENT}" "${MATRIX_SIZE}" "${MEMORY_MIB}" "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "${RUN_COMMAND}"

echo "Started tmux session '${SESSION_NAME}'."
echo "Duration: ${DURATION}; GPUs: ${GPU_IDS}; busy: ${BUSY_PERCENT}%"
echo "Target memory per GPU: ${MEMORY_MIB} MiB"
echo "Log:    ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Logs:   tail -f ${LOG_FILE}"
echo "Stop:   tmux kill-session -t ${SESSION_NAME}"
