#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CONFIG_PATH="config/dacon.json"
if [[ $# -ge 1 ]]; then
  CONFIG_PATH="$1"
else
  if [[ -t 0 ]]; then
    read -r -p "Config path [${DEFAULT_CONFIG_PATH}]: " INPUT_CONFIG_PATH
    CONFIG_PATH="${INPUT_CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}"
  else
    CONFIG_PATH="${DEFAULT_CONFIG_PATH}"
  fi
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

LOG_FILE="$(mktemp -t main_run_XXXX.log)"
trap 'rm -f "${LOG_FILE}"' EXIT

echo "[INFO] running main pipeline with config=${CONFIG_PATH}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u -m main --config "${CONFIG_PATH}" | tee "${LOG_FILE}"

RUN_ID="$(awk -F': ' '/^Run ID:/ {print $2}' "${LOG_FILE}" | tail -n1 | tr -d '\r')"
if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] failed to parse Run ID from main output." >&2
  echo "[HINT] check pipeline logs above. submission step was skipped." >&2
  exit 1
fi

echo "[INFO] parsed run_id=${RUN_ID}"
echo "[INFO] running submission generation"
"${PYTHON_BIN}" -m submission --config "${CONFIG_PATH}" --run_id "${RUN_ID}"
