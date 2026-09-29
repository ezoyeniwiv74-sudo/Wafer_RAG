#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_docker_image
require_runtime_paths

mkdir -p "${OUTPUT_DIR}/logs"
log_file="${OUTPUT_DIR}/logs/preflight_$(date +%Y%m%d_%H%M%S).log"
run_training_python /package/training/preflight.py --config /config/company.yaml 2>&1 | tee "${log_file}"
echo "[PASS] 预检查通过，日志: ${log_file}"
