#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_docker_image
require_runtime_paths

mkdir -p "${OUTPUT_DIR}/logs"
log_file="${OUTPUT_DIR}/logs/smoke_train_$(date +%Y%m%d_%H%M%S).log"
run_training_python /package/training/train.py \
  --config /config/company.yaml \
  --smoke-test 2>&1 | tee "${log_file}"
echo "[PASS] 两轮冒烟训练完成，日志: ${log_file}"
