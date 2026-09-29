#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_docker_image
require_runtime_paths

mkdir -p "${OUTPUT_DIR}/logs"
log_file="${OUTPUT_DIR}/logs/full_train_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] 正式训练输出: ${OUTPUT_DIR}/${MODEL_OUTPUT_SUBDIR}"
echo "[INFO] 日志: ${log_file}"
run_training_python /package/training/train.py \
  --config /config/company.yaml "$@" 2>&1 | tee "${log_file}"
echo "[PASS] 正式训练完成。最终 checkpoint: ${OUTPUT_DIR}/${MODEL_OUTPUT_SUBDIR}/best_model.pt"
