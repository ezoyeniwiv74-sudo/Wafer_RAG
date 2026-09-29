#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_docker_image
require_runtime_paths

checkpoint="/outputs/${MODEL_OUTPUT_SUBDIR}/best_model.pt"
input_path="/data/${PREDICT_INPUT_REL:-predict_images}"
output_path="/outputs/${MODEL_OUTPUT_SUBDIR}/${PREDICT_OUTPUT_NAME:-predictions.csv}"
run_training_python /package/training/predict.py \
  --config /config/company.yaml \
  --checkpoint "${checkpoint}" \
  --input "${input_path}" \
  --output "${output_path}"
echo "[PASS] 推理结果: ${OUTPUT_DIR}/${MODEL_OUTPUT_SUBDIR}/${PREDICT_OUTPUT_NAME:-predictions.csv}"
