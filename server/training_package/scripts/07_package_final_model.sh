#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_command sha256sum
require_command tar

model_dir="${OUTPUT_DIR}/${MODEL_OUTPUT_SUBDIR}"
checkpoint="${model_dir}/best_model.pt"
[[ -f "${checkpoint}" ]] || { echo "[ERROR] 找不到最终模型: ${checkpoint}" >&2; exit 1; }

stamp="$(date +%Y%m%d_%H%M%S)"
release_root="${OUTPUT_DIR}/releases"
release_dir="${release_root}/${MODEL_OUTPUT_SUBDIR}_${stamp}"
archive="${release_dir}.tar.gz"
mkdir -p "${release_dir}"

for name in best_model.pt resolved_config.json test_metrics.json classification_report.json confusion_matrix.png training_history.csv split_distribution.json splits.csv; do
  if [[ -f "${model_dir}/${name}" ]]; then cp -p "${model_dir}/${name}" "${release_dir}/"; fi
done
cp -p "${CONFIG_PATH}" "${release_dir}/config.company.yaml"
sha256sum "${release_dir}/best_model.pt" > "${release_dir}/CHECKSUMS.sha256"

cat > "${release_dir}/模型使用说明.txt" <<EOF
best_model.pt 是分类头以及可选解冻 block 的增量 checkpoint，不是独立 DINOv2 主干。
部署推理时必须同时保留：
1. 公司镜像 ${IMAGE_NAME}
2. 镜像内与训练一致的 DINOv2-L 主干权重
3. config.company.yaml / resolved_config.json 中的类别顺序
4. best_model.pt
EOF

tar -C "${release_root}" -czf "${archive}" "$(basename "${release_dir}")"
sha256sum "${archive}" > "${archive}.sha256"
echo "[PASS] 最终模型交付包: ${archive}"
echo "[PASS] 校验文件: ${archive}.sha256"
