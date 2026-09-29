#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_base_image

echo "========== 构建训练镜像 =========="
echo "基础镜像: ${BASE_IMAGE_NAME}"
echo "训练镜像: ${IMAGE_NAME}"

docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_NAME}" \
  --file "${PACKAGE_ROOT}/docker/Dockerfile.company-training" \
  --tag "${IMAGE_NAME}" \
  "${PACKAGE_ROOT}"

echo "========== 验证训练镜像 =========="
docker run --rm --entrypoint python "${IMAGE_NAME}" -c \
  "import torch, pandas, sklearn, matplotlib, yaml; print('torch=', torch.__version__); print('cuda_build=', torch.version.cuda); print('dependencies=PASS')"

echo "[PASS] 训练镜像已生成: ${IMAGE_NAME}"
