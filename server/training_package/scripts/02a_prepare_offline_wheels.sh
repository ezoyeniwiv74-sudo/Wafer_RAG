#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_base_image
mkdir -p "${PACKAGE_ROOT}/wheelhouse"

echo "[INFO] 此步骤需要当前机器能够访问 PyPI；只需在一台联网机器执行一次。"
docker run --rm \
  -v "${PACKAGE_ROOT}/training/requirements_company_image.txt:/requirements.txt:ro" \
  -v "${PACKAGE_ROOT}/training/constraints_company_base.txt:/constraints.txt:ro" \
  -v "${PACKAGE_ROOT}/wheelhouse:/wheelhouse" \
  --entrypoint python \
  "${BASE_IMAGE_NAME}" \
  -m pip download --only-binary=:all: --dest /wheelhouse \
  --constraint /constraints.txt -r /requirements.txt

echo "[PASS] 离线 wheel 已下载到 ${PACKAGE_ROOT}/wheelhouse"
echo "把整个部署包复制到离线服务器后，再运行 scripts/02b_build_training_image.sh。"
