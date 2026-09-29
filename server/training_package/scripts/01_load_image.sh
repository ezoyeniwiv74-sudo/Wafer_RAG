#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_command docker
require_command sha256sum
[[ -f "${IMAGE_TAR_PATH}" ]] || {
  echo "[ERROR] 未找到公司镜像: ${IMAGE_TAR_PATH}" >&2
  echo "请把公司原始 dinov2_cuda12.1_large.tar 放到 docker_image/。" >&2
  exit 1
}

echo "========== 公司镜像文件 =========="
ls -lh "${IMAGE_TAR_PATH}"
actual_sha256="$(sha256sum "${IMAGE_TAR_PATH}" | awk '{print toupper($1)}')"
printf '%s  %s\n' "${actual_sha256}" "${IMAGE_TAR_PATH}" | tee "${IMAGE_TAR_PATH}.sha256.actual"
if [[ -n "${EXPECTED_IMAGE_SHA256:-}" ]] && \
   [[ "${actual_sha256}" != "${EXPECTED_IMAGE_SHA256^^}" ]]; then
  echo "[ERROR] 公司镜像 SHA256 不匹配。" >&2
  echo "期望: ${EXPECTED_IMAGE_SHA256^^}" >&2
  echo "实际: ${actual_sha256}" >&2
  echo "停止加载；请检查上传是否损坏或文件是否被替换。" >&2
  exit 1
fi
echo "[PASS] 公司镜像 SHA256 匹配"

echo "========== docker load =========="
load_output="$(docker load -i "${IMAGE_TAR_PATH}" 2>&1)"
printf '%s\n' "${load_output}"

echo "========== 当前 DINOv2 相关镜像 =========="
docker image ls --no-trunc | grep -Ei 'dino|repository' || docker image ls --no-trunc

echo "[IMPORTANT] 检查上面的 Loaded image 标签。"
echo "本交付包已解析确认，公司 tar 的原始标签应为 ${BASE_IMAGE_NAME}。"
echo "如输出不同，请只修改 deploy.env 中的 BASE_IMAGE_NAME；不要改训练镜像 IMAGE_NAME。"
