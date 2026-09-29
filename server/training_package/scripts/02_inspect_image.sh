#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_base_image

echo "========== 镜像元数据 =========="
docker image inspect "${BASE_IMAGE_NAME}" --format 'Id={{.Id}} Architecture={{.Architecture}} OS={{.Os}} Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}}'

echo "========== Python / PyTorch / GPU =========="
docker run --rm --gpus "${GPU_SPEC}" --entrypoint python "${BASE_IMAGE_NAME}" -c \
  "import sys, torch; print('python=', sys.version); print('torch=', torch.__version__); print('built_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "========== 搜索镜像内 DINOv2 权重 =========="
docker run --rm --entrypoint sh "${BASE_IMAGE_NAME}" -lc '
  for d in /workspace /models /model /opt /app /root; do
    if [ -d "$d" ]; then
      find "$d" -maxdepth 7 -type f \( -iname "*dinov2*vitl*.pth" -o -iname "*dinov2*large*.pth" -o -iname "*vitl14*.pth" \) -print 2>/dev/null
    fi
  done
' | tee "${PACKAGE_ROOT}/image_inspection.txt"

echo "已解析的公司镜像应找到 /app/weights/dinov2_vitl14_pretrain.pth。"
echo "若找不到该文件或路径变化，先不要训练，并核对 tar 的 SHA256。"
