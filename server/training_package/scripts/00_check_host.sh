#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

require_command nvidia-smi
require_command docker

echo "========== 操作系统 =========="
uname -a
if [[ -r /etc/os-release ]]; then cat /etc/os-release; fi

echo "========== NVIDIA 驱动/GPU =========="
nvidia-smi

echo "========== Docker =========="
docker version
docker info --format 'Runtimes={{json .Runtimes}}'
docker info >/dev/null

echo "========== 磁盘 =========="
df -h "${PACKAGE_ROOT}" "$(dirname "${OUTPUT_DIR}")" 2>/dev/null || df -h "${PACKAGE_ROOT}"

echo "[PASS] 宿主机基础检查完成。下一步：bash scripts/01_load_image.sh"
