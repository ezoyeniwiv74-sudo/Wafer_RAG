#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${DEPLOY_ENV_FILE:-${PACKAGE_ROOT}/deploy.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] 找不到部署参数文件: ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${IMAGE_NAME:?请在 deploy.env 设置 IMAGE_NAME}"
: "${BASE_IMAGE_NAME:=dinov2:cuda12.1-large}"
: "${DATA_DIR:?请在 deploy.env 设置 DATA_DIR}"
: "${OUTPUT_DIR:?请在 deploy.env 设置 OUTPUT_DIR}"
: "${CONFIG_FILE:?请在 deploy.env 设置 CONFIG_FILE}"
: "${GPU_SPEC:=all}"
: "${SHM_SIZE:=8g}"
: "${MODEL_OUTPUT_SUBDIR:=company_dinov2_large}"
: "${RUN_AS_HOST_USER:=true}"

resolve_package_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s\n' "${PACKAGE_ROOT}/$1"
  fi
}

CONFIG_PATH="$(resolve_package_path "${CONFIG_FILE}")"
IMAGE_TAR_PATH="$(resolve_package_path "${IMAGE_TAR:-docker_image/dinov2_cuda12.1_large.tar}")"
DATA_DIR="$(resolve_package_path "${DATA_DIR}")"
OUTPUT_DIR="$(resolve_package_path "${OUTPUT_DIR}")"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] 缺少命令: $1" >&2
    exit 1
  }
}

require_docker_image() {
  require_command docker
  docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 || {
    echo "[ERROR] 找不到训练镜像 ${IMAGE_NAME}。" >&2
    echo "        请先运行 bash scripts/01_load_image.sh，再运行 bash scripts/02b_build_training_image.sh。" >&2
    exit 1
  }
}

require_base_image() {
  require_command docker
  docker image inspect "${BASE_IMAGE_NAME}" >/dev/null 2>&1 || {
    echo "[ERROR] 找不到公司基础镜像 ${BASE_IMAGE_NAME}；先运行 bash scripts/01_load_image.sh。" >&2
    exit 1
  }
}

require_runtime_paths() {
  [[ -d "${DATA_DIR}" ]] || { echo "[ERROR] DATA_DIR 不存在: ${DATA_DIR}" >&2; exit 1; }
  [[ -f "${CONFIG_PATH}" ]] || { echo "[ERROR] 配置不存在: ${CONFIG_PATH}" >&2; exit 1; }
  mkdir -p "${OUTPUT_DIR}"
}

container_user_args=()
if [[ "${RUN_AS_HOST_USER}" == "true" ]]; then
  container_user_args=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
fi

run_training_python() {
  docker run --rm \
    --gpus "${GPU_SPEC}" \
    --shm-size "${SHM_SIZE}" \
    "${container_user_args[@]}" \
    -v "${PACKAGE_ROOT}:/package:ro" \
    -v "${DATA_DIR}:/data:ro" \
    -v "${OUTPUT_DIR}:/outputs" \
    -v "${CONFIG_PATH}:/config/company.yaml:ro" \
    --entrypoint python \
    "${IMAGE_NAME}" "$@"
}
