#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/packages/common/common.sh"

MODEL_DIR="${ROOT_DIR}/models/squeezenet"
PACKAGE_NAME="squeezenet.aimod"
MODEL_URL="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-9.onnx"

: "${PREFERRED_PROVIDER:=cpu}"
: "${FALLBACK_PROVIDERS:=}"
: "${REQUIRED_PRECISION:=fp32}"
: "${GRAPH_OPT_LEVEL:=all}"
: "${DEVICE_ID:=0}"
: "${STRATEGY:=pool}"

PREFERRED_MODELS=(
    "squeezenet.onnx"
)

SEARCH_DIRS=(
    "${MODEL_DIR}"
)

mkdir -p "${MODEL_DIR}"

MODEL_PATH="$(select_model_file || true)"
if [ -z "${MODEL_PATH}" ]; then
    echo "Downloading SqueezeNet model..."
    curl -L -o "${MODEL_DIR}/squeezenet.onnx" "${MODEL_URL}"
    MODEL_PATH="${MODEL_DIR}/squeezenet.onnx"
fi

stage_init
stage_model_and_data "${MODEL_PATH}"

write_metadata "${STAGE_DIR}/metadata.json" "squeezenet" "onnx" "${MODEL_BASENAME}"

cd "${MODEL_DIR}"
package_kapsl "${MODEL_DIR}" "${PACKAGE_NAME}"
