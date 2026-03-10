#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/packages/common/common.sh"

MODEL_DIR="${ROOT_DIR}/models/gpt-llm"
PACKAGE_NAME="gpt-oss-20b.aimod"

BASE_REPO="onnxruntime/gpt-oss-20b-onnx"
BASE_PATH="cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"

MODEL_URL="https://huggingface.co/${BASE_REPO}/resolve/main/${BASE_PATH}/model.onnx"
DATA_URL="https://huggingface.co/${BASE_REPO}/resolve/main/${BASE_PATH}/model.onnx.data"
TOKENIZER_URL="https://huggingface.co/${BASE_REPO}/resolve/main/tokenizer.json"

: "${PREFERRED_PROVIDER:=cpu}"
: "${FALLBACK_PROVIDERS:=}"
: "${REQUIRED_PRECISION:=fp32}"
: "${GRAPH_OPT_LEVEL:=all}"
: "${DEVICE_ID:=0}"
: "${STRATEGY:=pool}"

PREFERRED_MODELS=(
    "model.onnx"
)

SEARCH_DIRS=(
    "${MODEL_DIR}"
)

mkdir -p "${MODEL_DIR}"

MODEL_PATH="$(select_model_file || true)"
if [ -z "${MODEL_PATH}" ]; then
    echo "Downloading GPT-OSS 20B ONNX model..."
    curl -L -o "${MODEL_DIR}/model.onnx" "${MODEL_URL}"
    MODEL_PATH="${MODEL_DIR}/model.onnx"
fi

if [ ! -f "${MODEL_DIR}/model.onnx.data" ]; then
    if curl -fL -o "${MODEL_DIR}/model.onnx.data" "${DATA_URL}"; then
        echo "Downloaded model data file."
    else
        rm -f "${MODEL_DIR}/model.onnx.data"
    fi
fi

if [ ! -f "${MODEL_DIR}/tokenizer.json" ]; then
    echo "Downloading tokenizer..."
    curl -L -o "${MODEL_DIR}/tokenizer.json" "${TOKENIZER_URL}"
fi

stage_init
stage_model_and_data "${MODEL_PATH}"

EXTRA_FILES=(
    "tokenizer.json"
    "tokenizer.model"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "config.json"
    "generation_config.json"
    "README.md"
)

stage_extra_files "${EXTRA_FILES[@]}"
require_staged_files "tokenizer.json"

write_metadata "${STAGE_DIR}/metadata.json" "gpt-oss-20b" "llm" "${MODEL_BASENAME}"

cd "${MODEL_DIR}"
package_kapsl "${MODEL_DIR}" "${PACKAGE_NAME}"
