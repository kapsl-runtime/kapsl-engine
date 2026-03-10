#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/packages/common/common.sh"

MODEL_DIR="${ROOT_DIR}/models/deepseek-llm-no-cache"
PACKAGE_NAME="deepseek_r1.5b.aimod"

ONNX_REPO="onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
MODEL_URL="https://huggingface.co/${ONNX_REPO}/resolve/main/onnx/model_q4f16.onnx"
TOKENIZER_URL="https://huggingface.co/${ONNX_REPO}/resolve/main/tokenizer.json"

: "${PREFERRED_PROVIDER:=directml}"
: "${FALLBACK_PROVIDERS:=}"
: "${REQUIRED_PRECISION:=fp32}"
: "${GRAPH_OPT_LEVEL:=all}"
: "${DEVICE_ID:=0}"
: "${STRATEGY:=pool}"
: "${LLM_DISABLE_KV_CACHE:=true}"

PREFERRED_MODELS=(
    "model_fp16.onnx"
    "model_q4f16.onnx"
    "model_q4.onnx"
    "model_int8.onnx"
    "model_uint8.onnx"
    "model_quantized.onnx"
    "model_bnb4.onnx"
    "model.onnx"
)

SEARCH_DIRS=(
    "${MODEL_DIR}"
)

mkdir -p "${MODEL_DIR}"

MODEL_PATH="$(select_model_file || true)"
if [ -z "${MODEL_PATH}" ]; then
    echo "Downloading DeepSeek R1-Distill (1.5B) ONNX model..."
    curl -L -o "${MODEL_DIR}/model_q4f16.onnx" "${MODEL_URL}"
    MODEL_PATH="${MODEL_DIR}/model_q4f16.onnx"
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

write_metadata "${STAGE_DIR}/metadata.json" "deepseek" "llm" "${MODEL_BASENAME}"

cd "${MODEL_DIR}"
package_kapsl "${MODEL_DIR}" "${PACKAGE_NAME}"
