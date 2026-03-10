#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/packages/common/common.sh"

MODEL_ID="gemma-3-4b-it"
MODEL_DIR="${ROOT_DIR}/models/${MODEL_ID}"
PACKAGE_NAME="${MODEL_ID}.aimod"

: "${PREFERRED_PROVIDER:=coreml}"
: "${FALLBACK_PROVIDERS:=cpu}"
: "${REQUIRED_PRECISION:=fp32}"
: "${GRAPH_OPT_LEVEL:=all}"
: "${DEVICE_ID:=0}"
: "${STRATEGY:=pool}"

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
    "${MODEL_DIR}/onnx"
    "${MODEL_DIR}/onnx-export"
    "${MODEL_DIR}"
)

[ -d "${MODEL_DIR}" ] || die "Model directory '${MODEL_DIR}' not found."

echo "Packaging Gemma files from ${MODEL_DIR}..."

stage_init
MODEL_PATH="$(select_model_file || true)"
[ -n "${MODEL_PATH}" ] || die "No .onnx file found in ${MODEL_DIR}."

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

write_metadata "${STAGE_DIR}/metadata.json" "${MODEL_ID}" "llm" "${MODEL_BASENAME}"

cd "${MODEL_DIR}"
package_kapsl "${MODEL_DIR}" "${PACKAGE_NAME}"
