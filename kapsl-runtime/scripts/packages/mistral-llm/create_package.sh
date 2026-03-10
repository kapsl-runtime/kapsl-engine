#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/packages/common/common.sh"

MODEL_ID="mistral-llm"
MODEL_DIR="${ROOT_DIR}/models/mistral"
ONNX_DIR="${MODEL_DIR}/onnx"
PACKAGE_NAME="mistral.aimod"

: "${MISTRAL_VARIANT:=q4f16}"  # q4f16 | q4 | fp16 | quantized | base
: "${PREFERRED_PROVIDER:=cpu}"
: "${FALLBACK_PROVIDERS:=}"
: "${REQUIRED_PRECISION:=fp32}"
: "${GRAPH_OPT_LEVEL:=all}"
: "${DEVICE_ID:=0}"
: "${STRATEGY:=pool}"
: "${LLM_SAFE_LOAD:=true}"
: "${LLM_ISOLATE_PROCESS:=true}"

case "${MISTRAL_VARIANT}" in
    q4f16)
        DECODER_MODEL="decoder_model_merged_q4f16.onnx"
        EMBED_MODEL="embed_tokens_q4f16.onnx"
        VISION_MODEL="vision_encoder_q4f16.onnx"
        ;;
    q4)
        DECODER_MODEL="decoder_model_merged_q4.onnx"
        EMBED_MODEL="embed_tokens_q4.onnx"
        VISION_MODEL="vision_encoder_q4.onnx"
        ;;
    fp16)
        DECODER_MODEL="decoder_model_merged_fp16.onnx"
        EMBED_MODEL="embed_tokens_fp16.onnx"
        VISION_MODEL="vision_encoder_fp16.onnx"
        ;;
    quantized)
        DECODER_MODEL="decoder_model_merged_quantized.onnx"
        EMBED_MODEL="embed_tokens_quantized.onnx"
        VISION_MODEL="vision_encoder_quantized.onnx"
        ;;
    base)
        DECODER_MODEL="decoder_model_merged.onnx"
        EMBED_MODEL="embed_tokens.onnx"
        VISION_MODEL="vision_encoder.onnx"
        ;;
    *)
        die "Unknown MISTRAL_VARIANT='${MISTRAL_VARIANT}'. Expected: q4f16|q4|fp16|quantized|base"
        ;;
esac

copy_external_data_variants() {
    local model_path="$1"
    local candidate
    for candidate in \
        "${model_path}_data" \
        "${model_path}.data" \
        "${model_path}_data_"* \
        "${model_path}.data_"*
    do
        [ -f "${candidate}" ] || continue
        cp "${candidate}" "${STAGE_DIR}/$(basename "${candidate}")"
    done
}

stage_model_with_data() {
    local model_path="$1"
    [ -f "${model_path}" ] || die "Required model file not found: ${model_path}"
    cp "${model_path}" "${STAGE_DIR}/$(basename "${model_path}")"
    copy_external_data_variants "${model_path}"
}

[ -d "${MODEL_DIR}" ] || die "Model directory '${MODEL_DIR}' not found."
[ -d "${ONNX_DIR}" ] || die "ONNX directory '${ONNX_DIR}' not found."

echo "Packaging Mistral from ${MODEL_DIR} (variant=${MISTRAL_VARIANT})..."

stage_init

MAIN_MODEL_PATH="${ONNX_DIR}/${DECODER_MODEL}"
MODEL_BASENAME="$(basename "${MAIN_MODEL_PATH}")"

# Stage decoder + embed + vision components for the selected variant.
stage_model_with_data "${MAIN_MODEL_PATH}"
stage_model_with_data "${ONNX_DIR}/${EMBED_MODEL}"
stage_model_with_data "${ONNX_DIR}/${VISION_MODEL}"

# Stage tokenizer/config files used by Mistral/Pixtral-style processors.
EXTRA_FILES=(
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "config.json"
    "generation_config.json"
    "processor_config.json"
    "preprocessor_config.json"
    "chat_template.jinja"
    "README.md"
)

for file in "${EXTRA_FILES[@]}"; do
    if [ -f "${MODEL_DIR}/${file}" ]; then
        cp "${MODEL_DIR}/${file}" "${STAGE_DIR}/${file}"
    fi
done

require_staged_files \
    "${MODEL_BASENAME}" \
    "${EMBED_MODEL}" \
    "${VISION_MODEL}" \
    "tokenizer.json" \
    "config.json"

write_metadata "${STAGE_DIR}/metadata.json" "${MODEL_ID}" "llm" "${MODEL_BASENAME}"

mkdir -p "${MODEL_DIR}"
package_kapsl "${MODEL_DIR}" "${PACKAGE_NAME}"
