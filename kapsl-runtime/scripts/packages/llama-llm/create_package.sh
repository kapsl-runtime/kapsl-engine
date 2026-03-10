#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${ROOT_DIR}/scripts/packages/common/common.sh"

MODEL_ID="llama-3.2-3-it"
MODEL_DIR="${ROOT_DIR}/models/${MODEL_ID}"
PACKAGE_NAME="${MODEL_ID}.aimod"

HF_REPO="onnx-community/Llama-3.2-3B-Instruct-ONNX"
ONNX_PREFIX="onnx/"

: "${PREFERRED_PROVIDER:=cpu}"
: "${FALLBACK_PROVIDERS:=}"
: "${REQUIRED_PRECISION:=int8}"
: "${GRAPH_OPT_LEVEL:=all}"
: "${DEVICE_ID:=0}"
: "${STRATEGY:=pool}"
: "${LLM_SAFE_LOAD:=true}"
: "${LLM_ISOLATE_PROCESS:=true}"

PREFERRED_MODELS=(
    "model_uint8.onnx"
    "model_fp16.onnx"
    "model_q4f16.onnx"
    "model_q4.onnx"
    "model_bnb4.onnx"
    "model.onnx"
)

SEARCH_DIRS=(
    "${MODEL_DIR}/onnx"
    "${MODEL_DIR}/onnx-export"
    "${MODEL_DIR}"
)

mkdir -p "${MODEL_DIR}"

MODEL_PATH="$(select_model_file || true)"

if [ -z "${MODEL_PATH}" ]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to download model files."
    command -v python3 >/dev/null 2>&1 || die "python3 is required to list files from Hugging Face."

    ensure_parent_dir() {
        local target="$1"
        local dir
        dir="$(dirname "${target}")"
        if [ "${dir}" != "." ]; then
            mkdir -p "${dir}"
        fi
    }

    TMP_MANIFEST="$(mktemp)"
    TMP_MODEL_FILE="$(mktemp)"

    HF_REPO="${HF_REPO}" ONNX_PREFIX="${ONNX_PREFIX}" MODEL_FILE_OUT="${TMP_MODEL_FILE}" MODEL_FILE="${MODEL_FILE:-}" \
    python3 - <<'PY' > "${TMP_MANIFEST}"
import json, os, sys
from urllib.request import urlopen

repo = os.environ["HF_REPO"]
prefix = os.environ.get("ONNX_PREFIX", "onnx/")
model_file_out = os.environ.get("MODEL_FILE_OUT", "")
model_file = os.environ.get("MODEL_FILE", "").strip()

api_url = f"https://huggingface.co/api/models/{repo}"
with urlopen(api_url) as resp:
    data = json.load(resp)

siblings = data.get("siblings", [])
files = [item.get("rfilename") for item in siblings if item.get("rfilename")]

onnx_files = [f for f in files if f.startswith(prefix)]
onnx_models = [f for f in onnx_files if f.endswith(".onnx")]
preferred = [
    "model_uint8.onnx",
    "model_fp16.onnx",
    "model_q4f16.onnx",
    "model_q4.onnx",
    "model_bnb4.onnx",
    "model.onnx",
]

if model_file:
    if not model_file.startswith(prefix):
        model_file = f"{prefix}{model_file}"
else:
    for name in preferred:
        candidate = f"{prefix}{name}"
        if candidate in onnx_models:
            model_file = candidate
            break
    if not model_file and onnx_models:
        model_file = onnx_models[0]

if not model_file or model_file not in files:
    print("No ONNX model file selected.", file=sys.stderr)
    sys.exit(1)

if model_file_out:
    with open(model_file_out, "w") as handle:
        handle.write(model_file)

required = [f for f in files if f == model_file or f.startswith(f"{model_file}_data")]
required.sort()

extra = []
for cand in (
    "tokenizer.json",
    "tokenizer.model",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "generation_config.json",
):
    if cand in files:
        extra.append(cand)

for f in required:
    print(f)
for f in extra:
    print(f)
PY

    if [ ! -s "${TMP_MANIFEST}" ]; then
        die "No files found (empty manifest). Check repo or ONNX_PREFIX='${ONNX_PREFIX}'."
    fi

    MODEL_FILE_FROM_REMOTE="$(cat "${TMP_MODEL_FILE}")"
    rm -f "${TMP_MODEL_FILE}"

    [ -n "${MODEL_FILE_FROM_REMOTE}" ] || die "No ONNX model file selected."

    echo "Fetching files from ${HF_REPO} (model: ${MODEL_FILE_FROM_REMOTE})..."
    while IFS= read -r path; do
        [ -z "${path}" ] && continue
        url="https://huggingface.co/${HF_REPO}/resolve/main/${path}"
        local_path="${path}"
        if [ ! -f "${local_path}" ]; then
            echo "Downloading ${path}..."
            ensure_parent_dir "${local_path}"
            curl -fL -o "${local_path}" "${url}"
        fi
    done < "${TMP_MANIFEST}"

    rm -f "${TMP_MANIFEST}"

    MODEL_PATH="${MODEL_DIR}/${MODEL_FILE_FROM_REMOTE}"
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

write_metadata "${STAGE_DIR}/metadata.json" "${MODEL_ID}" "llm" "${MODEL_BASENAME}"

cd "${MODEL_DIR}"
package_kapsl "${MODEL_DIR}" "${PACKAGE_NAME}"
