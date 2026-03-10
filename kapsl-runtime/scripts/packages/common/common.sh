#!/bin/bash

json_trim() {
    local s="$1"
    s="${s#${s%%[![:space:]]*}}"
    s="${s%${s##*[![:space:]]}}"
    printf '%s' "$s"
}

json_array() {
    local input="$1"
    local out="["
    local first=1
    local part trimmed
    if [ -z "${input}" ]; then
        echo "[]"
        return 0
    fi
    IFS=',' read -r -a parts <<< "${input}"
    for part in "${parts[@]}"; do
        trimmed="$(json_trim "${part}")"
        if [ -z "${trimmed}" ]; then
            continue
        fi
        if [ ${first} -eq 0 ]; then
            out+=" ,"
        fi
        out+="\"${trimmed}\""
        first=0
    done
    out+="]"
    echo "${out}"
}

json_bool() {
    local value="${1,,}"
    case "${value}" in
        1|true|yes|on) echo "true"; return 0;;
        0|false|no|off) echo "false"; return 0;;
        *) return 1;;
    esac
}

die() {
    echo "Error: $*" 1>&2
    exit 1
}

select_model_file() {
    local dir candidate

    if [ -n "${MODEL_FILE:-}" ]; then
        for dir in "${SEARCH_DIRS[@]}"; do
            if [ -f "${dir}/${MODEL_FILE}" ]; then
                echo "${dir}/${MODEL_FILE}"
                return 0
            fi
        done
    fi

    for dir in "${SEARCH_DIRS[@]}"; do
        for candidate in "${PREFERRED_MODELS[@]}"; do
            if [ -f "${dir}/${candidate}" ]; then
                echo "${dir}/${candidate}"
                return 0
            fi
        done
    done

    for dir in "${SEARCH_DIRS[@]}"; do
        for candidate in "${dir}"/*.onnx; do
            if [ -f "${candidate}" ]; then
                echo "${candidate}"
                return 0
            fi
        done
    done

    return 1
}

find_external_data() {
    local model_path="$1"
    if [ -f "${model_path}_data" ]; then
        echo "${model_path}_data"
        return 0
    fi
    if [ -f "${model_path}.data" ]; then
        echo "${model_path}.data"
        return 0
    fi
    return 1
}

stage_init() {
    STAGE_DIR="$(mktemp -d)"
    cleanup_stage() {
        rm -rf "${STAGE_DIR}"
    }
    trap cleanup_stage EXIT
}

stage_copy() {
    local src="$1"
    local dst="$2"
    cp "${src}" "${STAGE_DIR}/${dst}"
}

stage_model_and_data() {
    local model_path="$1"
    local data_path
    MODEL_BASENAME="$(basename "${model_path}")"
    stage_copy "${model_path}" "${MODEL_BASENAME}"

    data_path="$(find_external_data "${model_path}" || true)"
    DATA_BASENAME=""
    if [ -n "${data_path}" ]; then
        DATA_BASENAME="$(basename "${data_path}")"
        stage_copy "${data_path}" "${DATA_BASENAME}"
    fi

    echo "Detected model file: ${model_path}"
    if [ -n "${data_path}" ]; then
        echo "Found external data file: ${data_path}"
    else
        echo "No external data file found for ${model_path}"
    fi
}

stage_extra_files() {
    local file dir
    for file in "$@"; do
        for dir in "${SEARCH_DIRS[@]}"; do
            if [ -f "${dir}/${file}" ]; then
                stage_copy "${dir}/${file}" "${file}"
                break
            fi
        done
    done
}

require_staged_files() {
    local file
    for file in "$@"; do
        if [ ! -f "${STAGE_DIR}/${file}" ]; then
            die "Missing required file: ${file}"
        fi
    done
}

write_metadata() {
    local path="$1"
    local project="$2"
    local framework="$3"
    local model_file="$4"

    local preferred_provider="${PREFERRED_PROVIDER:-cpu}"
    local fallback_providers="${FALLBACK_PROVIDERS:-}"
    local required_precision="${REQUIRED_PRECISION:-fp32}"
    local graph_opt_level="${GRAPH_OPT_LEVEL:-all}"
    local device_id="${DEVICE_ID:-0}"
    local strategy="${STRATEGY:-pool}"

    local fallback_json
    fallback_json="$(json_array "${fallback_providers}")"

    local llm_fields=()
    local kv_fields=()
    local bool_value

    if [ -n "${LLM_DISABLE_KV_CACHE:-}" ]; then
        if bool_value="$(json_bool "${LLM_DISABLE_KV_CACHE}" 2>/dev/null)"; then
            llm_fields+=("\"disable_kv_cache\": ${bool_value}")
        fi
    fi

    if [ -n "${LLM_SAFE_LOAD:-}" ]; then
        if [ "${LLM_SAFE_LOAD,,}" = "auto" ]; then
            llm_fields+=("\"safe_load\": \"auto\"")
        else
            if bool_value="$(json_bool "${LLM_SAFE_LOAD}" 2>/dev/null)"; then
                llm_fields+=("\"safe_load\": ${bool_value}")
            fi
        fi
    fi

    if [ -n "${LLM_ISOLATE_PROCESS:-}" ]; then
        if bool_value="$(json_bool "${LLM_ISOLATE_PROCESS}" 2>/dev/null)"; then
            llm_fields+=("\"isolate_process\": ${bool_value}")
        fi
    fi

    if [ -n "${LLM_KV_CACHE_MODE:-}" ]; then
        kv_fields+=("\"mode\": \"${LLM_KV_CACHE_MODE}\"")
    fi
    if [ -n "${LLM_KV_CACHE_BLOCK_SIZE:-}" ] && [[ "${LLM_KV_CACHE_BLOCK_SIZE}" =~ ^[0-9]+$ ]]; then
        kv_fields+=("\"block_size\": ${LLM_KV_CACHE_BLOCK_SIZE}")
    fi
    if [ -n "${LLM_KV_CACHE_TOTAL_BLOCKS:-}" ] && [[ "${LLM_KV_CACHE_TOTAL_BLOCKS}" =~ ^[0-9]+$ ]]; then
        kv_fields+=("\"total_blocks\": ${LLM_KV_CACHE_TOTAL_BLOCKS}")
    fi
    if [ -n "${LLM_KV_CACHE_EVICTION:-}" ]; then
        kv_fields+=("\"eviction\": \"${LLM_KV_CACHE_EVICTION}\"")
    fi

    if [ ${#kv_fields[@]} -gt 0 ]; then
        local joined_kv
        joined_kv="$(IFS=,; echo "${kv_fields[*]}")"
        llm_fields+=("\"kv_cache\": { ${joined_kv} }")
    fi

    local metadata_block=""
    if [ ${#llm_fields[@]} -gt 0 ]; then
        local joined_llm
        joined_llm="$(IFS=,; echo "${llm_fields[*]}")"
        metadata_block=$'\n  ,"metadata": {\n    "llm": { '
        metadata_block+="${joined_llm}"
        metadata_block+=$' }\n  }'
    fi

    cat > "${path}" <<EOF
{
  "project_name": "${project}",
  "framework": "${framework}",
  "version": "1.0",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "model_file": "${model_file}",
  "hardware_requirements": {
    "preferred_provider": "${preferred_provider}",
    "fallback_providers": ${fallback_json},
    "required_precision": "${required_precision}",
    "graph_optimization_level": "${graph_opt_level}",
    "device_id": ${device_id},
    "strategy": "${strategy}"
  }${metadata_block}
}
EOF
}

package_kapsl() {
    local output_dir="$1"
    local package_name="$2"
    local output_path="${output_dir}/${package_name}"
    mkdir -p "${output_dir}"
    tar -czf "${output_path}.tmp" -C "${STAGE_DIR}" .
    mv "${output_path}.tmp" "${output_path}"
    echo "Created ${output_path}"
}
