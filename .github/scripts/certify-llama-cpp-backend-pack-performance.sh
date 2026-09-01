#!/usr/bin/env bash
# Real-GPU eager-vs-pack performance certification. The public signed pack is
# deliberately downloaded by the candidate run; startup/download time is not
# included in inference measurements.
set -Eeuo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
benchmark="$script_dir/benchmark-llama-cpp-backend-pack.py"
runtime_pid=""

die() {
  echo "FAIL: $*" >&2
  exit 1
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup_runtime() {
  if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
    kill "$runtime_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$runtime_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$runtime_pid" 2>/dev/null; then
      kill -KILL "$runtime_pid" 2>/dev/null || true
    fi
    wait "$runtime_pid" 2>/dev/null || true
  fi
  runtime_pid=""
}

cleanup() {
  cleanup_runtime
}
trap cleanup EXIT INT TERM

wait_for_model() {
  local base_url="$1"
  local log_file="$2"
  local deadline=$((SECONDS + ${KAPSL_GPU_CERT_LOAD_TIMEOUT_SECONDS:-300}))
  while (( SECONDS < deadline )); do
    if [[ -n "$runtime_pid" ]] && ! kill -0 "$runtime_pid" 2>/dev/null; then
      tail -n 200 "$log_file" >&2 || true
      die "runtime exited before its model became ready"
    fi
    if curl -fsS "$base_url/v1/models" \
      | jq -e '((.data // .) | type == "array" and length > 0)' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  tail -n 200 "$log_file" >&2 || true
  die "runtime model did not become ready before the timeout"
}

assert_single_in_process_cuda_participant() {
  local pid="$1"
  local log_file="$2"
  local pids_file="$3"
  local baseline_pids_file="$4"
  local runtime_binary="$5"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits >"$pids_file" 2>/dev/null \
    || die "nvidia-smi could not report compute participants"
  local runtime_entries
  runtime_entries="$(awk -v pid="$pid" '$1 == pid { count++ } END { print count + 0 }' "$pids_file")"
  if [[ "$runtime_entries" == "0" ]]; then
    # nvidia-smi reports host-namespace PIDs when certification runs inside a
    # container with a private PID namespace (for example, Vast). Identify the
    # one participant created after launch and verify that NVML attributes it
    # to the runtime executable. Exactly one new participant also rules out a
    # CUDA-owning backend child in that environment.
    local new_participants
    new_participants="$(awk -v before_file="$baseline_pids_file" '
      BEGIN {
        while ((getline line < before_file) > 0) {
          split(line, fields)
          if (fields[1] ~ /^[0-9]+$/) before[fields[1]] = 1
        }
        close(before_file)
      }
      $1 ~ /^[0-9]+$/ && !before[$1] { print $1 }
    ' "$pids_file")"
    local new_participant_count
    new_participant_count="$(awk 'NF { count++ } END { print count + 0 }' <<<"$new_participants")"
    [[ "$new_participant_count" == "1" ]] \
      || die "runtime PID is namespaced and expected one new GPU compute participant, observed $new_participant_count"

    local host_pid
    host_pid="$(awk 'NF { print; exit }' <<<"$new_participants")"
    local participant_name
    participant_name="$(nvidia-smi \
      --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v pid="$host_pid" '
          { observed = $1; gsub(/[[:space:]]/, "", observed) }
          observed == pid {
            name = $2
            sub(/^[[:space:]]+/, "", name)
            sub(/[[:space:]]+$/, "", name)
            print name
            exit
          }
        ' )"
    local runtime_name="${runtime_binary##*/}"
    if [[ -z "$participant_name" || "$participant_name" == "[Not Found]" ]]; then
      # Some container runtimes hide host /proc entries from NVML, so the
      # process name cannot be resolved either. The core's successful pool
      # allocation proves that it owns a CUDA context; one new NVML entry and
      # no child process therefore prove that no backend child owns another.
      grep -Fq 'GPU device pool allocated' "$log_file" \
        || die "cannot attribute namespaced GPU participant $host_pid to the runtime-owned pool"
      local runtime_children
      runtime_children="$(pgrep -P "$pid" || true)"
      [[ -z "$runtime_children" ]] \
        || die "runtime PID is namespaced and has child processes during native-pack certification: $runtime_children"
    else
      [[ "$participant_name" == *"$runtime_name"* ]] \
        || die "new GPU participant $host_pid belongs to '$participant_name', expected runtime '$runtime_name'"
    fi
    echo "INFO: mapped container runtime PID $pid to host GPU participant PID $host_pid"
  elif [[ "$runtime_entries" != "1" ]]; then
    die "expected one GPU compute participant entry for runtime pid $pid, observed $runtime_entries"
  fi

  while IFS= read -r child; do
    [[ "$child" =~ ^[0-9]+$ ]] || continue
    if awk -v pid="$child" '$1 == pid { found=1 } END { exit !found }' "$pids_file"; then
      ps -fp "$child" >&2 || true
      die "backend child pid $child owns a CUDA compute participant; llama.cpp packs must remain in-process"
    fi
  done < <(pgrep -P "$pid" || true)

  ! grep -Fq 'Process isolation enabled' "$log_file" \
    || die "runtime enabled process isolation during native-pack certification"
}

run_case() {
  local label="$1"
  local binary="$2"
  local port="$3"
  local lazy="$4"
  local state_dir="$output_dir/state-$label"
  local cache_dir="$output_dir/cache-$label"
  local socket_path="$output_dir/$label.sock"
  local log_file="$output_dir/$label-runtime.log"
  local result_file="$output_dir/$label-results.json"
  local baseline_pids_file="$output_dir/$label-compute-pids-before.txt"
  mkdir -p "$state_dir" "$cache_dir"

  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    >"$baseline_pids_file" 2>/dev/null \
    || die "nvidia-smi could not report compute participants before $label launch"

  env \
    "CUDA_VISIBLE_DEVICES=$cuda_visible_devices" \
    "KAPSL_GPU_DEVICE_POOL_MODE=auto" \
    "KAPSL_GPU_DEVICE_POOL_MODE_0=auto" \
    "KAPSL_GPU_DEVICE_POOL_DISABLED=0" \
    "KAPSL_LAZY_LLAMA_CPP_PACKS=$lazy" \
    "KAPSL_LLAMA_CPP_ALLOW_NATIVE_KV=0" \
    "KAPSL_BACKEND_INDEX_URL=$backend_index_url" \
    "KAPSL_BACKEND_PUBLIC_KEYS=$backend_public_keys" \
    "KAPSL_BACKEND_CACHE_DIR=$cache_dir" \
    "KAPSL_API_TOKEN_READER=" \
    "KAPSL_API_TOKEN_WRITER=" \
    "KAPSL_API_TOKEN_ADMIN=" \
    "KAPSL_ALLOW_INSECURE_HTTP=1" \
    "RUST_LOG=${RUST_LOG:-info}" \
    "$binary" run \
      --model "$gguf_model" \
      --transport socket \
      --socket "$socket_path" \
      --metrics-port "$port" \
      --http-bind 127.0.0.1 \
      --state-dir "$state_dir" \
      >"$log_file" 2>&1 &
  runtime_pid="$!"

  local base_url="http://127.0.0.1:$port"
  wait_for_model "$base_url" "$log_file"
  assert_single_in_process_cuda_participant \
    "$runtime_pid" "$log_file" "$output_dir/$label-compute-pids.txt" \
    "$baseline_pids_file" "$binary"

  "$benchmark" run \
    --base-url "$base_url" \
    --label "$label" \
    --requests "$requests" \
    --warmup "$warmup" \
    --max-tokens "$max_tokens" \
    --timeout-seconds "$request_timeout" \
    --output "$result_file"

  if [[ "$lazy" == "1" ]]; then
    grep -Fq 'Activated signed llama.cpp backend pack llama-cpp/cuda12' "$log_file" \
      || die "candidate did not activate the signed CUDA llama.cpp pack"
    grep -Fq 'llama.cpp pack attached runtime-owned shared KV pool' "$log_file" \
      || die "candidate pack did not attach Kapsl core's shared pool"
    grep -Fq 'llama_kv_cache_kapsl: using external Kapsl KV pool' "$log_file" \
      || die "candidate SDK did not activate the C-ABI external KV provider"
    [[ "$(grep -F -c 'Downloading Kapsl backend llama-cpp/cuda12' "$log_file" || true)" == "1" ]] \
      || die "candidate did not perform exactly one first-use pack download"
  fi
  ! grep -Fq 'kv_path=native' "$log_file" \
    || die "$label runtime fell back to native KV"
  grep -Fq 'kv_path=shared-kv' "$log_file" \
    || die "$label runtime did not activate shared KV"

  cleanup_runtime
  sleep 2
}

if ! is_true "${KAPSL_GPU_CERTIFICATION:-0}"; then
  echo "SKIP: set KAPSL_GPU_CERTIFICATION=1 to run real-GPU pack certification"
  exit 0
fi

for command_name in curl jq nvidia-smi pgrep ps python3; do
  command -v "$command_name" >/dev/null 2>&1 \
    || die "required command not found: $command_name"
done
[[ -x "$benchmark" ]] || die "benchmark helper is missing or not executable: $benchmark"

eager_binary="${KAPSL_GPU_CERT_EAGER_BINARY:?KAPSL_GPU_CERT_EAGER_BINARY is required}"
lazy_binary="${KAPSL_GPU_CERT_LAZY_BINARY:?KAPSL_GPU_CERT_LAZY_BINARY is required}"
gguf_model="${KAPSL_GPU_CERT_GGUF_MODEL:?KAPSL_GPU_CERT_GGUF_MODEL is required}"
backend_index_url="${KAPSL_GPU_CERT_BACKEND_INDEX_URL:?KAPSL_GPU_CERT_BACKEND_INDEX_URL is required}"
backend_public_keys="${KAPSL_GPU_CERT_BACKEND_PUBLIC_KEYS:?KAPSL_GPU_CERT_BACKEND_PUBLIC_KEYS is required}"
output_dir="${KAPSL_GPU_CERT_OUTPUT_DIR:?KAPSL_GPU_CERT_OUTPUT_DIR is required}"
cuda_visible_devices="${KAPSL_GPU_CERT_CUDA_VISIBLE_DEVICES:-0}"
requests="${KAPSL_GPU_CERT_REQUESTS:-20}"
warmup="${KAPSL_GPU_CERT_WARMUP:-3}"
max_tokens="${KAPSL_GPU_CERT_MAX_TOKENS:-64}"
request_timeout="${KAPSL_GPU_CERT_REQUEST_TIMEOUT_SECONDS:-180}"
eager_port="${KAPSL_GPU_CERT_EAGER_PORT:-19105}"
lazy_port="${KAPSL_GPU_CERT_LAZY_PORT:-19106}"

[[ -x "$eager_binary" ]] || die "eager reference binary is not executable: $eager_binary"
[[ -x "$lazy_binary" ]] || die "lazy candidate binary is not executable: $lazy_binary"
[[ -s "$gguf_model" ]] || die "GGUF model is missing or empty: $gguf_model"
[[ "$backend_index_url" == https://* ]] || die "backend index URL must use HTTPS"
[[ "$cuda_visible_devices" != *,* ]] || die "certification requires exactly one visible GPU"
for value in "$requests" "$max_tokens" "$eager_port" "$lazy_port"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "expected a positive integer, got: $value"
done
[[ "$warmup" =~ ^[0-9]+$ ]] || die "warmup must be a non-negative integer"

mkdir -p "$output_dir"
nvidia-smi -L >"$output_dir/nvidia-smi.txt"
"$eager_binary" --version >"$output_dir/eager-version.txt" 2>&1 || true
"$lazy_binary" --version >"$output_dir/lazy-version.txt" 2>&1 || true

run_case eager "$eager_binary" "$eager_port" 0
run_case lazy "$lazy_binary" "$lazy_port" 1

"$benchmark" compare \
  --reference "$output_dir/eager-results.json" \
  --candidate "$output_dir/lazy-results.json" \
  --max-throughput-regression-percent 2 \
  --max-latency-regression-percent 5 \
  --output "$output_dir/comparison.json"

echo "PASS: lazy llama.cpp pack stayed in-process and met the 2% throughput / 5% latency budgets"
