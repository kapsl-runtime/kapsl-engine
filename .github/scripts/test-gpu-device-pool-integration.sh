#!/usr/bin/env bash
# Opt-in, real-GPU integration test for one runtime-owned CUDA pool shared by
# ONNX Runtime and GGUF shared KV.
#
# Required when KAPSL_GPU_INTEGRATION=1:
#   KAPSL_GPU_TEST_BINARY      CUDA + gguf-cuda-shared-kv kapsl binary
#   KAPSL_GPU_TEST_ORT_MODEL   ONNX or AIMOD model path
#   KAPSL_GPU_TEST_GGUF_MODEL  uniform-KV causal GGUF or AIMOD model path
#
# Useful optional inputs:
#   KAPSL_GPU_TEST_ORT_REQUEST       JSON inference request for the ONNX model
#   KAPSL_GPU_TEST_POOL_BYTES        exact fixed backing size (default: strict auto)
#   KAPSL_GPU_TEST_CUDA_VISIBLE_DEVICES  one physical GPU/UUID (default: 0)
#   KAPSL_GPU_TEST_VRAM_GROWTH_BYTES reload tolerance (default: 256 MiB)
#   KAPSL_GPU_TEST_FRAGMENTATION_TOLERANCE allowed ratio increase (default: 0.01)
#   KAPSL_GPU_TEST_REQUIRE_LAZY_LLAMA_PACK require signed lazy-pack markers
#   KAPSL_GPU_TEST_OUTPUT_DIR        retained logs and snapshots
#
# The two model paths are startup arguments so pool registration necessarily
# precedes both backend/session constructions. Positive `owner="onnx"` bytes
# prove that ORT actually suballocated from it even for generative ONNX paths
# that do not emit the generic OnnxBackend log marker. Each model is then
# stopped and started again under the same id while the other remains resident.
set -Eeuo pipefail

script_name="$(basename "$0")"
runtime_pid=""
created_output_dir=0

usage() {
  cat <<EOF
Usage:
  $script_name --self-test
  KAPSL_GPU_INTEGRATION=1 \\
    KAPSL_GPU_TEST_BINARY=/path/to/kapsl \\
    KAPSL_GPU_TEST_ORT_MODEL=/models/model.onnx \\
    KAPSL_GPU_TEST_GGUF_MODEL=/models/model.gguf \\
    $script_name

The GPU run fails unless all of these are observed in one process:
  * exactly one runtime GPU backing allocation and ORT registration
  * positive live owner="onnx" bytes from the registered ORT allocator
  * GGUF reporting kv_path=shared-kv, both before and after reload
  * each owner disappears on unload and live/free state recovers on reload
  * stable pool capacity and external-memory accounting across both reloads
  * no process-isolation or native-KV fallback marker
  * when KAPSL_GPU_TEST_REQUIRE_LAZY_LLAMA_PACK=1, exactly one lazy pack
    download plus the signed ABI and core-owned shared-pool markers

Without KAPSL_GPU_INTEGRATION=1 the script exits successfully with SKIP.
EOF
}

die() {
  echo "FAIL: $*" >&2
  exit 1
}

note() {
  echo "gpu-pool-integration: $*"
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

count_log_marker() {
  local log_file="$1"
  local marker="$2"
  grep -F -c -- "$marker" "$log_file" 2>/dev/null || true
}

require_log_marker() {
  local log_file="$1"
  local marker="$2"
  local minimum="${3:-1}"
  local count
  count="$(count_log_marker "$log_file" "$marker")"
  if (( count < minimum )); then
    echo "Expected at least $minimum occurrence(s) of: $marker" >&2
    return 1
  fi
}

reject_log_marker() {
  local log_file="$1"
  local marker="$2"
  if grep -F -q -- "$marker" "$log_file"; then
    echo "Unexpected fallback/safeguard marker: $marker" >&2
    return 1
  fi
}

assert_runtime_markers() {
  local log_file="$1"
  local minimum_shared_kv="$2"
  local allocated registered

  allocated="$(count_log_marker "$log_file" "GPU device pool allocated:")"
  registered="$(count_log_marker "$log_file" "Registered runtime GPU device pool with ORT for device")"
  if [[ "$allocated" != "1" ]]; then
    echo "Expected exactly one GPU backing allocation, observed $allocated" >&2
    return 1
  fi
  if [[ "$registered" != "1" ]]; then
    echo "Expected exactly one ORT pool registration, observed $registered" >&2
    return 1
  fi

  require_log_marker "$log_file" "kv_path=shared-kv" "$minimum_shared_kv" || return 1

  reject_log_marker "$log_file" "kv_path=native" || return 1
  reject_log_marker "$log_file" "continuing without a runtime-owned pool" || return 1
  reject_log_marker "$log_file" "no physical GPU pool materialized" || return 1
  reject_log_marker "$log_file" "physical CUDA pooling disabled for this process" || return 1
  reject_log_marker "$log_file" "implicit parent CUDA pool suppressed" || return 1
  reject_log_marker "$log_file" "Process isolation enabled" || return 1

  if is_true "${KAPSL_GPU_TEST_REQUIRE_LAZY_LLAMA_PACK:-0}"; then
    local downloads
    downloads="$(count_log_marker "$log_file" "Downloading Kapsl backend llama-cpp/cuda12")"
    if [[ "$downloads" != "1" ]]; then
      echo "Expected exactly one lazy llama.cpp pack download, observed $downloads" >&2
      return 1
    fi
    require_log_marker "$log_file" \
      "Activated signed llama.cpp backend pack llama-cpp/cuda12" 1 || return 1
    require_log_marker "$log_file" \
      "llama.cpp pack attached runtime-owned shared KV pool" "$minimum_shared_kv" || return 1
    require_log_marker "$log_file" \
      "Kapsl C-ABI runtime-owned KV pool active" "$minimum_shared_kv" || return 1
  fi
}

metric_from_text() {
  local metric="$1"
  local device="$2"
  awk -v metric="$metric" -v device="device=\"$device\"" '
    $1 ~ ("^" metric "\\{") && index($1, device) { print $2; exit }
  '
}

owner_metric_from_text() {
  local metric="$1"
  local device="$2"
  local owner="$3"
  awk -v metric="$metric" -v device="device=\"$device\"" -v owner="owner=\"$owner\"" '
    $1 ~ ("^" metric "\\{") && index($1, device) && index($1, owner) { print $2; exit }
  '
}

metric_from_file() {
  local metric="$1"
  local file="$2"
  metric_from_text "$metric" "$logical_device_id" <"$file"
}

owner_metric_from_file() {
  local metric="$1"
  local owner="$2"
  local file="$3"
  owner_metric_from_text "$metric" "$logical_device_id" "$owner" <"$file"
}

assert_pool_snapshot_consistent() {
  local file="$1"
  local backing_bytes="$2"
  local allocated live free free_ranges largest fragmentation
  allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$file")"
  live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$file")"
  free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$file")"
  free_ranges="$(metric_from_file kapsl_gpu_device_pool_free_ranges "$file")"
  largest="$(metric_from_file kapsl_gpu_device_pool_largest_free_range_bytes "$file")"
  fragmentation="$(metric_from_file kapsl_gpu_device_pool_fragmentation_ratio "$file")"

  [[ "$allocated" =~ ^[0-9]+$ ]] || return 1
  [[ "$live" =~ ^[0-9]+$ ]] || return 1
  [[ "$free" =~ ^[0-9]+$ ]] || return 1
  [[ "$free_ranges" =~ ^[0-9]+$ ]] || return 1
  [[ "$largest" =~ ^[0-9]+$ ]] || return 1
  [[ "$fragmentation" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || return 1
  (( allocated + free == backing_bytes )) || return 1
  (( largest <= free )) || return 1
  if (( free == 0 )); then
    (( free_ranges == 0 && largest == 0 )) || return 1
  else
    (( free_ranges > 0 )) || return 1
  fi
  awk -v ratio="$fragmentation" 'BEGIN { exit !(ratio >= 0.0 && ratio <= 1.0) }'
}

require_active_owner() {
  local file="$1"
  local owner="$2"
  local usage guaranteed max admitted allocatable
  usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes "$owner" "$file")"
  guaranteed="$(owner_metric_from_file kapsl_gpu_device_pool_owner_quota_guaranteed_bytes "$owner" "$file")"
  max="$(owner_metric_from_file kapsl_gpu_device_pool_owner_quota_max_bytes "$owner" "$file")"
  admitted="$(owner_metric_from_file kapsl_gpu_device_pool_owner_admitted "$owner" "$file")"
  allocatable="$(owner_metric_from_file kapsl_gpu_device_pool_owner_allocatable_bytes "$owner" "$file")"
  [[ "$usage" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$guaranteed" =~ ^[0-9]+$ ]] || return 1
  [[ "$max" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$allocatable" =~ ^[0-9]+$ ]] || return 1
  [[ "$admitted" == "1" ]] || return 1
  (( guaranteed <= max && usage <= max && allocatable <= max - usage ))
}

require_admitted_owner() {
  local file="$1"
  local owner="$2"
  local usage guaranteed max admitted allocatable
  usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes "$owner" "$file")"
  guaranteed="$(owner_metric_from_file kapsl_gpu_device_pool_owner_quota_guaranteed_bytes "$owner" "$file")"
  max="$(owner_metric_from_file kapsl_gpu_device_pool_owner_quota_max_bytes "$owner" "$file")"
  admitted="$(owner_metric_from_file kapsl_gpu_device_pool_owner_admitted "$owner" "$file")"
  allocatable="$(owner_metric_from_file kapsl_gpu_device_pool_owner_allocatable_bytes "$owner" "$file")"
  [[ "$usage" =~ ^[0-9]+$ ]] || return 1
  [[ "$guaranteed" =~ ^[0-9]+$ ]] || return 1
  [[ "$max" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$allocatable" =~ ^[0-9]+$ ]] || return 1
  [[ "$admitted" == "1" ]] || return 1
  (( guaranteed <= max && usage <= max && allocatable <= max - usage ))
}

require_owner_absent() {
  local file="$1"
  local owner="$2"
  ! grep -F -q -- "owner=\"$owner\"" "$file"
}

assert_fragmentation_not_worse() {
  local baseline="$1"
  local current="$2"
  local tolerance="$3"
  awk -v baseline="$baseline" -v current="$current" -v tolerance="$tolerance" \
    'BEGIN { exit !(current <= baseline + tolerance) }'
}

stopped_pool_snapshot_matches() {
  local candidate="$1"
  local initial_external="$2"
  local initial_allocated="$3"
  local initial_live="$4"
  local initial_free="$5"
  local backing_bytes="$6"
  local external allocated live free
  external="$(metric_from_file kapsl_device_memory_external_bytes "$candidate")"
  allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$candidate")"
  live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$candidate")"
  free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$candidate")"
  [[ "$external" =~ ^[0-9]+$ ]] \
    && [[ "$allocated" =~ ^[0-9]+$ ]] \
    && [[ "$live" =~ ^[0-9]+$ ]] \
    && [[ "$free" =~ ^[0-9]+$ ]] \
    && (( external < initial_external )) \
    && (( allocated <= initial_allocated )) \
    && (( live <= initial_live )) \
    && (( free >= initial_free )) \
    && assert_pool_snapshot_consistent "$candidate" "$backing_bytes" \
    && require_active_owner "$candidate" onnx \
    && require_owner_absent "$candidate" "gguf_kv:$gguf_model_id"
}

reloaded_pool_snapshot_matches() {
  local candidate="$1"
  local initial_external="$2"
  local initial_allocated="$3"
  local initial_live="$4"
  local initial_free="$5"
  local backing_bytes="$6"
  local external allocated live free pooled
  external="$(metric_from_file kapsl_device_memory_external_bytes "$candidate")"
  pooled="$(metric_from_file kapsl_device_memory_pooled_bytes "$candidate")"
  allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$candidate")"
  live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$candidate")"
  free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$candidate")"
  [[ "$external" == "$initial_external" ]] \
    && [[ "$pooled" == "$backing_bytes" ]] \
    && [[ "$allocated" == "$initial_allocated" ]] \
    && [[ "$live" == "$initial_live" ]] \
    && [[ "$free" == "$initial_free" ]] \
    && assert_pool_snapshot_consistent "$candidate" "$backing_bytes" \
    && require_active_owner "$candidate" onnx \
    && require_admitted_owner "$candidate" "gguf_kv:$gguf_model_id"
}

ort_reloaded_pool_snapshot_matches() {
  local candidate="$1"
  local initial_external="$2"
  local stopped_external="$3"
  local initial_allocated="$4"
  local initial_live="$5"
  local initial_free="$6"
  local backing_bytes="$7"
  local external allocated live free pooled
  external="$(metric_from_file kapsl_device_memory_external_bytes "$candidate")"
  pooled="$(metric_from_file kapsl_device_memory_pooled_bytes "$candidate")"
  allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$candidate")"
  live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$candidate")"
  free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$candidate")"
  [[ "$external" =~ ^[0-9]+$ ]] \
    && (( external >= stopped_external )) \
    && (( external <= initial_external )) \
    && [[ "$pooled" == "$backing_bytes" ]] \
    && [[ "$allocated" == "$initial_allocated" ]] \
    && [[ "$live" == "$initial_live" ]] \
    && [[ "$free" == "$initial_free" ]] \
    && assert_pool_snapshot_consistent "$candidate" "$backing_bytes" \
    && require_active_owner "$candidate" onnx \
    && require_admitted_owner "$candidate" "gguf_kv:$gguf_model_id"
}

ort_stopped_pool_snapshot_matches() {
  local candidate="$1"
  local initial_external="$2"
  local initial_allocated="$3"
  local initial_live="$4"
  local initial_free="$5"
  local backing_bytes="$6"
  local external allocated live free
  external="$(metric_from_file kapsl_device_memory_external_bytes "$candidate")"
  allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$candidate")"
  live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$candidate")"
  free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$candidate")"
  # ORT's environment allocator draws weights/arenas from the shared pool, but
  # the CUDA execution provider can also own device state outside that
  # allocator. Unloading ORT may therefore reduce external accounting. Accept
  # either behavior here; reload must still restore the exact pool snapshot and
  # must not grow external memory beyond the initial load.
  [[ "$external" =~ ^[0-9]+$ ]] \
    && (( external <= initial_external )) \
    && [[ "$allocated" =~ ^[0-9]+$ ]] \
    && [[ "$live" =~ ^[0-9]+$ ]] \
    && [[ "$free" =~ ^[0-9]+$ ]] \
    && (( allocated < initial_allocated )) \
    && (( live < initial_live )) \
    && (( free > initial_free )) \
    && assert_pool_snapshot_consistent "$candidate" "$backing_bytes" \
    && require_owner_absent "$candidate" onnx \
    && require_admitted_owner "$candidate" "gguf_kv:$gguf_model_id"
}

run_self_test() {
  local fixture
  fixture="$(mktemp -d)"

  printf '%s\n' \
    'GPU device pool allocated: 4096 MiB' \
    'Registered runtime GPU device pool with ORT for device 0: 4096 MiB' \
    'ONNX session (device 0) using shared Kapsl GPU pool allocator' \
    '[gguf] kv_path=shared-kv Kapsl paged external KV pool active on device 0' \
    '[gguf] kv_path=shared-kv Kapsl paged external KV pool active on device 0' \
    >"$fixture/good.log"
  assert_runtime_markers "$fixture/good.log" 2

  cp "$fixture/good.log" "$fixture/fallback.log"
  printf '%s\n' '[gguf] kv_path=native shared-KV disabled' >>"$fixture/fallback.log"
  if assert_runtime_markers "$fixture/fallback.log" 2 >/dev/null 2>&1; then
    die "self-test accepted a native-KV fallback"
  fi
  cp "$fixture/good.log" "$fixture/duplicate-pool.log"
  printf '%s\n' 'GPU device pool allocated: 4096 MiB' >>"$fixture/duplicate-pool.log"
  if assert_runtime_markers "$fixture/duplicate-pool.log" 2 >/dev/null 2>&1; then
    die "self-test accepted a second GPU backing allocation"
  fi

  local value
  cat >"$fixture/metrics.txt" <<'EOF'
# HELP kapsl_device_memory_pooled_bytes test
kapsl_device_memory_pooled_bytes{device="0"} 4294967296
kapsl_device_memory_external_bytes{device="0"} 2000
kapsl_gpu_device_pool_allocated_bytes{device="0"} 3221225472
kapsl_gpu_device_pool_live_allocations{device="0"} 7
kapsl_gpu_device_pool_free_bytes{device="0"} 1073741824
kapsl_gpu_device_pool_free_ranges{device="0"} 2
kapsl_gpu_device_pool_largest_free_range_bytes{device="0"} 805306368
kapsl_gpu_device_pool_fragmentation_ratio{device="0"} 0.25
kapsl_gpu_device_pool_owner_usage_bytes{device="0",owner="onnx"} 2147483648
kapsl_gpu_device_pool_owner_quota_guaranteed_bytes{device="0",owner="onnx"} 1073741824
kapsl_gpu_device_pool_owner_quota_max_bytes{device="0",owner="onnx"} 4294967296
kapsl_gpu_device_pool_owner_admitted{device="0",owner="onnx"} 1
kapsl_gpu_device_pool_owner_allocatable_bytes{device="0",owner="onnx"} 1073741824
kapsl_gpu_device_pool_owner_usage_bytes{device="0",owner="gguf_kv:1"} 1073741824
kapsl_gpu_device_pool_owner_quota_guaranteed_bytes{device="0",owner="gguf_kv:1"} 0
kapsl_gpu_device_pool_owner_quota_max_bytes{device="0",owner="gguf_kv:1"} 4294967296
kapsl_gpu_device_pool_owner_admitted{device="0",owner="gguf_kv:1"} 1
kapsl_gpu_device_pool_owner_allocatable_bytes{device="0",owner="gguf_kv:1"} 1073741824
EOF
  logical_device_id=0
  value="$(metric_from_file kapsl_device_memory_pooled_bytes "$fixture/metrics.txt")"
  [[ "$value" == "4294967296" ]] || die "metric parser returned: $value"
  assert_pool_snapshot_consistent "$fixture/metrics.txt" "$value" \
    || die "self-test rejected a consistent live pool snapshot"
  require_active_owner "$fixture/metrics.txt" onnx \
    || die "self-test did not find active ONNX ownership"
  require_active_owner "$fixture/metrics.txt" gguf_kv:1 \
    || die "self-test did not find active GGUF ownership"
  sed 's/owner="onnx"} 2147483648/owner="onnx"} 0/' \
    "$fixture/metrics.txt" >"$fixture/no-onnx-usage.txt"
  if require_active_owner "$fixture/no-onnx-usage.txt" onnx; then
    die "self-test accepted an ONNX owner without a live allocation"
  fi
  if require_owner_absent "$fixture/metrics.txt" gguf_kv:1; then
    die "self-test treated a live GGUF owner as absent"
  fi

  cat >"$fixture/stopped-metrics.txt" <<'EOF'
kapsl_device_memory_pooled_bytes{device="0"} 4294967296
kapsl_device_memory_external_bytes{device="0"} 0
kapsl_gpu_device_pool_allocated_bytes{device="0"} 2147483648
kapsl_gpu_device_pool_live_allocations{device="0"} 3
kapsl_gpu_device_pool_free_bytes{device="0"} 2147483648
kapsl_gpu_device_pool_free_ranges{device="0"} 1
kapsl_gpu_device_pool_largest_free_range_bytes{device="0"} 2147483648
kapsl_gpu_device_pool_fragmentation_ratio{device="0"} 0
kapsl_gpu_device_pool_owner_usage_bytes{device="0",owner="onnx"} 2147483648
kapsl_gpu_device_pool_owner_quota_guaranteed_bytes{device="0",owner="onnx"} 1073741824
kapsl_gpu_device_pool_owner_quota_max_bytes{device="0",owner="onnx"} 4294967296
kapsl_gpu_device_pool_owner_admitted{device="0",owner="onnx"} 1
kapsl_gpu_device_pool_owner_allocatable_bytes{device="0",owner="onnx"} 2147483648
EOF
  gguf_model_id=1
  stopped_pool_snapshot_matches \
    "$fixture/stopped-metrics.txt" 2000 3221225472 7 1073741824 4294967296 \
    || die "self-test rejected a valid unload snapshot"
  reloaded_pool_snapshot_matches \
    "$fixture/metrics.txt" 2000 3221225472 7 1073741824 4294967296 \
    || die "self-test rejected a valid reuse snapshot"
  if stopped_pool_snapshot_matches \
    "$fixture/metrics.txt" 2000 3221225472 7 1073741824 4294967296; then
    die "self-test accepted a snapshot that retained the GGUF owner"
  fi
  assert_fragmentation_not_worse 0.25 0.255 0.01 \
    || die "self-test rejected fragmentation within tolerance"
  if assert_fragmentation_not_worse 0.25 0.30 0.01; then
    die "self-test accepted excessive fragmentation growth"
  fi

  cat >"$fixture/ort-stopped-metrics.txt" <<'EOF'
kapsl_device_memory_pooled_bytes{device="0"} 4294967296
kapsl_device_memory_external_bytes{device="0"} 2000
kapsl_gpu_device_pool_allocated_bytes{device="0"} 1073741824
kapsl_gpu_device_pool_live_allocations{device="0"} 4
kapsl_gpu_device_pool_free_bytes{device="0"} 3221225472
kapsl_gpu_device_pool_free_ranges{device="0"} 1
kapsl_gpu_device_pool_largest_free_range_bytes{device="0"} 3221225472
kapsl_gpu_device_pool_fragmentation_ratio{device="0"} 0
kapsl_gpu_device_pool_owner_usage_bytes{device="0",owner="gguf_kv:1"} 1073741824
kapsl_gpu_device_pool_owner_quota_guaranteed_bytes{device="0",owner="gguf_kv:1"} 0
kapsl_gpu_device_pool_owner_quota_max_bytes{device="0",owner="gguf_kv:1"} 4294967296
kapsl_gpu_device_pool_owner_admitted{device="0",owner="gguf_kv:1"} 1
kapsl_gpu_device_pool_owner_allocatable_bytes{device="0",owner="gguf_kv:1"} 3221225472
EOF
  ort_stopped_pool_snapshot_matches \
    "$fixture/ort-stopped-metrics.txt" 2000 3221225472 7 1073741824 4294967296 \
    || die "self-test rejected a valid ORT unload snapshot"
  sed 's/kapsl_device_memory_external_bytes{device="0"} 2000/kapsl_device_memory_external_bytes{device="0"} 1500/' \
    "$fixture/ort-stopped-metrics.txt" >"$fixture/ort-stopped-external-released.txt"
  ort_stopped_pool_snapshot_matches \
    "$fixture/ort-stopped-external-released.txt" 2000 3221225472 7 1073741824 4294967296 \
    || die "self-test rejected ORT external CUDA state released on unload"
  sed 's/kapsl_device_memory_external_bytes{device="0"} 2000/kapsl_device_memory_external_bytes{device="0"} 1750/' \
    "$fixture/metrics.txt" >"$fixture/ort-reloaded-external-reused.txt"
  ort_reloaded_pool_snapshot_matches \
    "$fixture/ort-reloaded-external-reused.txt" 2000 1500 3221225472 7 1073741824 4294967296 \
    || die "self-test rejected lower ORT external CUDA state after reload"
  rm -rf "$fixture"
  note "harness contract self-test passed"
}

cleanup() {
  local status="$?"
  if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
    kill -TERM "$runtime_pid" 2>/dev/null || true
    wait "$runtime_pid" 2>/dev/null || true
  fi

  if [[ -n "${output_dir:-}" ]]; then
    if (( status == 0 )) && (( created_output_dir == 1 )) && ! is_true "${KAPSL_GPU_TEST_KEEP_ARTIFACTS:-0}"; then
      rm -rf "$output_dir"
    else
      echo "GPU integration artifacts: $output_dir" >&2
    fi
  fi
  return "$status"
}

api_get() {
  curl -fsS --max-time 15 "$base_url$1"
}

api_post() {
  local path="$1"
  shift
  curl -fsS --max-time 30 -X POST "$base_url$path" "$@"
}

runtime_is_alive() {
  [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null
}

wait_for_api() {
  local deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < deadline )); do
    if api_get /api/models >/dev/null 2>&1; then
      return 0
    fi
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

wait_for_model_status() {
  local model_id="$1"
  local wanted="$2"
  local deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < deadline )); do
    local status
    status="$(api_get /api/models 2>/dev/null \
      | jq -r --argjson id "$model_id" '.[] | select(.id == $id) | .status' \
      | head -n 1 || true)"
    if [[ "$status" == "$wanted" ]]; then
      return 0
    fi
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

wait_for_log_count() {
  local marker="$1"
  local minimum="$2"
  local deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < deadline )); do
    local count
    count="$(count_log_marker "$runtime_log" "$marker")"
    if (( count >= minimum )); then
      return 0
    fi
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

wait_for_stopped_pool_snapshot() {
  local initial_external="$1"
  local initial_allocated="$2"
  local initial_live="$3"
  local initial_free="$4"
  local backing_bytes="$5"
  local destination="$6"
  local deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < deadline )); do
    local candidate="$destination.next"
    if api_get /metrics >"$candidate" 2>/dev/null; then
      if stopped_pool_snapshot_matches \
        "$candidate" "$initial_external" "$initial_allocated" "$initial_live" \
        "$initial_free" "$backing_bytes"; then
        mv "$candidate" "$destination"
        return 0
      fi
    fi
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

wait_for_reloaded_pool_snapshot() {
  local initial_external="$1"
  local initial_allocated="$2"
  local initial_live="$3"
  local initial_free="$4"
  local backing_bytes="$5"
  local destination="$6"
  local deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < deadline )); do
    local candidate="$destination.next"
    if api_get /metrics >"$candidate" 2>/dev/null; then
      if reloaded_pool_snapshot_matches \
        "$candidate" "$initial_external" "$initial_allocated" "$initial_live" \
        "$initial_free" "$backing_bytes"; then
        mv "$candidate" "$destination"
        return 0
      fi
    fi
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

wait_for_ort_reloaded_pool_snapshot() {
  local initial_external="$1"
  local stopped_external="$2"
  local initial_allocated="$3"
  local initial_live="$4"
  local initial_free="$5"
  local backing_bytes="$6"
  local destination="$7"
  local deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < deadline )); do
    local candidate="$destination.next"
    if api_get /metrics >"$candidate" 2>/dev/null; then
      if ort_reloaded_pool_snapshot_matches \
        "$candidate" "$initial_external" "$stopped_external" \
        "$initial_allocated" "$initial_live" "$initial_free" "$backing_bytes"; then
        mv "$candidate" "$destination"
        return 0
      fi
    fi
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

wait_for_ort_stopped_pool_snapshot() {
  local initial_external="$1"
  local initial_allocated="$2"
  local initial_live="$3"
  local initial_free="$4"
  local backing_bytes="$5"
  local destination="$6"
  local deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < deadline )); do
    local candidate="$destination.next"
    if api_get /metrics >"$candidate" 2>/dev/null; then
      if ort_stopped_pool_snapshot_matches \
        "$candidate" "$initial_external" "$initial_allocated" "$initial_live" \
        "$initial_free" "$backing_bytes"; then
        mv "$candidate" "$destination"
        return 0
      fi
    fi
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

process_vram_bytes() {
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v wanted="$runtime_pid" '
        {
          pid=$1; memory=$2
          gsub(/[[:space:]]/, "", pid)
          gsub(/[[:space:]]/, "", memory)
          if (pid == wanted && memory ~ /^[0-9]+$/) {
            total += memory
            found = 1
          }
        }
        END { if (found) printf "%.0f\n", total * 1024 * 1024 }
      '
}

run_gguf_inference() {
  local destination="$1"
  local payload
  payload="$(jq -cn \
    --arg prompt "${KAPSL_GPU_TEST_GGUF_PROMPT:-Read these words carefully: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega red orange yellow green blue indigo violet north south east west spring summer autumn winter. Reply with the word CUDA.}" \
    --argjson tokens "${KAPSL_GPU_TEST_GGUF_TOKENS:-4}" \
    '{input:{shape:[1,1],dtype:"string",data_base64:($prompt|@base64)},metadata:{max_tokens:$tokens,min_tokens:$tokens,temperature:0.0}}')"
  api_post "/api/models/$gguf_model_id/infer" \
    -H 'Content-Type: application/json' \
    --data-binary "$payload" >"$destination"
}

assert_gguf_response_correct() {
  local response_file="$1"
  local decoded
  decoded="$(jq -er '.data_base64 | @base64d' "$response_file")" \
    || die "GGUF response is missing a decodable data_base64 payload: $response_file"
  [[ "$decoded" == CUDA* ]] \
    || die "GGUF decoded text failed correctness assertion in $response_file: $decoded"
}

run_gguf_concurrency_transition_test() {
  local destination_dir="$1"
  mkdir -p "$destination_dir"
  # Let request 0 enter decode alone, then add peers. Shorter peers retire
  # while request 0 survives, exercising singleton -> multi -> singleton-like
  # concurrency while addressing must remain permanently sequence-aware.
  KAPSL_GPU_TEST_GGUF_TOKENS=64 run_gguf_inference "$destination_dir/response-0.json" &
  local first_pid=$!
  sleep 0.05
  local pids=()
  local index
  for index in 1 2 3 4 5 6 7; do
    KAPSL_GPU_TEST_GGUF_TOKENS=8 run_gguf_inference "$destination_dir/response-$index.json" &
    pids+=("$!")
  done
  wait "$first_pid" || die "long-running GGUF transition request failed"
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || die "GGUF transition peer request $((index + 1)) failed"
  done
  for index in 0 1 2 3 4 5 6 7; do
    assert_gguf_response_correct "$destination_dir/response-$index.json"
  done
}

run_onnx_inference_if_configured() {
  local destination="$1"
  local request_file="${KAPSL_GPU_TEST_ORT_REQUEST:-}"
  if [[ -z "$request_file" ]]; then
    note "ONNX request not supplied; session construction/weight allocation is tested, inference is skipped"
    return 0
  fi
  [[ -s "$request_file" ]] || die "ONNX request file is missing or empty: $request_file"
  api_post "/api/models/$onnx_model_id/infer" \
    -H 'Content-Type: application/json' \
    --data-binary "@$request_file" >"$destination"
}

main() {
  if ! is_true "${KAPSL_GPU_INTEGRATION:-0}"; then
    echo "SKIP: set KAPSL_GPU_INTEGRATION=1 and provide the binary plus ORT/GGUF model paths"
    return 0
  fi

  require_command curl
  require_command jq
  require_command nvidia-smi
  require_command awk

  binary="${KAPSL_GPU_TEST_BINARY:-}"
  ort_model="${KAPSL_GPU_TEST_ORT_MODEL:-}"
  gguf_model="${KAPSL_GPU_TEST_GGUF_MODEL:-}"
  [[ -n "$binary" ]] || die "KAPSL_GPU_TEST_BINARY is required"
  [[ -x "$binary" ]] || die "Kapsl binary is not executable: $binary"
  [[ -s "$ort_model" ]] || die "ORT model is missing or empty: $ort_model"
  [[ -s "$gguf_model" ]] || die "GGUF model is missing or empty: $gguf_model"

  if is_true "${KAPSL_GPU_DEVICE_POOL_DISABLED:-0}"; then
    die "KAPSL_GPU_DEVICE_POOL_DISABLED is an isolated-worker safeguard and cannot be enabled for this in-process sharing test"
  fi

  cuda_visible_devices="${KAPSL_GPU_TEST_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
  [[ -n "$cuda_visible_devices" ]] || die "a single CUDA device must be visible"
  [[ "$cuda_visible_devices" != *,* ]] || die "set KAPSL_GPU_TEST_CUDA_VISIBLE_DEVICES to exactly one GPU"

  load_timeout_seconds="${KAPSL_GPU_TEST_LOAD_TIMEOUT_SECONDS:-300}"
  settle_seconds="${KAPSL_GPU_TEST_SETTLE_SECONDS:-3}"
  vram_growth_bytes="${KAPSL_GPU_TEST_VRAM_GROWTH_BYTES:-268435456}"
  fragmentation_tolerance="${KAPSL_GPU_TEST_FRAGMENTATION_TOLERANCE:-0.01}"
  [[ "$load_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || die "load timeout must be a positive integer"
  [[ "$settle_seconds" =~ ^[0-9]+$ ]] || die "settle seconds must be a non-negative integer"
  [[ "$vram_growth_bytes" =~ ^[0-9]+$ ]] || die "VRAM growth tolerance must be a non-negative integer"
  [[ "$fragmentation_tolerance" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || die "fragmentation tolerance must be a non-negative decimal"
  awk -v value="$fragmentation_tolerance" 'BEGIN { exit !(value >= 0.0 && value <= 1.0) }' \
    || die "fragmentation tolerance must be between zero and one"

  if [[ -n "${KAPSL_GPU_TEST_OUTPUT_DIR:-}" ]]; then
    output_dir="$KAPSL_GPU_TEST_OUTPUT_DIR"
    mkdir -p "$output_dir"
  else
    output_dir="$(mktemp -d)"
    created_output_dir=1
  fi
  runtime_log="$output_dir/runtime.log"
  state_dir="$output_dir/state"
  socket_path="$output_dir/kapsl.sock"
  mkdir -p "$state_dir"
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  port="${KAPSL_GPU_TEST_PORT:-19095}"
  [[ "$port" =~ ^[0-9]+$ ]] && (( port > 0 && port < 65536 )) || die "invalid KAPSL_GPU_TEST_PORT: $port"
  base_url="http://127.0.0.1:$port"
  logical_device_id=0
  onnx_model_id=0
  gguf_model_id=1

  nvidia-smi -L >"$output_dir/nvidia-smi.txt"
  "$binary" --version >"$output_dir/kapsl-version.txt" 2>&1 || true

  pool_mode=auto
  pool_bytes="${KAPSL_GPU_TEST_POOL_BYTES:-}"
  if [[ -n "$pool_bytes" ]]; then
    pool_mode=fixed
  fi

  note "starting ORT model 0 and GGUF model 1 on CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
  env_args=(
    "CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
    "KAPSL_GPU_DEVICE_POOL_MODE=$pool_mode"
    "KAPSL_GPU_DEVICE_POOL_MODE_0=$pool_mode"
    "KAPSL_GPU_DEVICE_POOL_DISABLED=0"
    "KAPSL_GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES="
    "KAPSL_GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES_0="
    "KAPSL_GPU_ONNX_GUARANTEED_BYTES="
    "KAPSL_GPU_ONNX_GUARANTEED_BYTES_0="
    "KAPSL_GPU_ONNX_MAX_BYTES="
    "KAPSL_GPU_ONNX_MAX_BYTES_0="
    "KAPSL_GPU_GGUF_GUARANTEED_BYTES="
    "KAPSL_GPU_GGUF_GUARANTEED_BYTES_0="
    "KAPSL_GPU_GGUF_MAX_BYTES="
    "KAPSL_GPU_GGUF_MAX_BYTES_0="
    "KAPSL_GGUF_DISABLE_SHARED_KV=0"
    "KAPSL_PROVIDER_POLICY=fastest"
    "KAPSL_API_TOKEN_READER="
    "KAPSL_API_TOKEN_WRITER="
    "KAPSL_API_TOKEN_ADMIN="
    "KAPSL_DISCARD_PACKAGE_AFTER_LOAD=0"
    "KAPSL_LITE_DISCARD_PACKAGE_AFTER_LOAD=0"
    "KAPSL_ALLOW_INSECURE_HTTP=1"
    "RUST_LOG=${RUST_LOG:-info}"
  )
  if [[ -n "$pool_bytes" ]]; then
    env_args+=(
      "KAPSL_GPU_DEVICE_POOL_BYTES=$pool_bytes"
      "KAPSL_GPU_DEVICE_POOL_BYTES_0=$pool_bytes"
    )
  else
    env_args+=(
      "KAPSL_GPU_DEVICE_POOL_BYTES="
      "KAPSL_GPU_DEVICE_POOL_BYTES_0="
    )
  fi

  env "${env_args[@]}" "$binary" run \
    --model "$ort_model" \
    --model "$gguf_model" \
    --transport socket \
    --socket "$socket_path" \
    --metrics-port "$port" \
    --http-bind 127.0.0.1 \
    --state-dir "$state_dir" \
    >"$runtime_log" 2>&1 &
  runtime_pid="$!"

  if ! wait_for_api; then
    tail -n 200 "$runtime_log" >&2 || true
    die "runtime did not become ready"
  fi
  wait_for_model_status "$onnx_model_id" active || die "ORT model did not become active"
  wait_for_model_status "$gguf_model_id" active || die "GGUF model did not become active"
  api_get /api/models >"$output_dir/models-initial.json"

  if [[ "${ort_model,,}" == *.onnx ]]; then
    jq -e --argjson id "$onnx_model_id" \
      '.[] | select(.id == $id and ((.format // .framework) == "onnx"))' \
      "$output_dir/models-initial.json" >/dev/null \
      || die "raw model 0 is not reported as ONNX"
  fi
  if [[ "${gguf_model,,}" == *.gguf ]]; then
    jq -e --argjson id "$gguf_model_id" \
      '.[] | select(.id == $id and ((.format // .framework) == "gguf"))' \
      "$output_dir/models-initial.json" >/dev/null \
      || die "raw model 1 is not reported as GGUF"
  fi

  assert_runtime_markers "$runtime_log" 1 || die "runtime did not use both sides of the shared pool"
  run_onnx_inference_if_configured "$output_dir/onnx-inference-initial.json"
  run_gguf_inference "$output_dir/gguf-inference-initial.json"
  assert_gguf_response_correct "$output_dir/gguf-inference-initial.json"
  run_gguf_concurrency_transition_test "$output_dir/gguf-concurrency-transition"
  sleep "$settle_seconds"
  api_get /metrics >"$output_dir/metrics-initial.txt"

  initial_pool="$(metric_from_file kapsl_device_memory_pooled_bytes "$output_dir/metrics-initial.txt")"
  initial_external="$(metric_from_file kapsl_device_memory_external_bytes "$output_dir/metrics-initial.txt")"
  initial_allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$output_dir/metrics-initial.txt")"
  initial_live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$output_dir/metrics-initial.txt")"
  initial_free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$output_dir/metrics-initial.txt")"
  initial_free_ranges="$(metric_from_file kapsl_gpu_device_pool_free_ranges "$output_dir/metrics-initial.txt")"
  initial_largest_free="$(metric_from_file kapsl_gpu_device_pool_largest_free_range_bytes "$output_dir/metrics-initial.txt")"
  initial_fragmentation="$(metric_from_file kapsl_gpu_device_pool_fragmentation_ratio "$output_dir/metrics-initial.txt")"
  [[ "$initial_pool" =~ ^[1-9][0-9]*$ ]] || die "missing/non-positive pooled-bytes metric: $initial_pool"
  [[ "$initial_external" =~ ^[1-9][0-9]*$ ]] || die "GGUF weights did not produce positive external-memory accounting: $initial_external"
  assert_pool_snapshot_consistent "$output_dir/metrics-initial.txt" "$initial_pool" \
    || die "initial live allocator snapshot is missing or inconsistent"
  require_active_owner "$output_dir/metrics-initial.txt" onnx \
    || die "ONNX has no positive live allocation/admission in the shared pool"
  require_admitted_owner "$output_dir/metrics-initial.txt" "gguf_kv:$gguf_model_id" \
    || die "GGUF shared-KV owner is not admitted with a valid quota snapshot"
  initial_onnx_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes onnx "$output_dir/metrics-initial.txt")"
  initial_gguf_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes "gguf_kv:$gguf_model_id" "$output_dir/metrics-initial.txt")"
  if grep -F -q -- 'owner="native_kv:' "$output_dir/metrics-initial.txt"; then
    die "native-KV owner unexpectedly appeared in a shared-KV integration run"
  fi
  initial_vram="$(process_vram_bytes || true)"

  note "stopping GGUF while ORT remains active"
  api_post "/api/models/$gguf_model_id/stop" >"$output_dir/stop-response.json"
  wait_for_model_status "$gguf_model_id" inactive || die "GGUF model did not stop"
  wait_for_model_status "$onnx_model_id" active || die "ORT model stopped during GGUF unload"
  wait_for_log_count "[gguf] Backend unloaded" 1 || die "GGUF backend did not report unload"

  wait_for_stopped_pool_snapshot \
    "$initial_external" "$initial_allocated" "$initial_live" "$initial_free" \
    "$initial_pool" "$output_dir/metrics-stopped.txt" \
    || die "GGUF unload did not release its pool owner/ranges and external weights"
  stopped_external="$(metric_from_file kapsl_device_memory_external_bytes "$output_dir/metrics-stopped.txt")"
  stopped_pool="$(metric_from_file kapsl_device_memory_pooled_bytes "$output_dir/metrics-stopped.txt")"
  stopped_allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$output_dir/metrics-stopped.txt")"
  stopped_live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$output_dir/metrics-stopped.txt")"
  stopped_free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$output_dir/metrics-stopped.txt")"
  stopped_onnx_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes onnx "$output_dir/metrics-stopped.txt")"
  [[ "$stopped_pool" == "$initial_pool" ]] \
    || die "pool backing changed during unload: $initial_pool -> $stopped_pool"
  api_get /api/models >"$output_dir/models-stopped.json"

  note "reloading GGUF model id $gguf_model_id"
  start_payload="$(jq -cn --arg path "$gguf_model" --argjson id "$gguf_model_id" \
    '{model_path:$path,model_id:$id,topology:"data-parallel",tp_degree:1}')"
  api_post /api/models/start \
    -H 'Content-Type: application/json' \
    --data-binary "$start_payload" >"$output_dir/start-response.json"
  wait_for_model_status "$gguf_model_id" active || die "GGUF model did not become active after reload"
  wait_for_model_status "$onnx_model_id" active || die "ORT model stopped during GGUF reload"
  wait_for_log_count "kv_path=shared-kv" 2 || die "reloaded GGUF did not return to shared-KV"
  run_gguf_inference "$output_dir/gguf-inference-reloaded.json"
  sleep "$settle_seconds"

  api_get /api/models >"$output_dir/models-reloaded.json"
  wait_for_reloaded_pool_snapshot \
    "$initial_external" "$initial_allocated" "$initial_live" "$initial_free" \
    "$initial_pool" "$output_dir/metrics-reloaded.txt" \
    || die "GGUF reload did not recover the original live/free pool snapshot"
  reloaded_pool="$(metric_from_file kapsl_device_memory_pooled_bytes "$output_dir/metrics-reloaded.txt")"
  reloaded_external="$(metric_from_file kapsl_device_memory_external_bytes "$output_dir/metrics-reloaded.txt")"
  reloaded_allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$output_dir/metrics-reloaded.txt")"
  reloaded_live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$output_dir/metrics-reloaded.txt")"
  reloaded_free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$output_dir/metrics-reloaded.txt")"
  reloaded_free_ranges="$(metric_from_file kapsl_gpu_device_pool_free_ranges "$output_dir/metrics-reloaded.txt")"
  reloaded_largest_free="$(metric_from_file kapsl_gpu_device_pool_largest_free_range_bytes "$output_dir/metrics-reloaded.txt")"
  reloaded_fragmentation="$(metric_from_file kapsl_gpu_device_pool_fragmentation_ratio "$output_dir/metrics-reloaded.txt")"
  reloaded_onnx_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes onnx "$output_dir/metrics-reloaded.txt")"
  reloaded_gguf_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes "gguf_kv:$gguf_model_id" "$output_dir/metrics-reloaded.txt")"
  reloaded_vram="$(process_vram_bytes || true)"

  [[ "$reloaded_pool" == "$initial_pool" ]] \
    || die "reload grew/replaced the pool backing: $initial_pool -> $reloaded_pool"
  [[ "$reloaded_external" == "$initial_external" ]] \
    || die "reload changed external-memory accounting: $initial_external -> $reloaded_external"
  assert_fragmentation_not_worse \
    "$initial_fragmentation" "$reloaded_fragmentation" "$fragmentation_tolerance" \
    || die "fragmentation grew after GGUF reload: $initial_fragmentation -> $reloaded_fragmentation"
  assert_runtime_markers "$runtime_log" 2 || die "reload used a fallback or allocated a second pool"

  if [[ "$initial_vram" =~ ^[0-9]+$ ]] && [[ "$reloaded_vram" =~ ^[0-9]+$ ]]; then
    max_vram=$((initial_vram + vram_growth_bytes))
    (( reloaded_vram <= max_vram )) \
      || die "process VRAM grew beyond tolerance after reload: $initial_vram -> $reloaded_vram (tolerance $vram_growth_bytes)"
  elif is_true "${KAPSL_GPU_TEST_REQUIRE_PROCESS_VRAM:-0}"; then
    die "nvidia-smi did not expose per-process VRAM for runtime pid $runtime_pid"
  else
    note "per-process VRAM unavailable; exact pool/external metrics remain authoritative"
  fi

  note "stopping ORT while GGUF remains active"
  api_post "/api/models/$onnx_model_id/stop" >"$output_dir/ort-stop-response.json"
  wait_for_model_status "$onnx_model_id" inactive || die "ORT model did not stop"
  wait_for_model_status "$gguf_model_id" active || die "GGUF model stopped during ORT unload"
  wait_for_ort_stopped_pool_snapshot \
    "$initial_external" "$initial_allocated" "$initial_live" "$initial_free" \
    "$initial_pool" "$output_dir/metrics-ort-stopped.txt" \
    || die "ORT unload did not release its owner/ranges while GGUF remained resident"
  ort_stopped_allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$output_dir/metrics-ort-stopped.txt")"
  ort_stopped_live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$output_dir/metrics-ort-stopped.txt")"
  ort_stopped_free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$output_dir/metrics-ort-stopped.txt")"
  ort_stopped_external="$(metric_from_file kapsl_device_memory_external_bytes "$output_dir/metrics-ort-stopped.txt")"
  ort_stopped_gguf_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes "gguf_kv:$gguf_model_id" "$output_dir/metrics-ort-stopped.txt")"
  api_get /api/models >"$output_dir/models-ort-stopped.json"

  note "reloading ORT model id $onnx_model_id"
  ort_start_payload="$(jq -cn --arg path "$ort_model" --argjson id "$onnx_model_id" \
    '{model_path:$path,model_id:$id,topology:"data-parallel",tp_degree:1}')"
  api_post /api/models/start \
    -H 'Content-Type: application/json' \
    --data-binary "$ort_start_payload" >"$output_dir/ort-start-response.json"
  wait_for_model_status "$onnx_model_id" active || die "ORT model did not become active after reload"
  wait_for_model_status "$gguf_model_id" active || die "GGUF model stopped during ORT reload"
  run_onnx_inference_if_configured "$output_dir/onnx-inference-reloaded.json"
  sleep "$settle_seconds"
  wait_for_ort_reloaded_pool_snapshot \
    "$initial_external" "$ort_stopped_external" \
    "$initial_allocated" "$initial_live" "$initial_free" \
    "$initial_pool" "$output_dir/metrics-final.txt" \
    || die "ORT reload did not recover the original live/free pool snapshot"
  api_get /api/models >"$output_dir/models-final.json"
  final_pool="$(metric_from_file kapsl_device_memory_pooled_bytes "$output_dir/metrics-final.txt")"
  final_external="$(metric_from_file kapsl_device_memory_external_bytes "$output_dir/metrics-final.txt")"
  final_allocated="$(metric_from_file kapsl_gpu_device_pool_allocated_bytes "$output_dir/metrics-final.txt")"
  final_live="$(metric_from_file kapsl_gpu_device_pool_live_allocations "$output_dir/metrics-final.txt")"
  final_free="$(metric_from_file kapsl_gpu_device_pool_free_bytes "$output_dir/metrics-final.txt")"
  final_free_ranges="$(metric_from_file kapsl_gpu_device_pool_free_ranges "$output_dir/metrics-final.txt")"
  final_largest_free="$(metric_from_file kapsl_gpu_device_pool_largest_free_range_bytes "$output_dir/metrics-final.txt")"
  final_fragmentation="$(metric_from_file kapsl_gpu_device_pool_fragmentation_ratio "$output_dir/metrics-final.txt")"
  final_onnx_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes onnx "$output_dir/metrics-final.txt")"
  final_gguf_usage="$(owner_metric_from_file kapsl_gpu_device_pool_owner_usage_bytes "gguf_kv:$gguf_model_id" "$output_dir/metrics-final.txt")"
  final_vram="$(process_vram_bytes || true)"

  [[ "$final_pool" == "$initial_pool" ]] || die "ORT reload replaced the pool backing"
  (( final_external >= ort_stopped_external && final_external <= initial_external )) \
    || die "ORT reload external accounting escaped its stopped/initial bounds"
  assert_fragmentation_not_worse \
    "$initial_fragmentation" "$final_fragmentation" "$fragmentation_tolerance" \
    || die "fragmentation grew after ORT reload: $initial_fragmentation -> $final_fragmentation"
  assert_runtime_markers "$runtime_log" 2 || die "ORT reload allocated another pool or changed GGUF path"
  if [[ "$initial_vram" =~ ^[0-9]+$ ]] && [[ "$final_vram" =~ ^[0-9]+$ ]]; then
    max_vram=$((initial_vram + vram_growth_bytes))
    (( final_vram <= max_vram )) \
      || die "process VRAM grew beyond tolerance after both reloads: $initial_vram -> $final_vram (tolerance $vram_growth_bytes)"
  elif is_true "${KAPSL_GPU_TEST_REQUIRE_PROCESS_VRAM:-0}"; then
    die "nvidia-smi did not expose final per-process VRAM for runtime pid $runtime_pid"
  fi

  note "stopping all consumers and verifying backing reclamation"
  api_post "/api/models/$gguf_model_id/stop" >"$output_dir/final-gguf-stop-response.json"
  api_post "/api/models/$onnx_model_id/stop" >"$output_dir/final-ort-stop-response.json"
  wait_for_model_status "$gguf_model_id" inactive || die "GGUF model did not stop for reclamation"
  wait_for_model_status "$onnx_model_id" inactive || die "ORT model did not stop for reclamation"
  reclaim_deadline=$((SECONDS + load_timeout_seconds))
  while (( SECONDS < reclaim_deadline )); do
    api_get /metrics >"$output_dir/metrics-reclaimed.txt.next" 2>/dev/null || true
    reclaimed_pool="$(metric_from_file kapsl_device_memory_pooled_bytes "$output_dir/metrics-reclaimed.txt.next")"
    if [[ "$reclaimed_pool" == "0" ]]; then
      mv "$output_dir/metrics-reclaimed.txt.next" "$output_dir/metrics-reclaimed.txt"
      break
    fi
    sleep 1
  done
  [[ "${reclaimed_pool:-}" == "0" ]] \
    || die "idle pool backing was not reclaimed after all consumers stopped"

  cat >"$output_dir/summary.env" <<EOF
status=passed
runtime_pid=$runtime_pid
pool_bytes=$initial_pool
pool_bytes_after_all_stopped=$reclaimed_pool
external_bytes_initial=$initial_external
external_bytes_stopped=$stopped_external
external_bytes_reloaded=$reloaded_external
external_bytes_ort_stopped=$ort_stopped_external
external_bytes_final=$final_external
pool_allocated_bytes_initial=$initial_allocated
pool_allocated_bytes_stopped=$stopped_allocated
pool_allocated_bytes_reloaded=$reloaded_allocated
pool_allocated_bytes_ort_stopped=$ort_stopped_allocated
pool_allocated_bytes_final=$final_allocated
pool_live_allocations_initial=$initial_live
pool_live_allocations_stopped=$stopped_live
pool_live_allocations_reloaded=$reloaded_live
pool_live_allocations_ort_stopped=$ort_stopped_live
pool_live_allocations_final=$final_live
pool_free_bytes_initial=$initial_free
pool_free_bytes_stopped=$stopped_free
pool_free_bytes_reloaded=$reloaded_free
pool_free_bytes_ort_stopped=$ort_stopped_free
pool_free_bytes_final=$final_free
pool_free_ranges_initial=$initial_free_ranges
pool_free_ranges_reloaded=$reloaded_free_ranges
pool_free_ranges_final=$final_free_ranges
pool_largest_free_range_bytes_initial=$initial_largest_free
pool_largest_free_range_bytes_reloaded=$reloaded_largest_free
pool_largest_free_range_bytes_final=$final_largest_free
pool_fragmentation_ratio_initial=$initial_fragmentation
pool_fragmentation_ratio_reloaded=$reloaded_fragmentation
pool_fragmentation_ratio_final=$final_fragmentation
pool_owner_onnx_usage_bytes_initial=$initial_onnx_usage
pool_owner_onnx_usage_bytes_stopped=$stopped_onnx_usage
pool_owner_onnx_usage_bytes_reloaded=$reloaded_onnx_usage
pool_owner_onnx_usage_bytes_final=$final_onnx_usage
pool_owner_gguf_usage_bytes_initial=$initial_gguf_usage
pool_owner_gguf_usage_bytes_reloaded=$reloaded_gguf_usage
pool_owner_gguf_usage_bytes_ort_stopped=$ort_stopped_gguf_usage
pool_owner_gguf_usage_bytes_final=$final_gguf_usage
process_vram_bytes_initial=${initial_vram:-unavailable}
process_vram_bytes_reloaded=${reloaded_vram:-unavailable}
process_vram_bytes_final=${final_vram:-unavailable}
EOF

  note "PASS: ORT and GGUF shared one pool; both unload/reload cycles reused it without growth"
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --self-test)
    run_self_test
    exit 0
    ;;
  "")
    main
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
