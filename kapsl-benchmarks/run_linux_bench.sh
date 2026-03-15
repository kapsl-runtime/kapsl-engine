#!/usr/bin/env bash
# run_linux_bench.sh — standalone Linux benchmark for kapsl-engine.
#
# Runs kapsl-only by default; optionally compares against a running vLLM.
# Embeds Linux system info (CPU, memory, NUMA, thermal, governor) in results.
#
# Quick start (kapsl-only):
#   ./run_linux_bench.sh --model ./models/qwen/qwen2.5-1.5b-instruct.aimod
#
# With vLLM comparison:
#   ./run_linux_bench.sh --model ./models/qwen/qwen2.5-1.5b-instruct.aimod \
#       --vllm-model Qwen/Qwen2.5-1.5B-Instruct
#
# Skip rebuild if binary already exists:
#   ./run_linux_bench.sh --model ./models/... --no-build

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/../kapsl-runtime" && pwd)"
RUNTIME_BIN="$RUNTIME_DIR/target/release/kapsl"

KAPSL_URL="${KAPSL_URL:-http://127.0.0.1:9195}"
KAPSL_MODEL_ID="${KAPSL_MODEL_ID:-0}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000}"
VLLM_MODEL="${VLLM_MODEL:-}"
MODEL_PATH="${MODEL_PATH:-}"
REQUESTS="${REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-1,4,8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
WARMUP="${WARMUP:-3}"
TIMEOUT="${TIMEOUT:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
QUEUE_DELAY_MS="${QUEUE_DELAY_MS:-1}"
BUILD_RELEASE=1
KEEP_RUNNING=0
OUTPUT=""
NO_GGUF=0

# ── Argument parsing ─────────────────────────────────────────────────────────

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

  --model PATH           Path to .aimod model package (required)
  --output PATH          Output JSON path (default: auto-timestamped)
  --requests N           Requests per concurrency level (default: $REQUESTS)
  --concurrency LIST     Comma-separated concurrency levels (default: $CONCURRENCY)
  --max-tokens N         Max new tokens per request (default: $MAX_TOKENS)
  --temperature X        Sampling temperature (default: $TEMPERATURE)
  --warmup N             Warmup requests before timing (default: $WARMUP)
  --vllm-model NAME      Enable kapsl vs vLLM comparison (vLLM must be running)
  --no-build             Skip cargo build --release
  --no-gguf              Build without GGUF/llama.cpp (faster build, no GGUF models)
  --keep-running         Do not stop kapsl after the benchmark
  -h, --help             Show this help

Environment overrides:
  KAPSL_URL  VLLM_URL  KAPSL_MODEL_ID  REQUESTS  CONCURRENCY
  MAX_TOKENS  TEMPERATURE  WARMUP  TIMEOUT  BATCH_SIZE  QUEUE_DELAY_MS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL_PATH="$2";   shift 2 ;;
    --output)      OUTPUT="$2";       shift 2 ;;
    --requests)    REQUESTS="$2";     shift 2 ;;
    --concurrency) CONCURRENCY="$2";  shift 2 ;;
    --max-tokens)  MAX_TOKENS="$2";   shift 2 ;;
    --temperature) TEMPERATURE="$2";  shift 2 ;;
    --warmup)      WARMUP="$2";       shift 2 ;;
    --vllm-model)  VLLM_MODEL="$2";   shift 2 ;;
    --no-build)    BUILD_RELEASE=0;   shift ;;
    --no-gguf)     NO_GGUF=1;         shift ;;
    --keep-running) KEEP_RUNNING=1;   shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "error: --model is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  exit 1
fi

# ── Output path ───────────────────────────────────────────────────────────────

if [[ -z "$OUTPUT" ]]; then
  TS="$(date +%Y%m%d-%H%M%S)"
  OUTPUT="$SCRIPT_DIR/results-linux-${TS}.json"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

step()  { echo; echo "━━━━━━━━  $*  ━━━━━━━━"; }
info()  { echo "  $*"; }
ok()    { echo "  ✓ $*"; }
warn()  { echo "  ⚠ $*"; }
die()   { echo "error: $*" >&2; exit 1; }

wait_http() {
  local url="$1" label="$2" tries="${3:-120}"
  echo -n "  Waiting for $label "
  for _ in $(seq 1 "$tries"); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then echo "✓"; return 0; fi
    echo -n "."
    sleep 1
  done
  echo " ✗ (timed out)"
  return 1
}

python_bin() {
  local venv="$SCRIPT_DIR/.venv/bin/python"
  if [[ -x "$venv" ]]; then echo "$venv"; return; fi
  for py in python3 python; do
    if command -v "$py" &>/dev/null; then echo "$py"; return; fi
  done
  die "python3 not found; create a venv at $SCRIPT_DIR/.venv or install python3"
}

# ── System info (Linux-specific) ──────────────────────────────────────────────

collect_sysinfo() {
  step "Collecting Linux system info"

  # Kernel
  SYSINFO_KERNEL="$(uname -r)"
  ok "kernel: $SYSINFO_KERNEL"

  # CPU model and counts
  SYSINFO_CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo unknown)"
  SYSINFO_CPU_PHYSICAL="$(grep 'physical id' /proc/cpuinfo 2>/dev/null | sort -u | wc -l || echo 1)"
  SYSINFO_CPU_CORES="$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || nproc)"
  ok "cpu: $SYSINFO_CPU_MODEL"
  ok "cpu cores (logical): $SYSINFO_CPU_CORES  physical sockets: $SYSINFO_CPU_PHYSICAL"

  # NUMA topology
  if command -v numactl &>/dev/null; then
    SYSINFO_NUMA="$(numactl --hardware 2>/dev/null | head -4 | tr '\n' '|' | sed 's/|$//')"
  elif [[ -d /sys/devices/system/node ]]; then
    SYSINFO_NUMA="nodes=$(ls /sys/devices/system/node | grep -c '^node[0-9]' || echo 1)"
  else
    SYSINFO_NUMA="unavailable"
  fi
  ok "numa: $SYSINFO_NUMA"

  # RAM
  SYSINFO_RAM_TOTAL_KB="$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)"
  SYSINFO_RAM_AVAIL_KB="$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)"
  SYSINFO_RAM_TOTAL_GB="$(awk "BEGIN{printf \"%.1f\", $SYSINFO_RAM_TOTAL_KB/1048576}")"
  SYSINFO_RAM_AVAIL_GB="$(awk "BEGIN{printf \"%.1f\", $SYSINFO_RAM_AVAIL_KB/1048576}")"
  ok "ram: total=${SYSINFO_RAM_TOTAL_GB}GiB  available=${SYSINFO_RAM_AVAIL_GB}GiB"

  # CPU frequency governor
  SYSINFO_GOV="unknown"
  local gov_path="/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
  if [[ -r "$gov_path" ]]; then
    SYSINFO_GOV="$(cat "$gov_path")"
  fi
  ok "cpu governor: $SYSINFO_GOV"
  if [[ "$SYSINFO_GOV" != "performance" ]]; then
    warn "governor is '$SYSINFO_GOV' (not 'performance') — latency variance may be higher"
    warn "to pin: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
  fi

  # Current CPU frequencies (MHz)
  local freq_path="/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
  if [[ -r "$freq_path" ]]; then
    SYSINFO_CPU_FREQ_MHZ="$(awk '{printf "%.0f", $1/1000}' "$freq_path")"
    ok "cpu0 current freq: ${SYSINFO_CPU_FREQ_MHZ} MHz"
  else
    SYSINFO_CPU_FREQ_MHZ="unknown"
  fi

  # Transparent huge pages
  SYSINFO_THP="unknown"
  if [[ -r /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    SYSINFO_THP="$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  fi
  ok "transparent huge pages: $SYSINFO_THP"

  # Thermal zones (first 3)
  SYSINFO_THERMAL="none"
  local thermal_parts=()
  for zone in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "$zone" ]] || continue
    local zone_dir
    zone_dir="$(dirname "$zone")"
    local zone_name
    zone_name="$(cat "$zone_dir/type" 2>/dev/null || echo "zone")"
    local temp_mc
    temp_mc="$(cat "$zone" 2>/dev/null || echo 0)"
    local temp_c
    temp_c="$(awk "BEGIN{printf \"%.1f\", $temp_mc/1000}")"
    thermal_parts+=("${zone_name}=${temp_c}°C")
    [[ ${#thermal_parts[@]} -ge 3 ]] && break
  done
  if [[ ${#thermal_parts[@]} -gt 0 ]]; then
    SYSINFO_THERMAL="$(IFS=','; echo "${thermal_parts[*]}")"
    ok "thermal: $SYSINFO_THERMAL"
  fi

  # GPU (lspci — best-effort)
  SYSINFO_GPU="none"
  if command -v lspci &>/dev/null; then
    SYSINFO_GPU="$(lspci 2>/dev/null | grep -iE 'VGA|3D|Display|NVIDIA|AMD|Intel.*Graphics' | head -2 | tr '\n' ';' | sed 's/;$//' || echo none)"
    ok "gpu: ${SYSINFO_GPU:-none}"
  fi

  # CUDA availability
  SYSINFO_CUDA="unavailable"
  if command -v nvidia-smi &>/dev/null; then
    SYSINFO_CUDA="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ';' | sed 's/;$//' || echo 'nvidia-smi error')"
    ok "cuda: $SYSINFO_CUDA"
  fi

  # OS release
  SYSINFO_OS="$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
  ok "os: $SYSINFO_OS"

  # Rustc version
  SYSINFO_RUSTC="$(rustc --version 2>/dev/null || echo unknown)"
  ok "rustc: $SYSINFO_RUSTC"
}

# ── Build ─────────────────────────────────────────────────────────────────────

build_kapsl() {
  step "Building kapsl release binary"
  local features_flag=""
  if [[ "$NO_GGUF" -eq 1 ]]; then
    features_flag="--no-default-features"
    info "(--no-gguf: skipping llama.cpp build)"
  fi
  (cd "$RUNTIME_DIR" && cargo build -p kapsl --release $features_flag)
  ok "binary: $RUNTIME_BIN ($(du -sh "$RUNTIME_BIN" 2>/dev/null | cut -f1 || echo '?'))"
}

# ── kapsl startup / teardown ─────────────────────────────────────────────────

KAPSL_PID=""

start_kapsl() {
  step "Starting kapsl runtime"
  # Kill any leftover instance on the same port.
  pkill -f "kapsl.*--metrics-port.*9195" 2>/dev/null || true
  sleep 1

  KAPSL_LOG="$SCRIPT_DIR/kapsl-linux-bench.log"
  KAPSL_PROVIDER_POLICY=manifest \
  KAPSL_LLM_SAFE_LOAD=0 \
  KAPSL_LLM_ISOLATE_PROCESS=0 \
    "$RUNTIME_BIN" run \
      --model "$MODEL_PATH" \
      --socket /tmp/kapsl-linux-bench.sock \
      --metrics-port 9195 \
      --http-bind 127.0.0.1 \
      --batch-size "$BATCH_SIZE" \
      --scheduler-max-micro-batch "$BATCH_SIZE" \
      --scheduler-queue-delay-ms "$QUEUE_DELAY_MS" \
      >"$KAPSL_LOG" 2>&1 &
  KAPSL_PID=$!
  info "pid=$KAPSL_PID  log=$KAPSL_LOG"
}

stop_kapsl() {
  if [[ -n "$KAPSL_PID" ]] && [[ "$KEEP_RUNNING" -eq 0 ]]; then
    kill "$KAPSL_PID" 2>/dev/null || true
    wait "$KAPSL_PID" 2>/dev/null || true
  fi
}
trap stop_kapsl EXIT

# ── vLLM check ────────────────────────────────────────────────────────────────

check_vllm() {
  if [[ -n "$VLLM_MODEL" ]]; then
    step "Checking vLLM availability"
    if ! curl -s -o /dev/null -w '%{http_code}' "$VLLM_URL/v1/models" 2>/dev/null | grep -q '^200$'; then
      warn "vLLM not reachable at $VLLM_URL/v1/models — running kapsl-only"
      VLLM_MODEL=""
    else
      ok "vLLM reachable at $VLLM_URL, model=$VLLM_MODEL"
    fi
  fi
}

# ── Thermal snapshot ─────────────────────────────────────────────────────────

thermal_snapshot() {
  local parts=()
  for zone in /sys/class/thermal/thermal_zone*/temp; do
    [[ -r "$zone" ]] || continue
    local zone_dir
    zone_dir="$(dirname "$zone")"
    local zone_name
    zone_name="$(cat "$zone_dir/type" 2>/dev/null || echo "zone")"
    local temp_mc
    temp_mc="$(cat "$zone" 2>/dev/null || echo 0)"
    local temp_c
    temp_c="$(awk "BEGIN{printf \"%.1f\", $temp_mc/1000}")"
    parts+=("${zone_name}=${temp_c}")
    [[ ${#parts[@]} -ge 3 ]] && break
  done
  if [[ ${#parts[@]} -gt 0 ]]; then
    local IFS=','
    echo "${parts[*]}"
  else
    echo "unavailable"
  fi
}

# ── Embed sysinfo into result JSON ────────────────────────────────────────────

embed_sysinfo() {
  local result_file="$1"
  local thermal_after="$2"
  [[ -f "$result_file" ]] || return

  PYTHON="$(python_bin)"
  "$PYTHON" - "$result_file" <<PYEOF
import json, sys, datetime

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

data["linux_sysinfo"] = {
    "kernel":         "$SYSINFO_KERNEL",
    "os":             "$SYSINFO_OS",
    "cpu_model":      "$SYSINFO_CPU_MODEL",
    "cpu_logical":    int("$SYSINFO_CPU_CORES"),
    "cpu_sockets":    int("$SYSINFO_CPU_PHYSICAL"),
    "numa":           "$SYSINFO_NUMA",
    "ram_total_gib":  float("$SYSINFO_RAM_TOTAL_GB"),
    "ram_avail_gib":  float("$SYSINFO_RAM_AVAIL_GB"),
    "cpu_governor":   "$SYSINFO_GOV",
    "cpu_freq_mhz":   "$SYSINFO_CPU_FREQ_MHZ",
    "thp":            "$SYSINFO_THP",
    "thermal_before": "$SYSINFO_THERMAL",
    "thermal_after":  "$thermal_after",
    "gpu":            "$SYSINFO_GPU",
    "cuda":           "$SYSINFO_CUDA",
    "rustc":          "$SYSINFO_RUSTC",
    "batch_size":     int("$BATCH_SIZE"),
    "queue_delay_ms": int("$QUEUE_DELAY_MS"),
}

with open(path, "w") as f:
    json.dump(data, f, indent=2)
print(f"sysinfo embedded in {path}")
PYEOF
}

# ── Print summary ─────────────────────────────────────────────────────────────

print_summary() {
  local result_file="$1"
  [[ -f "$result_file" ]] || return

  PYTHON="$(python_bin)"
  "$PYTHON" - "$result_file" <<'PYEOF'
import json, sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

si = data.get("linux_sysinfo", {})
print()
print("=" * 68)
print("LINUX BENCHMARK SUMMARY")
print("=" * 68)
print(f"  CPU:      {si.get('cpu_model', 'unknown')}")
print(f"  Cores:    {si.get('cpu_logical')} logical / {si.get('cpu_sockets')} socket(s)")
print(f"  RAM:      {si.get('ram_total_gib')} GiB total, {si.get('ram_avail_gib')} GiB available")
print(f"  Governor: {si.get('cpu_governor')}")
print(f"  Thermal (before/after): {si.get('thermal_before')} → {si.get('thermal_after')}")
print(f"  GPU:      {si.get('gpu', 'none')}")
print(f"  CUDA:     {si.get('cuda', 'unavailable')}")
print()

def fmt_result(label, results_by_conc):
    print(f"  [{label}]")
    print(f"  {'conc':>6}  {'req/s':>8}  {'p50 ms':>8}  {'p90 ms':>8}  {'p99 ms':>8}  {'tok/s':>8}  {'errs':>5}")
    print("  " + "-" * 60)
    for conc_key in sorted(results_by_conc.keys(), key=lambda k: int(k.lstrip("c"))):
        r = results_by_conc[conc_key]
        lat = r.get("latency_ms") or {}
        tok = r.get("overall_tok_per_sec", 0)
        print(
            f"  {r.get('concurrency', '?'):>6}"
            f"  {r.get('throughput_rps', 0):>8.2f}"
            f"  {lat.get('p50', 0):>8.0f}"
            f"  {lat.get('p90', 0):>8.0f}"
            f"  {lat.get('p99', 0):>8.0f}"
            f"  {tok:>8.1f}"
            f"  {r.get('errors', 0):>5}"
        )
    print()

kapsl_data = data.get("kapsl", {})
vllm_data  = data.get("vllm", {})

if kapsl_data:
    fmt_result("kapsl", kapsl_data)

if vllm_data:
    fmt_result("vLLM", vllm_data)

# Speedup table
if kapsl_data and vllm_data:
    print("  [kapsl / vLLM speedup]")
    print(f"  {'conc':>6}  {'tp ratio':>10}  {'lat p50 ratio':>14}  {'tok/s ratio':>12}")
    print("  " + "-" * 48)
    for conc_key in sorted(kapsl_data.keys(), key=lambda k: int(k.lstrip("c"))):
        if conc_key not in vllm_data:
            continue
        kr = kapsl_data[conc_key]
        vr = vllm_data[conc_key]
        tp_ratio = (kr.get("throughput_rps", 0) / vr["throughput_rps"]) if vr.get("throughput_rps") else 0
        lat_ratio = (kr.get("latency_ms", {}).get("p50", 0) / vr["latency_ms"]["p50"]) if vr.get("latency_ms", {}).get("p50") else 0
        tok_ratio = (kr.get("overall_tok_per_sec", 0) / vr["overall_tok_per_sec"]) if vr.get("overall_tok_per_sec") else 0
        print(f"  {kr.get('concurrency', '?'):>6}  {tp_ratio:>10.3f}x  {lat_ratio:>14.3f}x  {tok_ratio:>12.3f}x")
    print()

print(f"  Results: {path}")
print("=" * 68)
PYEOF
}

# ── Main ─────────────────────────────────────────────────────────────────────

collect_sysinfo

check_vllm

if [[ "$BUILD_RELEASE" -eq 1 ]]; then
  build_kapsl
fi

if [[ ! -x "$RUNTIME_BIN" ]]; then
  die "kapsl binary not found: $RUNTIME_BIN  (run with --build or build manually)"
fi

start_kapsl

wait_http "$KAPSL_URL/api/models" "kapsl" 120 || {
  echo "kapsl startup log:"
  tail -30 "$KAPSL_LOG" >&2
  die "kapsl did not become ready"
}

step "Running benchmark"
info "model:       $MODEL_PATH"
info "requests:    $REQUESTS per concurrency level"
info "concurrency: $CONCURRENCY"
info "max_tokens:  $MAX_TOKENS"
info "warmup:      $WARMUP"

PYTHON="$(python_bin)"

# Ensure requests dep available
"$PYTHON" -c "import requests" 2>/dev/null || {
  warn "'requests' not installed — installing into venv or system"
  "$PYTHON" -m pip install --quiet requests
}

THERMAL_BEFORE="$(thermal_snapshot)"

BENCH_ARGS=(
  "$SCRIPT_DIR/bench_comprehensive.py"
  "--kapsl-url"      "$KAPSL_URL"
  "--kapsl-model-id" "$KAPSL_MODEL_ID"
  "--requests"       "$REQUESTS"
  "--concurrency"    "$CONCURRENCY"
  "--max-tokens"     "$MAX_TOKENS"
  "--temperature"    "$TEMPERATURE"
  "--warmup"         "$WARMUP"
  "--timeout"        "$TIMEOUT"
  "--output"         "$OUTPUT"
)

if [[ -n "$VLLM_MODEL" ]]; then
  BENCH_ARGS+=("--vllm-url" "$VLLM_URL" "--vllm-model" "$VLLM_MODEL")
else
  BENCH_ARGS+=("--kapsl-only")
fi

"$PYTHON" "${BENCH_ARGS[@]}"

THERMAL_AFTER="$(thermal_snapshot)"

step "Embedding system info into results"
embed_sysinfo "$OUTPUT" "$THERMAL_AFTER"

print_summary "$OUTPUT"
