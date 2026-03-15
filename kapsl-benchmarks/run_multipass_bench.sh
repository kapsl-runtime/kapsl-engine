#!/usr/bin/env bash
# run_multipass_bench.sh — run kapsl-engine benchmarks inside a Multipass VM.
#
# Delegates all benchmark logic to run_linux_bench.sh inside the VM, which
# provides the comprehensive multi-concurrency results + Linux system info
# (CPU, NUMA, thermal, governor).
#
# VM prerequisites (one-time setup):
#   multipass launch 24.04 --name kapsl-bench --cpus 4 --memory 8G --disk 50G
#
# Quick start:
#   ./run_multipass_bench.sh --model ./models/qwen/qwen2.5-1.5b-instruct.aimod
#
# With vLLM comparison:
#   ./run_multipass_bench.sh \
#       --model ./models/qwen/qwen2.5-1.5b-instruct.aimod \
#       --vllm-model Qwen/Qwen2.5-1.5B-Instruct
#
# Skip rebuild on subsequent runs:
#   ./run_multipass_bench.sh --model ./models/... --no-build

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/../kapsl-runtime" && pwd)"

VM_NAME="${VM_NAME:-kapsl-bench}"
VM_BENCH_DIR="/home/ubuntu/kapsl-benchmarks"
VM_RUNTIME_DIR="/home/ubuntu/kapsl-runtime"

MODEL_PATH=""
VLLM_MODEL=""
REQUESTS="${REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-1,4,8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
WARMUP="${WARMUP:-3}"
TIMEOUT="${TIMEOUT:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
QUEUE_DELAY_MS="${QUEUE_DELAY_MS:-1}"
BUILD_RELEASE=1
NO_GGUF=0
VLLM_START=0

# ── Argument parsing ──────────────────────────────────────────────────────────

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

  --model PATH           Path to .aimod model package on host (required)
  --vllm-model NAME      Enable kapsl vs vLLM comparison; starts vLLM in the VM
  --requests N           Requests per concurrency level (default: $REQUESTS)
  --concurrency LIST     Comma-separated concurrency levels (default: $CONCURRENCY)
  --max-tokens N         Max new tokens per request (default: $MAX_TOKENS)
  --temperature X        Sampling temperature (default: $TEMPERATURE)
  --warmup N             Warmup requests (default: $WARMUP)
  --no-build             Skip cargo build --release inside VM
  --no-gguf              Build without llama.cpp (faster build, no GGUF models)
  --vm NAME              Multipass VM name (default: $VM_NAME)
  -h, --help             Show this help

Environment overrides:
  VM_NAME  REQUESTS  CONCURRENCY  MAX_TOKENS  TEMPERATURE  WARMUP
  TIMEOUT  BATCH_SIZE  QUEUE_DELAY_MS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL_PATH="$2";    shift 2 ;;
    --vllm-model)  VLLM_MODEL="$2";    shift 2 ;;
    --requests)    REQUESTS="$2";      shift 2 ;;
    --concurrency) CONCURRENCY="$2";   shift 2 ;;
    --max-tokens)  MAX_TOKENS="$2";    shift 2 ;;
    --temperature) TEMPERATURE="$2";   shift 2 ;;
    --warmup)      WARMUP="$2";        shift 2 ;;
    --timeout)     TIMEOUT="$2";       shift 2 ;;
    --no-build)    BUILD_RELEASE=0;    shift ;;
    --no-gguf)     NO_GGUF=1;          shift ;;
    --vm)          VM_NAME="$2";       shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "error: --model is required" >&2
  usage
  exit 1
fi

MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  exit 1
fi

# Resolve model path relative to the runtime dir so we can form the VM path.
if [[ "$MODEL_PATH" != "$RUNTIME_DIR"/* ]]; then
  echo "error: --model must be inside the kapsl-runtime directory ($RUNTIME_DIR)" >&2
  echo "       got: $MODEL_PATH" >&2
  exit 1
fi
MODEL_RELATIVE="${MODEL_PATH#$RUNTIME_DIR/}"
VM_MODEL_PATH="$VM_RUNTIME_DIR/$MODEL_RELATIVE"

# ── Helpers ───────────────────────────────────────────────────────────────────

step() { echo; echo "━━━━━━━━  $*  ━━━━━━━━"; }
ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }
die()  { echo "error: $*" >&2; exit 1; }

vm()    { multipass exec "$VM_NAME" -- bash -c "$1"; }
vm_bg() { multipass exec "$VM_NAME" -- bash -c "$1" & }

wait_http_vm() {
  local url="$1" label="$2" tries="${3:-120}"
  echo -n "  Waiting for $label "
  for _ in $(seq 1 "$tries"); do
    code="$(vm "curl -s -o /dev/null -w '%{http_code}' '$url' 2>/dev/null || true")"
    if [[ "$code" == "200" ]]; then echo "✓"; return 0; fi
    echo -n "."
    sleep 2
  done
  echo " ✗ (timed out)"
  return 1
}

# ── 1. VM state ────────────────────────────────────────────────────────────────

step "Checking VM: $VM_NAME"
VM_STATE="$(multipass info "$VM_NAME" 2>/dev/null | grep '^State:' | awk '{print $2}' || echo "missing")"
if [[ "$VM_STATE" == "missing" ]]; then
  die "VM '$VM_NAME' does not exist. Create it with:\n  multipass launch 24.04 --name $VM_NAME --cpus 4 --memory 8G --disk 50G"
fi
if [[ "$VM_STATE" != "Running" ]]; then
  echo "  VM is $VM_STATE — starting..."
  multipass start "$VM_NAME"
  sleep 3
fi
ok "VM is running"
multipass info "$VM_NAME" | grep -E 'IPv4:|CPUs:|Memory|Disk'

# ── 2. Mounts ─────────────────────────────────────────────────────────────────

step "Ensuring directory mounts"

ensure_mount() {
  local host_path="$1" vm_path="$2"
  # Check if already mounted by testing if a known file exists inside the VM.
  local test_file
  test_file="$(basename "$host_path")"
  if vm "ls '$vm_path' >/dev/null 2>&1"; then
    ok "already mounted: $vm_path"
  else
    echo "  Mounting $host_path → $VM_NAME:$vm_path"
    multipass mount "$host_path" "$VM_NAME:$vm_path"
    ok "mounted: $vm_path"
  fi
}

ensure_mount "$SCRIPT_DIR"   "$VM_BENCH_DIR"
ensure_mount "$RUNTIME_DIR"  "$VM_RUNTIME_DIR"

# ── 3. System dependencies ────────────────────────────────────────────────────

step "Installing system dependencies"
vm "
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libssl-dev git curl \
    python3-pip python3-venv python3-dev pciutils \
    >/dev/null 2>&1
  echo 'apt done'
"
ok "apt packages installed"

# ── 4. Rust toolchain ─────────────────────────────────────────────────────────

step "Rust toolchain"
vm "
  if ! command -v rustup &>/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
  fi
  source \$HOME/.cargo/env
  rustup toolchain install stable --no-self-update >/dev/null 2>&1
  rustup default stable >/dev/null 2>&1
  cargo --version
"

# ── 5. Python venv for bench scripts ─────────────────────────────────────────

step "Python venv"
vm "
  cd '$VM_BENCH_DIR'
  if [[ ! -f .venv/bin/python ]]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet requests
  echo 'venv ready'
"
ok "python venv ready at $VM_BENCH_DIR/.venv"

# ── 6. Build kapsl ────────────────────────────────────────────────────────────

if [[ "$BUILD_RELEASE" -eq 1 ]]; then
  step "Building kapsl release binary (first run: 10–25 min)"
  FEATURES_FLAG=""
  if [[ "$NO_GGUF" -eq 1 ]]; then
    FEATURES_FLAG="--no-default-features"
    echo "  (--no-gguf: skipping llama.cpp)"
  fi
  vm "
    source \$HOME/.cargo/env
    cd '$VM_RUNTIME_DIR'
    cargo build -p kapsl --release $FEATURES_FLAG 2>&1 | tail -5
    echo 'build done'
  "
fi

vm "ls -lh '$VM_RUNTIME_DIR/target/release/kapsl' 2>/dev/null || (echo 'binary missing!' && exit 1)"
ok "kapsl binary present"

# ── 7. Optional vLLM ─────────────────────────────────────────────────────────

if [[ -n "$VLLM_MODEL" ]]; then
  step "Setting up vLLM (CPU) in VM"
  vm "
    if ! python3 -c 'import vllm' 2>/dev/null; then
      pip3 install --quiet vllm 2>&1 | tail -3
    fi
    python3 -c 'import vllm; print(\"vLLM\", vllm.__version__)'
  "

  step "Starting vLLM server in VM"
  vm "pkill -f 'vllm.entrypoints' || true"
  sleep 2
  vm_bg "
    nohup python3 -m vllm.entrypoints.openai.api_server \
      --model '$VLLM_MODEL' \
      --device cpu \
      --dtype bfloat16 \
      --max-model-len 2048 \
      --max-num-seqs 4 \
      --port 8000 \
      >/tmp/vllm.log 2>&1
  "
  VLLM_START=1

  wait_http_vm "http://127.0.0.1:8000/v1/models" "vLLM" 180 || {
    echo "vLLM log tail:"
    vm "tail -20 /tmp/vllm.log" || true
    die "vLLM failed to start"
  }
fi

# ── 8. Run benchmark via run_linux_bench.sh ───────────────────────────────────

TS="$(date +%Y%m%d-%H%M%S)"
VM_RESULT="$VM_BENCH_DIR/results-multipass-${TS}.json"
LOCAL_RESULT="$SCRIPT_DIR/results-multipass-${TS}.json"

step "Running benchmark in VM"
echo "  model:       $VM_MODEL_PATH"
echo "  requests:    $REQUESTS per concurrency level"
echo "  concurrency: $CONCURRENCY"
echo "  max_tokens:  $MAX_TOKENS"
echo "  output:      $VM_RESULT"

BENCH_FLAGS=(
  "--model"       "$VM_MODEL_PATH"
  "--requests"    "$REQUESTS"
  "--concurrency" "$CONCURRENCY"
  "--max-tokens"  "$MAX_TOKENS"
  "--temperature" "$TEMPERATURE"
  "--warmup"      "$WARMUP"
  "--timeout"     "$TIMEOUT"
  "--batch-size"  "$BATCH_SIZE"
  "--output"      "$VM_RESULT"
  "--no-build"   # already built above
)

if [[ -n "$VLLM_MODEL" ]]; then
  BENCH_FLAGS+=("--vllm-model" "$VLLM_MODEL")
fi

BENCH_FLAGS_STR="${BENCH_FLAGS[*]}"

vm "
  source \$HOME/.cargo/env
  export PATH=\"$VM_BENCH_DIR/.venv/bin:\$PATH\"
  bash '$VM_BENCH_DIR/run_linux_bench.sh' $BENCH_FLAGS_STR
"

# ── 9. Copy results to host ───────────────────────────────────────────────────

step "Copying results to host"
multipass transfer "$VM_NAME:$VM_RESULT" "$LOCAL_RESULT"
ok "results saved: $LOCAL_RESULT"

# ── 10. Stop vLLM if we started it ───────────────────────────────────────────

if [[ "$VLLM_START" -eq 1 ]]; then
  vm "pkill -f 'vllm.entrypoints' || true"
fi

# ── 11. Print summary from host ───────────────────────────────────────────────

python3 - "$LOCAL_RESULT" <<'PYEOF'
import json, sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

si = data.get("linux_sysinfo", {})
print()
print("=" * 68)
print("MULTIPASS BENCHMARK SUMMARY")
print("=" * 68)
if si:
    print(f"  VM CPU:     {si.get('cpu_model', 'unknown')}")
    print(f"  VM Cores:   {si.get('cpu_logical')} logical / {si.get('cpu_sockets')} socket(s)")
    print(f"  VM RAM:     {si.get('ram_total_gib')} GiB total")
    print(f"  Governor:   {si.get('cpu_governor')}")
    print(f"  Thermal:    {si.get('thermal_before')} → {si.get('thermal_after')}")
    print(f"  Kernel:     {si.get('kernel')}")
    print()

def fmt_table(label, results_by_conc):
    print(f"  [{label}]")
    print(f"  {'conc':>6}  {'req/s':>8}  {'p50 ms':>8}  {'p90 ms':>8}  {'p99 ms':>8}  {'tok/s':>8}  {'errs':>5}")
    print("  " + "-" * 60)
    for key in sorted(results_by_conc.keys(), key=lambda k: int(k.lstrip("c"))):
        r = results_by_conc[key]
        lat = r.get("latency_ms") or {}
        print(
            f"  {r.get('concurrency', '?'):>6}"
            f"  {r.get('throughput_rps', 0):>8.2f}"
            f"  {lat.get('p50', 0):>8.0f}"
            f"  {lat.get('p90', 0):>8.0f}"
            f"  {lat.get('p99', 0):>8.0f}"
            f"  {r.get('overall_tok_per_sec', 0):>8.1f}"
            f"  {r.get('errors', 0):>5}"
        )
    print()

if data.get("kapsl"):
    fmt_table("kapsl", data["kapsl"])
if data.get("vllm"):
    fmt_table("vLLM", data["vllm"])

if data.get("kapsl") and data.get("vllm"):
    print("  [kapsl / vLLM speedup]")
    print(f"  {'conc':>6}  {'tp ratio':>10}  {'p50 ratio':>10}  {'tok/s ratio':>12}")
    print("  " + "-" * 44)
    for key in sorted(data["kapsl"].keys(), key=lambda k: int(k.lstrip("c"))):
        if key not in data["vllm"]:
            continue
        k = data["kapsl"][key]
        v = data["vllm"][key]
        tp  = k["throughput_rps"] / v["throughput_rps"] if v.get("throughput_rps") else 0
        lat = k["latency_ms"].get("p50", 0) / v["latency_ms"]["p50"] if (v.get("latency_ms") or {}).get("p50") else 0
        tok = k["overall_tok_per_sec"] / v["overall_tok_per_sec"] if v.get("overall_tok_per_sec") else 0
        print(f"  {k.get('concurrency', '?'):>6}  {tp:>10.3f}x  {lat:>10.3f}x  {tok:>12.3f}x")
    print()

print(f"  Full results: {path}")
print("=" * 68)
PYEOF
