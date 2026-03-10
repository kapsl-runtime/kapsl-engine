#!/usr/bin/env bash
# run_bench_multipass.sh — provision the kapsl-bench Multipass VM and run
# a kapsl-vs-vLLM benchmark against Qwen 2.5 1.5B.
#
# Usage (from macOS host):
#   ./kapsl-benchmarks/run_bench_multipass.sh [--no-gguf] [--no-build] [--no-vllm-start] \
#       [--requests N] [--concurrency N] [--max-tokens N]
#
# The VM must already exist:
#   multipass launch 24.04 --name kapsl-bench --cpus 4 --memory 8G --disk 50G
# Both directories must already be mounted:
#   multipass mount /path/to/kapsl/engine/kapsl-benchmarks kapsl-bench:/home/ubuntu/kapsl-benchmarks
#   multipass mount /path/to/kapsl/engine/kapsl-runtime    kapsl-bench:/home/ubuntu/kapsl-runtime

set -euo pipefail

VM_NAME="kapsl-bench"
REQUESTS="${REQUESTS:-20}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
WARMUP="${WARMUP:-1}"
TIMEOUT="${TIMEOUT:-300}"
PROMPT="${PROMPT:-Summarize the system metrics.}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

NO_GGUF=0
NO_BUILD=0
NO_VLLM_START=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gguf)         NO_GGUF=1;         shift ;;
    --no-build)        NO_BUILD=1;        shift ;;
    --no-vllm-start)   NO_VLLM_START=1;   shift ;;
    --requests)        REQUESTS="$2";     shift 2 ;;
    --concurrency)     CONCURRENCY="$2";  shift 2 ;;
    --max-tokens)      MAX_TOKENS="$2";   shift 2 ;;
    --temperature)     TEMPERATURE="$2";  shift 2 ;;
    --prompt)          PROMPT="$2";       shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

run() { multipass exec "$VM_NAME" -- bash -c "$1"; }
run_bg() { multipass exec "$VM_NAME" -- bash -c "$1" &; }

# ── Helpers ────────────────────────────────────────────────────────────────

step() { echo; echo "━━━ $* ━━━"; }

wait_http() {
  local url="$1" label="$2" tries="${3:-120}"
  echo -n "Waiting for $label"
  for _ in $(seq 1 "$tries"); do
    code="$(run "curl -s -o /dev/null -w '%{http_code}' '$url' || true")"
    if [[ "$code" == "200" ]]; then echo " ✓"; return 0; fi
    echo -n "."
    sleep 2
  done
  echo " ✗ (timed out)"
  return 1
}

# ── 0. Sanity ─────────────────────────────────────────────────────────────

step "Checking VM state"
multipass info "$VM_NAME" | grep -E "State:|IPv4:"
run "ls /home/ubuntu/kapsl-runtime /home/ubuntu/kapsl-benchmarks >/dev/null && echo 'mounts OK'"

# ── 1. System dependencies ────────────────────────────────────────────────

step "Installing system dependencies"
run "
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libssl-dev git curl \
    python3-pip python3-venv python3-dev \
    >/dev/null 2>&1
  echo 'apt done'
"

# ── 2. Rust toolchain ─────────────────────────────────────────────────────

step "Installing Rust"
run "
  if ! command -v rustup &>/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
  fi
  source \$HOME/.cargo/env
  rustup toolchain install stable --no-self-update
  rustup default stable
  cargo --version
"

# ── 3. Build kapsl release ────────────────────────────────────────────────

if [[ "$NO_BUILD" -eq 0 ]]; then
  step "Building kapsl release binary (this takes ~10–25 min on first run)"
  FEATURES_FLAG=""
  if [[ "$NO_GGUF" -eq 1 ]]; then
    # Disable gguf to skip llama.cpp compilation — saves ~5 min and ~2GB
    FEATURES_FLAG="--no-default-features"
    echo "(--no-gguf: skipping llama.cpp build)"
  fi
  run "
    source \$HOME/.cargo/env
    cd /home/ubuntu/kapsl-runtime
    cargo build -p kapsl --release $FEATURES_FLAG 2>&1 | tail -5
    echo 'build done'
  "
else
  step "Skipping build (--no-build)"
fi

run "ls -lh /home/ubuntu/kapsl-runtime/target/release/kapsl 2>/dev/null || (echo 'binary missing!' && exit 1)"

# ── 4. Python venv for bench.py ───────────────────────────────────────────

step "Setting up Python venv for bench.py"
run "
  cd /home/ubuntu/kapsl-benchmarks
  if [[ ! -f .venv/bin/python ]]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet requests tqdm
  echo 'bench.py deps installed'
"

# ── 5. vLLM install ───────────────────────────────────────────────────────

step "Installing vLLM (CPU)"
run "
  if ! python3 -c 'import vllm' 2>/dev/null; then
    pip3 install --quiet vllm 2>&1 | tail -3
    echo 'vLLM installed'
  else
    python3 -c 'import vllm; print(\"vLLM\", vllm.__version__, \"already installed\")'
  fi
"

# ── 6. Start vLLM ─────────────────────────────────────────────────────────

if [[ "$NO_VLLM_START" -eq 0 ]]; then
  step "Starting vLLM server (CPU, Qwen 2.5 1.5B)"
  # Kill any existing vLLM process
  run "pkill -f 'vllm.entrypoints' || true"
  sleep 2

  run_bg "
    nohup python3 -m vllm.entrypoints.openai.api_server \
      --model '$VLLM_MODEL' \
      --device cpu \
      --dtype bfloat16 \
      --max-model-len 2048 \
      --max-num-seqs 4 \
      --port 8000 \
      >/tmp/vllm.log 2>&1
  " &

  wait_http "http://127.0.0.1:8000/v1/models" "vLLM" 180 || {
    echo "vLLM log tail:"
    run "tail -20 /tmp/vllm.log"
    exit 1
  }
else
  step "Skipping vLLM start (--no-vllm-start)"
  wait_http "http://127.0.0.1:8000/v1/models" "vLLM (existing)" 10
fi

# ── 7. Run benchmark ──────────────────────────────────────────────────────

TS="$(date +%Y%m%d-%H%M%S)"
RESULT_FILE="/home/ubuntu/kapsl-benchmarks/results-multipass-${TS}.json"
LOCAL_RESULT="$(dirname "$0")/results-multipass-${TS}.json"

step "Running benchmark (requests=$REQUESTS concurrency=$CONCURRENCY max_tokens=$MAX_TOKENS)"
run "
  set -euo pipefail
  source \$HOME/.cargo/env
  cd /home/ubuntu

  MODEL_PATH=/home/ubuntu/kapsl-runtime/models/qwen/qwen2.5-1.5b-instruct.aimod
  RUNTIME_BIN=/home/ubuntu/kapsl-runtime/target/release/kapsl

  # Kill any leftover kapsl
  pkill -f '\$RUNTIME_BIN' || true
  sleep 1

  # Start kapsl
  KAPSL_PROVIDER_POLICY=manifest KAPSL_LLM_SAFE_LOAD=0 KAPSL_LLM_ISOLATE_PROCESS=0 \
    \$RUNTIME_BIN run \
      --model \$MODEL_PATH \
      --socket /tmp/kapsl-bench.sock \
      --metrics-port 9195 \
      --http-bind 127.0.0.1 \
      --batch-size 8 \
      --scheduler-max-micro-batch 8 \
      --scheduler-queue-delay-ms 1 \
      >/tmp/kapsl-bench.log 2>&1 &
  KAPSL_PID=\$!
  trap 'kill \$KAPSL_PID 2>/dev/null || true' EXIT

  # Wait for kapsl
  echo -n 'Waiting for kapsl'
  for _ in \$(seq 1 120); do
    code=\"\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9195/api/models || true)\"
    if [[ \"\$code\" == '200' ]]; then echo ' ✓'; break; fi
    echo -n '.'
    sleep 1
  done
  [[ \"\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9195/api/models)\" == '200' ]] || {
    echo 'kapsl failed to start'
    tail -20 /tmp/kapsl-bench.log
    exit 1
  }

  # Run bench.py
  /home/ubuntu/kapsl-benchmarks/.venv/bin/python /home/ubuntu/kapsl-benchmarks/bench.py \
    --kapsl-url http://127.0.0.1:9195 --kapsl-model-id 0 \
    --vllm-url http://127.0.0.1:8000 --vllm-model '$VLLM_MODEL' \
    --requests $REQUESTS --concurrency $CONCURRENCY \
    --warmup $WARMUP --timeout $TIMEOUT \
    --max-tokens $MAX_TOKENS --temperature $TEMPERATURE \
    --prompt '$PROMPT' \
    --output '$RESULT_FILE'
"

# ── 8. Print summary + copy results ───────────────────────────────────────

step "Summary"
run "python3 - <<'PY'
import json
j = json.load(open('$RESULT_FILE'))
km = j['kapsl']['models']['0']['benchmark']
vm = j['vllm']['models']['$VLLM_MODEL']['benchmark']
print(f\"{'':20s} {'kapsl':>12s}  {'vLLM':>12s}\")
print(f\"{'throughput (req/s)':20s} {km['throughput_rps']:>12.3f}  {vm['throughput_rps']:>12.3f}\")
print(f\"{'p50 latency (ms)':20s} {km['latency_ms']['p50']:>12.1f}  {vm['latency_ms']['p50']:>12.1f}\")
print(f\"{'p90 latency (ms)':20s} {km['latency_ms']['p90']:>12.1f}  {vm['latency_ms']['p90']:>12.1f}\")
print(f\"{'errors':20s} {km['errors']:>12d}  {vm['errors']:>12d}\")
if vm['throughput_rps']:
    print(f\"\\nkapsl/vLLM throughput ratio: {km['throughput_rps']/vm['throughput_rps']:.3f}x\")
PY
"

# Copy results to host
multipass transfer "$VM_NAME:$RESULT_FILE" "$LOCAL_RESULT"
echo
echo "Results saved to: $LOCAL_RESULT"
