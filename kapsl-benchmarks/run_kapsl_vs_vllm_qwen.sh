#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/kapsl-benchmarks"
RUNTIME_BIN="$ROOT_DIR/kapsl-runtime/target/release/kapsl"
MODEL_PATH="$ROOT_DIR/kapsl-runtime/models/qwen/qwen2.5-1.5b-instruct.aimod"
PYTHON_BIN_DEFAULT="$BENCH_DIR/.venv/bin/python"

KAPSL_URL="${KAPSL_URL:-http://127.0.0.1:9195}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
KAPSL_MODEL_ID="${KAPSL_MODEL_ID:-0}"
REQUESTS="${REQUESTS:-20}"
CONCURRENCY="${CONCURRENCY:-4}"
WARMUP="${WARMUP:-1}"
TIMEOUT="${TIMEOUT:-240}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0}"
PROMPT="${PROMPT:-Summarize the system metrics.}"

BUILD_RELEASE=1
KEEP_KAPSL_RUNNING=0
OUTPUT=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --output PATH            Output JSON file (default: timestamped file in kapsl-benchmarks/)
  --requests N             Total requests per backend (default: ${REQUESTS})
  --concurrency N          Concurrency level (default: ${CONCURRENCY})
  --max-tokens N           Max new tokens (default: ${MAX_TOKENS})
  --temperature X          Sampling temperature (default: ${TEMPERATURE})
  --prompt TEXT            Prompt text
  --no-build               Skip 'cargo build -p kapsl --release'
  --keep-kapsl-running     Do not stop Kapsl at script exit
  -h, --help               Show this help

Environment overrides:
  KAPSL_URL, VLLM_URL, VLLM_MODEL, KAPSL_MODEL_ID,
  REQUESTS, CONCURRENCY, WARMUP, TIMEOUT, MAX_TOKENS, TEMPERATURE, PROMPT
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --requests)
      REQUESTS="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --no-build)
      BUILD_RELEASE=0
      shift
      ;;
    --keep-kapsl-running)
      KEEP_KAPSL_RUNNING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN_DEFAULT" ]]; then
  echo "Missing Python venv binary: $PYTHON_BIN_DEFAULT" >&2
  exit 1
fi
PYTHON_BIN="$PYTHON_BIN_DEFAULT"

if [[ ! -x "$RUNTIME_BIN" ]]; then
  echo "Missing kapsl runtime binary: $RUNTIME_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model package: $MODEL_PATH" >&2
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  OUTPUT="$BENCH_DIR/results-kapsl-vs-vllm-qwen-batched-decode-${ts}.json"
fi

if ! curl -s -o /dev/null -w '%{http_code}' "$VLLM_URL/v1/models" | grep -q '^200$'; then
  echo "vLLM is not reachable at $VLLM_URL/v1/models" >&2
  exit 1
fi

if [[ "$BUILD_RELEASE" -eq 1 ]]; then
  echo "[1/4] Building release binary..."
  (cd "$ROOT_DIR/kapsl-runtime" && cargo build -p kapsl --release)
fi

echo "[2/4] Starting kapsl runtime..."
KAPSL_PROVIDER_POLICY=manifest KAPSL_LLM_SAFE_LOAD=0 KAPSL_LLM_ISOLATE_PROCESS=0 \
  "$RUNTIME_BIN" run \
    --model "$MODEL_PATH" \
    --socket /tmp/kapsl-qwen.sock \
    --metrics-port 9195 \
    --http-bind 127.0.0.1 \
    --batch-size 8 \
    --scheduler-max-micro-batch 8 \
    --scheduler-queue-delay-ms 1 \
    >/tmp/kapsl-benchmark-run.log 2>&1 &
KAPSL_PID=$!

cleanup() {
  if [[ "$KEEP_KAPSL_RUNNING" -eq 0 ]]; then
    kill "$KAPSL_PID" >/dev/null 2>&1 || true
    wait "$KAPSL_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "$KAPSL_URL/api/models" || true)"
  if [[ "$code" == "200" ]]; then
    break
  fi
  sleep 1
done

if ! curl -s -o /dev/null -w '%{http_code}' "$KAPSL_URL/api/models" | grep -q '^200$'; then
  echo "kapsl did not become ready at $KAPSL_URL/api/models" >&2
  exit 1
fi

echo "[3/4] Running benchmark..."
"$PYTHON_BIN" "$BENCH_DIR/bench.py" \
  --kapsl-url "$KAPSL_URL" --kapsl-model-id "$KAPSL_MODEL_ID" \
  --vllm-url "$VLLM_URL" --vllm-model "$VLLM_MODEL" \
  --requests "$REQUESTS" --concurrency "$CONCURRENCY" --warmup "$WARMUP" --timeout "$TIMEOUT" \
  --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --prompt "$PROMPT" \
  --output "$OUTPUT"

echo "[4/4] Summary"
"$PYTHON_BIN" - <<PY
import json
j=json.load(open("$OUTPUT"))
km=j['kapsl']['models'][str($KAPSL_MODEL_ID)]['benchmark']
vm=j['vllm']['models']["$VLLM_MODEL"]['benchmark']
print("output", "$OUTPUT")
print("kapsl", km['throughput_rps'], km['latency_ms']['p50'], km['latency_ms']['p90'], km['errors'])
print("vllm", vm['throughput_rps'], vm['latency_ms']['p50'], vm['latency_ms']['p90'], vm['errors'])
if vm['throughput_rps']:
    print("tp_ratio_kapsl_over_vllm", km['throughput_rps']/vm['throughput_rps'])
PY
