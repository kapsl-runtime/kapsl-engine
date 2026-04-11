#!/usr/bin/env bash
# run_benchmark.sh — set up and run the kapsl benchmark on a Lambda Labs instance.
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh                         # LLM benchmark, auto-detect model
#   ./run_benchmark.sh --mode tensor           # ONNX tensor benchmark
#   ./run_benchmark.sh --concurrency 1,4,8,16 --requests 200
#   ./run_benchmark.sh --output /tmp/results.json
#
# All extra arguments are forwarded to benchmark.py.
#
# Environment:
#   KAPSL_API_TOKEN     — bearer token for authenticated runtimes
#   KAPSL_BASE_URL      — runtime base URL (default: http://127.0.0.1:9095)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="${SCRIPT_DIR}/benchmark.py"
BASE_URL="${KAPSL_BASE_URL:-http://127.0.0.1:9095}"
OUTPUT="${BENCHMARK_OUTPUT:-/tmp/kapsl_benchmark_$(date +%Y%m%d_%H%M%S).json}"

echo "========================================"
echo "  Kapsl Runtime Benchmark Setup"
echo "========================================"

# ---- Python ------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install it first." >&2
    exit 1
fi
PYTHON=python3
echo "Python: $($PYTHON --version)"

# ---- pip deps (zero-dep script but keep requests/numpy available for other tests)
if ! $PYTHON -c "import urllib.request" 2>/dev/null; then
    echo "ERROR: urllib not available — broken Python install?" >&2
    exit 1
fi

# ---- nvidia-smi --------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "GPU(s) detected:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader,nounits | while IFS=, read -r idx name mem drv; do
        echo "  GPU $idx: $name  ${mem} MB  driver ${drv}"
    done
else
    echo "nvidia-smi not found — GPU stats will be skipped."
fi

# ---- runtime health check ----------------------------------------------------
echo ""
echo "Checking runtime at ${BASE_URL}..."
for i in 1 2 3 4 5; do
    if curl -sf "${BASE_URL}/api/health" -o /dev/null 2>&1; then
        echo "Runtime is up."
        break
    fi
    if [[ $i -eq 5 ]]; then
        echo "ERROR: runtime not reachable at ${BASE_URL}/api/health after 5 attempts." >&2
        echo "Make sure kapsl is running: kapsl run --model <path>" >&2
        exit 1
    fi
    echo "  Waiting for runtime... (attempt $i/5)"
    sleep 3
done

# ---- list loaded models ------------------------------------------------------
echo ""
echo "Loaded models:"
AUTH_HEADER=""
if [[ -n "${KAPSL_API_TOKEN:-}" ]]; then
    AUTH_HEADER="-H \"Authorization: Bearer ${KAPSL_API_TOKEN}\""
fi
curl -sf "${BASE_URL}/api/models" \
    ${KAPSL_API_TOKEN:+-H "Authorization: Bearer ${KAPSL_API_TOKEN}"} \
    | $PYTHON -c "
import json, sys
models = json.load(sys.stdin)
if not models:
    print('  (none loaded)')
else:
    for m in models:
        print(f'  id={m.get(\"id\")}  name={m.get(\"name\",\"?\")}  status={m.get(\"status\",\"?\")}')
" 2>/dev/null || echo "  (could not parse model list)"

# ---- run benchmark -----------------------------------------------------------
echo ""
echo "========================================"
echo "  Starting benchmark"
echo "========================================"
echo "  Output: ${OUTPUT}"
echo ""

$PYTHON "${BENCH_PY}" \
    --base-url "${BASE_URL}" \
    --output "${OUTPUT}" \
    "$@"

echo ""
echo "========================================"
echo "  Done. Results saved to: ${OUTPUT}"
echo "========================================"

# Pretty-print peak numbers from JSON if python/jq available
if $PYTHON -c "import json" 2>/dev/null && [[ -f "${OUTPUT}" ]]; then
    echo ""
    $PYTHON - <<'PYEOF'
import json, sys, os

path = os.environ.get("BENCHMARK_OUTPUT", "")
# find the most recent benchmark file in /tmp if path not set
if not path or not os.path.exists(path):
    import glob
    files = sorted(glob.glob("/tmp/kapsl_benchmark_*.json"))
    path = files[-1] if files else ""

if not path or not os.path.exists(path):
    sys.exit(0)

with open(path) as f:
    data = json.load(f)

results = data.get("results", [])
if not results:
    sys.exit(0)

best_rps = max(results, key=lambda r: r["throughput_rps"])
print(f"Peak throughput  : {best_rps['throughput_rps']:.2f} req/s  (concurrency={best_rps['concurrency']})")
print(f"  p50={best_rps['latency_p50_ms']:.1f}ms  p95={best_rps['latency_p95_ms']:.1f}ms  p99={best_rps['latency_p99_ms']:.1f}ms")

best_tok = max(results, key=lambda r: r.get("tokens_per_sec", 0))
if best_tok.get("tokens_per_sec", 0) > 0:
    print(f"Peak tokens/s    : {best_tok['tokens_per_sec']:.1f}  (concurrency={best_tok['concurrency']})")
PYEOF
fi
