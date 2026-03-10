# Kapsl Benchmarks

Benchmark harness for comparing kapsl-runtime against vLLM using consistent load profiles. It measures request latency, throughput, error rates, and memory-related estimates.

## What it does

- Sends concurrent requests to kapsl-runtime and/or vLLM.
- Computes latency percentiles (p50/p90/p99), throughput, and error counts.
- Captures memory usage snapshots and reports a fragmentation estimate/proxy.

## Requirements

- Python 3.8+
- `requests` (optional). If unavailable, the script falls back to Python's stdlib HTTP client.

## Quick start

### Kapsl only

```bash
python3 bench.py \
  --kapsl-model-id 0 \
  --requests 50 \
  --concurrency 4 \
  --prompt "Summarize the system metrics."
```

Multiple kapsl models in one run:

```bash
python3 bench.py \
  --kapsl-model-ids 0,2,5 \
  --requests 50 \
  --concurrency 4 \
  --prompt "Summarize the system metrics."
```

### vLLM only (OpenAI-compatible chat)

```bash
python3 bench.py \
  --vllm-model YOUR_MODEL_NAME \
  --requests 50 \
  --concurrency 4 \
  --prompt "Summarize the system metrics."
```

Multiple vLLM models (chat/completions endpoints):

```bash
python3 bench.py \
  --vllm-models model-a,model-b \
  --requests 50 \
  --concurrency 4 \
  --prompt "Summarize the system metrics."
```

### Compare both in one run

```bash
python3 bench.py \
  --kapsl-model-ids 0,2 \
  --vllm-models model-a,model-b \
  --requests 100 \
  --concurrency 8 \
  --prompt "Summarize the system metrics." \
  --output results.json
```

## Notes on memory fragmentation

- kapsl-runtime: `/api/models[*].memory_usage` is engine-reported model memory (may be `0` if the backend does not expose it). Runtime RSS/GPU memory is available via `GET /api/system/stats`. Any "fragmentation" computed as `1 - (max_model_memory / total_model_memory)` is a proxy and not true allocator fragmentation.
- vLLM: fragmentation is a proxy derived from `/metrics` if GPU memory totals are available. Override metric names with:
  - `--vllm-mem-used-metric`
  - `--vllm-mem-total-metric`

## Payloads

You can override the default payloads with JSON files:

```bash
python3 bench.py --kapsl-model-id 0 --kapsl-payload-file payloads/kapsl_string.json
python3 bench.py --vllm-model YOUR_MODEL_NAME --vllm-payload-file payloads/vllm_chat.json
```

See sample payloads in `payloads/`.

## Reproducible Kapsl vs vLLM Run (Qwen 1.5B)

One-command script for the exact profile used in our comparisons:

```bash
./run_kapsl_vs_vllm_qwen.sh
```

Script behavior:
- Verifies vLLM is reachable at `http://127.0.0.1:8000/v1/models`.
- Builds `kapsl` release binary by default.
- Starts kapsl runtime on `http://127.0.0.1:9195` with tuned launch flags.
- Runs `bench.py` with fixed defaults (`requests=20`, `concurrency=4`, `max_tokens=64`, `temperature=0`).
- Saves a timestamped JSON under `kapsl-benchmarks/`.

Useful options:

```bash
./run_kapsl_vs_vllm_qwen.sh --no-build
./run_kapsl_vs_vllm_qwen.sh --requests 50 --concurrency 8
./run_kapsl_vs_vllm_qwen.sh --output results-custom.json
```
