#!/usr/bin/env python3
"""
LLM Inference Benchmark for kapsl-runtime
Tests latency, TTFT, throughput, and concurrent load against a running
kapsl server loaded with a GGUF/pipeline model (e.g. Qwen).

Usage:
  # Start the server first:
  #   cargo run --release --bin kapsl -- run \
  #     --model /path/to/qwen2.5-1.5b-instruct.aimod --transport http
  #
  # Then run this script:
  #   python3 benchmarks/llm_bench.py [--host 127.0.0.1] [--port 9095] [--model-id qwen2.5-1.5b-instruct]
"""

import argparse
import json
import time
import statistics
import concurrent.futures
import urllib.request
import urllib.error
import sys

# ---------------------------------------------------------------------------
# Prompts used during the benchmark
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a haiku about machine learning.",
    "What are three tips for improving sleep quality?",
    "Summarise the plot of Romeo and Juliet in two sentences.",
    "What is 17 multiplied by 23?",
    "Name five programming languages and their primary use cases.",
    "Describe how a transformer neural network works at a high level.",
]


def build_request_body(prompt: str, max_new_tokens: int = 128) -> bytes:
    """Encode a text prompt as a kapsl InferenceRequest JSON payload."""
    payload = {
        "input": {
            # shape: [num_bytes] — 1-D byte sequence for text
            "shape": [len(prompt.encode())],
            "dtype": "string",
            "data": list(prompt.encode()),
        },
        "metadata": {
            "max_new_tokens": max_new_tokens,
        },
    }
    return json.dumps(payload).encode()


def infer(host: str, port: int, model_id: int, prompt: str, max_new_tokens: int = 128) -> dict:
    """
    POST a single inference request and return timing + response metadata.
    Returns a dict with keys: latency_ms, output_tokens (approx), error.
    """
    url = f"http://{host}:{port}/api/models/{model_id}/infer"
    body = build_request_body(prompt, max_new_tokens)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
            latency_ms = (time.perf_counter() - t0) * 1000
            try:
                data = json.loads(raw)
                # Response is a BinaryTensorPacket; data field holds UTF-8 bytes
                output_bytes = bytes(data.get("data", []))
                output_text = output_bytes.decode("utf-8", errors="replace")
                output_tokens = max(1, len(output_text.split()))
            except Exception:
                output_text = raw.decode("utf-8", errors="replace")[:120]
                output_tokens = 1
            return {
                "latency_ms": latency_ms,
                "output_tokens": output_tokens,
                "output_preview": output_text[:80].replace("\n", " "),
                "error": None,
            }
    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        body_err = e.read().decode("utf-8", errors="replace")[:120]
        return {"latency_ms": latency_ms, "output_tokens": 0, "output_preview": "", "error": f"HTTP {e.code}: {body_err}"}
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"latency_ms": latency_ms, "output_tokens": 0, "output_preview": "", "error": str(e)}


def check_server(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/models", timeout=5):
            return True
    except Exception:
        return False


def run_sequential(host, port, model_id, prompts, label="Sequential", max_new_tokens=128):
    print(f"\n{'='*64}")
    print(f"  {label}  ({len(prompts)} requests, 1 at a time)")
    print(f"{'='*64}")
    latencies = []
    token_counts = []
    errors = 0

    for i, prompt in enumerate(prompts):
        result = infer(host, port, model_id, prompt, max_new_tokens)
        if result["error"]:
            print(f"  [{i+1:02d}] ERROR: {result['error']}")
            errors += 1
        else:
            latencies.append(result["latency_ms"])
            token_counts.append(result["output_tokens"])
            print(
                f"  [{i+1:02d}] {result['latency_ms']:7.1f} ms  "
                f"~{result['output_tokens']:3d} tok  "
                f"\"{result['output_preview']}…\""
            )

    if latencies:
        total_tokens = sum(token_counts)
        total_sec = sum(latencies) / 1000
        print(f"\n  Requests  : {len(latencies)} ok, {errors} failed")
        print(f"  Latency   : avg {statistics.mean(latencies):.1f} ms  "
              f"p50 {statistics.median(latencies):.1f} ms  "
              f"p95 {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms  "
              f"min {min(latencies):.1f} ms  max {max(latencies):.1f} ms")
        print(f"  Throughput: {len(latencies)/total_sec:.2f} req/s  "
              f"~{total_tokens/total_sec:.1f} tok/s")
    return latencies


def run_concurrent(host, port, model_id, prompts, concurrency, label=None, max_new_tokens=128):
    label = label or f"Concurrent (c={concurrency})"
    print(f"\n{'='*64}")
    print(f"  {label}  ({len(prompts)} requests, {concurrency} parallel)")
    print(f"{'='*64}")
    latencies = []
    errors = 0
    t_wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(infer, host, port, model_id, p, max_new_tokens)
            for p in prompts
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            result = fut.result()
            if result["error"]:
                print(f"  [{i+1:02d}] ERROR: {result['error']}")
                errors += 1
            else:
                latencies.append(result["latency_ms"])
                print(
                    f"  [{i+1:02d}] {result['latency_ms']:7.1f} ms  "
                    f"~{result['output_tokens']:3d} tok"
                )

    wall_ms = (time.perf_counter() - t_wall_start) * 1000
    if latencies:
        print(f"\n  Requests   : {len(latencies)} ok, {errors} failed")
        print(f"  Wall time  : {wall_ms:.0f} ms")
        print(f"  Latency    : avg {statistics.mean(latencies):.1f} ms  "
              f"p50 {statistics.median(latencies):.1f} ms  "
              f"p95 {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms")
        print(f"  Throughput : {len(latencies)/(wall_ms/1000):.2f} req/s")
    return latencies


def main():
    parser = argparse.ArgumentParser(description="kapsl-runtime LLM benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9095)
    parser.add_argument("--model-id", default="0", dest="model_id")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of parallel workers for concurrent test")
    parser.add_argument("--repeat", type=int, default=2,
                        help="How many times to repeat the prompt list")
    parser.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens",
                        help="Max tokens to generate per request")
    args = parser.parse_args()

    print(f"\nkapsl-runtime LLM Benchmark")
    print(f"  Target  : http://{args.host}:{args.port}")
    print(f"  Model ID: {args.model_id}")

    print("\nChecking server...", end=" ", flush=True)
    if not check_server(args.host, args.port):
        print(f"FAILED\n\nServer not reachable at http://{args.host}:{args.port}")
        print("Start it with:")
        print("  cargo run --release --bin kapsl -- run \\")
        print("    --model /path/to/qwen2.5-1.5b-instruct.aimod --transport http")
        sys.exit(1)
    print("OK")

    prompts = PROMPTS * args.repeat

    # 1. Warmup
    print("\nWarming up (1 request)...", end=" ", flush=True)
    r = infer(args.host, args.port, args.model_id, "Hello!", args.max_new_tokens)
    if r["error"]:
        print(f"FAILED: {r['error']}\n")
        sys.exit(1)
    print(f"OK ({r['latency_ms']:.0f} ms)")

    # 2. Sequential baseline
    seq_latencies = run_sequential(
        args.host, args.port, args.model_id, prompts,
        max_new_tokens=args.max_new_tokens,
    )

    # 3. Concurrent load
    run_concurrent(
        args.host, args.port, args.model_id, prompts, args.concurrency,
        max_new_tokens=args.max_new_tokens,
    )

    # 4. Summary
    if seq_latencies:
        print(f"\n{'='*64}")
        print("  SUMMARY")
        print(f"{'='*64}")
        print(f"  Sequential avg latency : {statistics.mean(seq_latencies):.1f} ms")
        print(f"  Sequential p95 latency : {sorted(seq_latencies)[int(len(seq_latencies)*0.95)]:.1f} ms")
        print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
