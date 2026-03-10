#!/usr/bin/env python3
"""
Lightweight HTTP benchmark for kapsl-runtime infer API.

Outputs:
- p50 latency (ms)
- p95 latency (ms)
- throughput (successful requests / second)

Example:
  python3 scripts/benchmark_http_infer.py --model-id 0 --requests 100 --warmup 10
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple


def build_default_payload() -> Dict[str, Any]:
    # Default is MNIST-like [1,1,28,28] float32 input filled with 0.5.
    one_value = struct.pack("@f", 0.5)
    data = one_value * (1 * 1 * 28 * 28)
    return {
        "input": {
            "shape": [1, 1, 28, 28],
            "dtype": "float32",
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    }


def load_payload(payload_file: str | None) -> Dict[str, Any]:
    if not payload_file:
        return build_default_payload()
    with open(payload_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Payload file must contain a JSON object")
    return payload


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def send_infer(
    url: str,
    token: str | None,
    payload_json: bytes,
    timeout_seconds: float,
) -> Tuple[float, int]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=payload_json, headers=headers, method="POST"
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        _ = response.read()
        status = int(response.status)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return latency_ms, status


def run_phase(
    phase: str,
    requests: int,
    url: str,
    token: str | None,
    payload_json: bytes,
    timeout_seconds: float,
    quiet: bool,
) -> Tuple[List[float], int]:
    latencies_ms: List[float] = []
    failures = 0

    for i in range(requests):
        try:
            latency_ms, _ = send_infer(url, token, payload_json, timeout_seconds)
            latencies_ms.append(latency_ms)
            if not quiet:
                print(f"[{phase}] {i + 1}/{requests}: {latency_ms:.2f} ms")
        except urllib.error.HTTPError as err:
            failures += 1
            body = err.read().decode("utf-8", errors="replace").strip()
            if not quiet:
                print(
                    f"[{phase}] {i + 1}/{requests}: HTTP {err.code} {err.reason} {body}",
                    file=sys.stderr,
                )
        except Exception as err:  # noqa: BLE001 - small script, explicit surfacing
            failures += 1
            if not quiet:
                print(f"[{phase}] {i + 1}/{requests}: ERROR {err}", file=sys.stderr)

    return latencies_ms, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark kapsl-runtime HTTP infer endpoint"
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:9095", help="Runtime base URL"
    )
    parser.add_argument(
        "--model-id", type=int, required=True, help="Model ID to benchmark"
    )
    parser.add_argument(
        "--requests", type=int, default=100, help="Benchmark request count"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup request count")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request HTTP timeout",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional API token (falls back to KAPSL_API_TOKEN / KAPSL_DESKTOP_API_TOKEN; legacy KAPSL_* also works)",
    )
    parser.add_argument(
        "--payload-file",
        default=None,
        help="Optional JSON file for infer payload; defaults to MNIST-like tensor payload",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print final summary")
    args = parser.parse_args()

    if args.requests <= 0:
        print("--requests must be > 0", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2

    token = (
        args.token
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
    )
    try:
        payload = load_payload(args.payload_file)
    except Exception as err:  # noqa: BLE001
        print(f"Failed to load payload: {err}", file=sys.stderr)
        return 2
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    infer_url = f"{args.base_url.rstrip('/')}/api/models/{args.model_id}/infer"

    if not args.quiet:
        print("Benchmark configuration:")
        print(f"  url: {infer_url}")
        print(f"  requests: {args.requests}")
        print(f"  warmup: {args.warmup}")
        print(f"  timeout_seconds: {args.timeout_seconds}")
        print(f"  token: {'set' if token else 'not set'}")
        if args.payload_file:
            print(f"  payload_file: {args.payload_file}")
        else:
            print("  payload: built-in MNIST-like float32 tensor")
        print()

    if args.warmup > 0:
        _, warmup_failures = run_phase(
            "warmup",
            args.warmup,
            infer_url,
            token,
            payload_json,
            args.timeout_seconds,
            args.quiet,
        )
        if warmup_failures and not args.quiet:
            print(f"Warmup failures: {warmup_failures}", file=sys.stderr)

    bench_start = time.perf_counter()
    latencies_ms, failures = run_phase(
        "bench",
        args.requests,
        infer_url,
        token,
        payload_json,
        args.timeout_seconds,
        args.quiet,
    )
    bench_elapsed = time.perf_counter() - bench_start

    successes = len(latencies_ms)
    if successes == 0:
        print("No successful benchmark requests.", file=sys.stderr)
        print(
            json.dumps(
                {
                    "requests": args.requests,
                    "successes": 0,
                    "failures": failures,
                }
            )
        )
        return 1

    latencies_sorted = sorted(latencies_ms)
    p50 = percentile(latencies_sorted, 0.50)
    p95 = percentile(latencies_sorted, 0.95)
    avg = sum(latencies_ms) / successes
    throughput = successes / bench_elapsed if bench_elapsed > 0 else float("inf")

    print("Benchmark summary:")
    print(f"  requests:    {args.requests}")
    print(f"  successes:   {successes}")
    print(f"  failures:    {failures}")
    print(f"  avg_ms:      {avg:.2f}")
    print(f"  p50_ms:      {p50:.2f}")
    print(f"  p95_ms:      {p95:.2f}")
    print(f"  throughput:  {throughput:.2f} req/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
