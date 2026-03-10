#!/usr/bin/env python3
"""
Concurrent HTTP infer load generator that samples /api/models/{id} queue depth.

Usage example:
  python3 scripts/test_cpu_queue_depth_http.py --model-id 0 --concurrency 24 --requests-per-worker 8
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


def build_default_payload() -> Dict[str, Any]:
    # Default MNIST-like tensor [1,1,28,28] float32.
    one_value = struct.pack("@f", 0.5)
    data = one_value * (1 * 1 * 28 * 28)
    return {
        "input": {
            "shape": [1, 1, 28, 28],
            "dtype": "float32",
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    }


def load_payload(payload_file: Optional[str]) -> Dict[str, Any]:
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


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def post_json(
    url: str, token: Optional[str], payload: Dict[str, Any], timeout: float
) -> Any:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers=auth_headers(token),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return None
    return json.loads(raw)


def get_json(url: str, token: Optional[str], timeout: float) -> Any:
    req = urllib.request.Request(
        url,
        headers=auth_headers(token),
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def single_infer(
    infer_url: str,
    token: Optional[str],
    base_payload: Dict[str, Any],
    timeout_seconds: float,
    request_id: str,
    force_cpu: bool,
) -> Tuple[bool, float, str]:
    payload = dict(base_payload)
    metadata = dict(payload.get("metadata") or {})
    metadata["request_id"] = request_id
    metadata["priority"] = 1
    metadata["force_cpu"] = force_cpu
    payload["metadata"] = metadata

    started = time.perf_counter()
    try:
        post_json(infer_url, token, payload, timeout_seconds)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return True, latency_ms, ""
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace").strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return False, latency_ms, f"HTTP {err.code} {err.reason} {body}"
    except Exception as err:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000.0
        return False, latency_ms, str(err)


def queue_sampler(
    stop_event: threading.Event,
    samples: List[Dict[str, Any]],
    model_url: str,
    token: Optional[str],
    poll_interval: float,
    timeout_seconds: float,
) -> None:
    while not stop_event.is_set():
        at = time.time()
        try:
            model = get_json(model_url, token, timeout_seconds)
            queue_depth = model.get("queue_depth", [0, 0])
            active = int(model.get("active_inferences", 0))
            cpu_depth = int(queue_depth[0]) if len(queue_depth) > 0 else 0
            gpu_depth = int(queue_depth[1]) if len(queue_depth) > 1 else 0
            samples.append(
                {
                    "ts": at,
                    "cpu_depth": cpu_depth,
                    "gpu_depth": gpu_depth,
                    "active_inferences": active,
                }
            )
        except Exception:
            # Keep sampling loop alive even if model endpoint temporarily fails.
            samples.append(
                {
                    "ts": at,
                    "cpu_depth": -1,
                    "gpu_depth": -1,
                    "active_inferences": -1,
                }
            )
        time.sleep(poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run concurrent infer requests and observe CPU/GPU queue depth."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9095")
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests-per-worker", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--payload-file", default=None)
    parser.add_argument(
        "--token",
        default=None,
        help="Optional API token (falls back to KAPSL_API_TOKEN / KAPSL_DESKTOP_API_TOKEN; legacy KAPSL_* also works)",
    )
    parser.add_argument(
        "--force-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set request.metadata.force_cpu (default: true).",
    )
    parser.add_argument(
        "--require-cpu-depth-at-least",
        type=int,
        default=1,
        help="Exit non-zero if observed max CPU depth is below this value.",
    )
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.requests_per_worker <= 0:
        raise SystemExit("--requests-per-worker must be > 0")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be > 0")

    token = (
        args.token
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
    )

    base_payload = load_payload(args.payload_file)
    infer_url = f"{args.base_url.rstrip('/')}/api/models/{args.model_id}/infer"
    model_url = f"{args.base_url.rstrip('/')}/api/models/{args.model_id}"
    total_requests = args.concurrency * args.requests_per_worker

    print("CPU Queue Depth Test")
    print(f"  infer_url: {infer_url}")
    print(f"  model_url: {model_url}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  requests_per_worker: {args.requests_per_worker}")
    print(f"  total_requests: {total_requests}")
    print(f"  force_cpu: {args.force_cpu}")
    print(f"  poll_interval: {args.poll_interval}s")
    print(f"  token: {'set' if token else 'not set'}")

    # Quick preflight: fail fast if model endpoint is unavailable.
    try:
        _ = get_json(model_url, token, args.timeout_seconds)
    except Exception as err:  # noqa: BLE001
        print(f"\nPreflight failed: cannot fetch {model_url}: {err}")
        return 2

    samples: List[Dict[str, Any]] = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=queue_sampler,
        args=(
            stop_event,
            samples,
            model_url,
            token,
            args.poll_interval,
            args.timeout_seconds,
        ),
        daemon=True,
    )
    sampler.start()

    latencies: List[float] = []
    failures: List[str] = []

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for worker in range(args.concurrency):
            for req_idx in range(args.requests_per_worker):
                request_id = f"cpuq-{worker}-{req_idx}-{time.time_ns()}"
                futures.append(
                    executor.submit(
                        single_infer,
                        infer_url,
                        token,
                        base_payload,
                        args.timeout_seconds,
                        request_id,
                        args.force_cpu,
                    )
                )

        for future in as_completed(futures):
            ok, latency_ms, error = future.result()
            latencies.append(latency_ms)
            if not ok:
                failures.append(error)

    elapsed = time.perf_counter() - started
    stop_event.set()
    sampler.join(timeout=2.0)

    valid_samples = [s for s in samples if s["cpu_depth"] >= 0]
    max_cpu_depth = max((s["cpu_depth"] for s in valid_samples), default=0)
    max_gpu_depth = max((s["gpu_depth"] for s in valid_samples), default=0)
    max_active = max((s["active_inferences"] for s in valid_samples), default=0)

    lat_sorted = sorted(latencies)
    p50 = percentile(lat_sorted, 0.50)
    p95 = percentile(lat_sorted, 0.95)
    throughput = (
        (len(latencies) - len(failures)) / elapsed if elapsed > 0 else float("inf")
    )

    print("\nSummary")
    print(f"  elapsed_s: {elapsed:.2f}")
    print(f"  successes: {len(latencies) - len(failures)}")
    print(f"  failures: {len(failures)}")
    print(f"  throughput_req_s: {throughput:.2f}")
    print(f"  latency_p50_ms: {p50:.2f}")
    print(f"  latency_p95_ms: {p95:.2f}")
    print(f"  max_cpu_queue_depth: {max_cpu_depth}")
    print(f"  max_gpu_queue_depth: {max_gpu_depth}")
    print(f"  max_active_inferences: {max_active}")
    print(f"  queue_samples: {len(valid_samples)}")

    if failures:
        print("\nSample errors:")
        for err in failures[:5]:
            print(f"  - {err}")

    if max_cpu_depth < args.require_cpu_depth_at_least:
        print(
            f"\nFAIL: max_cpu_queue_depth={max_cpu_depth} < {args.require_cpu_depth_at_least}"
        )
        print(
            "Hint: ensure runtime includes metadata.force_cpu support and model_id points to a loaded model."
        )
        return 1

    print("\nPASS: observed CPU queue depth increase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
