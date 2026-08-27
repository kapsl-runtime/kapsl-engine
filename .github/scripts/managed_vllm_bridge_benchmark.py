#!/usr/bin/env python3
"""Reproducible direct-vLLM versus Kapsl OpenAI wire benchmark and gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import random
import statistics
import time
import urllib.parse
from pathlib import Path
from typing import Any, Sequence


class BenchmarkError(RuntimeError):
    pass


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("percentile requires values and a quantile in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _request_body(model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return an ordered list of short deterministic validation steps "
                    "for a GPU inference benchmark."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 23,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _stream_request(
    url: str, serialized_body: bytes, expected_tokens: int, timeout: float
) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=timeout
    )
    started = time.perf_counter()
    connection.request(
        "POST",
        parsed.path,
        body=serialized_body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raw = response.read()
        connection.close()
        raise BenchmarkError(
            f"HTTP {response.status} from {url}: {raw[:1000]!r}"
        )
    first_token_at: float | None = None
    done = False
    usage: dict[str, Any] | None = None
    completion_ids: set[str] = set()
    chunks = 0
    relayed_bytes = 0
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        relayed_bytes += len(raw_line)
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        chunks += 1
        if event.get("id"):
            completion_ids.add(str(event["id"]))
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                content = (choice.get("delta") or {}).get("content")
                if content and first_token_at is None:
                    first_token_at = time.perf_counter()
    finished = time.perf_counter()
    response.close()
    connection.close()
    if not done or first_token_at is None or usage is None:
        raise BenchmarkError("stream omitted [DONE], first content, or exact usage")
    completion_tokens = int(usage.get("completion_tokens", -1))
    if completion_tokens != expected_tokens:
        raise BenchmarkError(
            f"actual output length {completion_tokens} differs from required {expected_tokens}"
        )
    if len(completion_ids) != 1:
        raise BenchmarkError(f"stream completion IDs diverged: {completion_ids}")
    return {
        "ttft_seconds": first_token_at - started,
        "latency_seconds": finished - started,
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_id": next(iter(completion_ids)),
        "chunks": chunks,
        "relayed_bytes": relayed_bytes,
    }


def run_target(args: argparse.Namespace) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    body = _request_body(args.model, args.max_tokens)
    serialized = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body_digest = "sha256:" + hashlib.sha256(serialized).hexdigest()
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if not concurrencies or any(value <= 0 for value in concurrencies):
        raise BenchmarkError("concurrencies must be positive")
    if args.kv_cache_memory_bytes <= 0 or args.tensor_parallel_size <= 0:
        raise BenchmarkError("exact KV bytes and tensor parallel size must be positive")
    points: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        warmup_count = max(concurrency, args.warmup_requests)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            warmups = [
                pool.submit(
                    _stream_request,
                    url,
                    serialized,
                    args.max_tokens,
                    args.request_timeout,
                )
                for _ in range(warmup_count)
            ]
            for future in warmups:
                future.result()
        trials = []
        request_count = max(args.minimum_requests, concurrency * args.requests_per_worker)
        for trial_index in range(args.trials):
            trial_started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(
                        _stream_request,
                        url,
                        serialized,
                        args.max_tokens,
                        args.request_timeout,
                    )
                    for _ in range(request_count)
                ]
                requests = [future.result() for future in futures]
            elapsed = time.perf_counter() - trial_started
            total_tokens = sum(row["completion_tokens"] for row in requests)
            ttfts = [row["ttft_seconds"] for row in requests]
            latencies = [row["latency_seconds"] for row in requests]
            trials.append(
                {
                    "trial": trial_index,
                    "elapsed_seconds": elapsed,
                    "output_tokens": total_tokens,
                    "output_tokens_per_second": total_tokens / elapsed,
                    "median_ttft_seconds": statistics.median(ttfts),
                    "p95_ttft_seconds": percentile(ttfts, 0.95),
                    "median_latency_seconds": statistics.median(latencies),
                    "p95_latency_seconds": percentile(latencies, 0.95),
                    "requests": requests,
                }
            )
        points.append(
            {
                "concurrency": concurrency,
                "request_count_per_trial": request_count,
                "trials": trials,
            }
        )
    report = {
        "schema_version": 1,
        "status": "measured",
        "target": args.target,
        "base_url": args.base_url,
        "model": args.model,
        "request_body_sha256": body_digest,
        "request_body": body,
        "trials": args.trials,
        "engine_settings": {
            "vllm_version": args.vllm_version,
            "vllm_build_id": args.vllm_build_id,
            "model_revision": args.model_revision,
            "kv_cache_memory_bytes_per_rank": args.kv_cache_memory_bytes,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "attention_backend": "FLASH_ATTN",
            "enforce_eager": True,
            "prefix_caching": True,
        },
        "target_artifacts": {
            "runtime_build_id": args.runtime_build_id,
            "adapter_build_id": args.adapter_build_id,
        },
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _bootstrap_interval(
    direct: Sequence[float],
    kapsl: Sequence[float],
    operation,
    *,
    seed: int,
    samples: int = 20_000,
) -> tuple[float, float]:
    if len(direct) < 3 or len(kapsl) < 3:
        raise BenchmarkError("confidence intervals require at least three trials")
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled_direct = [generator.choice(direct) for _ in direct]
        sampled_kapsl = [generator.choice(kapsl) for _ in kapsl]
        estimates.append(
            operation(statistics.median(sampled_direct), statistics.median(sampled_kapsl))
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def compare_reports(
    direct: dict[str, Any], kapsl: dict[str, Any]
) -> dict[str, Any]:
    for field in (
        "schema_version",
        "model",
        "request_body_sha256",
        "trials",
        "engine_settings",
    ):
        if direct.get(field) != kapsl.get(field):
            raise BenchmarkError(f"benchmark reports differ in {field}")
    direct_points = {row["concurrency"]: row for row in direct["points"]}
    kapsl_points = {row["concurrency"]: row for row in kapsl["points"]}
    if direct_points.keys() != kapsl_points.keys():
        raise BenchmarkError("benchmark reports cover different concurrencies")
    comparisons = []
    failures = []
    for concurrency in sorted(direct_points):
        direct_trials = direct_points[concurrency]["trials"]
        kapsl_trials = kapsl_points[concurrency]["trials"]
        if len(direct_trials) != len(kapsl_trials):
            raise BenchmarkError("benchmark reports have different trial counts")
        direct_throughput = [row["output_tokens_per_second"] for row in direct_trials]
        kapsl_throughput = [row["output_tokens_per_second"] for row in kapsl_trials]
        direct_median_ttft = [row["median_ttft_seconds"] for row in direct_trials]
        kapsl_median_ttft = [row["median_ttft_seconds"] for row in kapsl_trials]
        direct_p95_ttft = [row["p95_ttft_seconds"] for row in direct_trials]
        kapsl_p95_ttft = [row["p95_ttft_seconds"] for row in kapsl_trials]

        throughput_loss = 1 - (
            statistics.median(kapsl_throughput)
            / statistics.median(direct_throughput)
        )
        throughput_ci = _bootstrap_interval(
            direct_throughput,
            kapsl_throughput,
            lambda baseline, candidate: 1 - candidate / baseline,
            seed=1000 + concurrency,
        )
        median_ttft_delta = statistics.median(kapsl_median_ttft) - statistics.median(
            direct_median_ttft
        )
        median_ttft_ci = _bootstrap_interval(
            direct_median_ttft,
            kapsl_median_ttft,
            lambda baseline, candidate: candidate - baseline,
            seed=2000 + concurrency,
        )
        p95_ttft_delta = statistics.median(kapsl_p95_ttft) - statistics.median(
            direct_p95_ttft
        )
        p95_ttft_ci = _bootstrap_interval(
            direct_p95_ttft,
            kapsl_p95_ttft,
            lambda baseline, candidate: candidate - baseline,
            seed=3000 + concurrency,
        )
        gates = {
            "throughput_loss_at_most_2_percent": throughput_loss <= 0.02 + 1e-12,
            "throughput_95ci_upper_at_most_2_percent": throughput_ci[1]
            <= 0.02 + 1e-12,
            "median_ttft_delta_at_most_5ms": median_ttft_delta <= 0.005 + 1e-12,
            "p95_ttft_delta_at_most_10ms": p95_ttft_delta <= 0.010 + 1e-12,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            failures.append({"concurrency": concurrency, "gates": failed})
        comparisons.append(
            {
                "concurrency": concurrency,
                "direct_median_output_tokens_per_second": statistics.median(
                    direct_throughput
                ),
                "kapsl_median_output_tokens_per_second": statistics.median(
                    kapsl_throughput
                ),
                "throughput_loss_fraction": throughput_loss,
                "throughput_loss_95ci": list(throughput_ci),
                "median_ttft_delta_seconds": median_ttft_delta,
                "median_ttft_delta_95ci": list(median_ttft_ci),
                "p95_ttft_delta_seconds": p95_ttft_delta,
                "p95_ttft_delta_95ci": list(p95_ttft_ci),
                "gates": gates,
            }
        )
    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "request_body_sha256": direct["request_body_sha256"],
        "comparisons": comparisons,
        "failures": failures,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    direct = json.loads(args.direct.read_text(encoding="utf-8"))
    kapsl = json.loads(args.kapsl.read_text(encoding="utf-8"))
    report = compare_reports(direct, kapsl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["status"] != "passed":
        raise BenchmarkError(f"managed-vLLM bridge gates failed: {report['failures']}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--target", choices=("direct", "kapsl"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--concurrencies", default="1,4,8,16")
    run.add_argument("--trials", type=int, default=7)
    run.add_argument("--requests-per-worker", type=int, default=4)
    run.add_argument("--minimum-requests", type=int, default=8)
    run.add_argument("--warmup-requests", type=int, default=2)
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--request-timeout", type=float, default=180)
    run.add_argument("--vllm-version", required=True)
    run.add_argument("--vllm-build-id", required=True)
    run.add_argument("--model-revision", required=True)
    run.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    run.add_argument("--tensor-parallel-size", type=int, required=True)
    run.add_argument("--max-model-len", type=int, default=1024)
    run.add_argument("--runtime-build-id", default="not-applicable")
    run.add_argument("--adapter-build-id", default="not-applicable")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--direct", type=Path, required=True)
    compare_parser.add_argument("--kapsl", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        run_target(args)
    else:
        compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
