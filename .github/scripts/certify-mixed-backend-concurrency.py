#!/usr/bin/env python3
"""Exercise llama.cpp and managed vLLM concurrently through one Kapsl runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from managed_vllm_gpu_conformance import parse_prometheus


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    expected: re.Pattern[str]


CASES = (
    PromptCase("Reply with only the result of 2 + 3.", re.compile(r"\b5\b")),
    PromptCase(
        "What is the capital of France? Reply in one short sentence.",
        re.compile(r"\bParis\b", re.IGNORECASE),
    ),
    PromptCase(
        "Name the planet humans live on. Reply with one word.",
        re.compile(r"\bEarth\b", re.IGNORECASE),
    ),
    PromptCase(
        "What color is a clear daytime sky? Reply with one word.",
        re.compile(r"\bblue\b", re.IGNORECASE),
    ),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def request_once(
    base_url: str,
    backend: str,
    model: str,
    case: PromptCase,
    barrier: threading.Barrier,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": case.prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1234,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    barrier.wait(timeout=timeout)
    started = time.perf_counter()
    error_message: str | None = None
    content = ""
    status: int | None = None
    try:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = json.loads(response.read())
        content = str(body["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as error:
        status = error.code
        error_message = error.read().decode("utf-8", errors="replace")[:1000]
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        error_message = str(error)
    finished = time.perf_counter()
    semantically_valid = error_message is None and bool(case.expected.search(content))
    return {
        "backend": backend,
        "model": model,
        "prompt": case.prompt,
        "expected_regex": case.expected.pattern,
        "http_status": status,
        "content": content,
        "error": error_message,
        "semantically_valid": semantically_valid,
        "started": started,
        "finished": finished,
        "latency_ms": (finished - started) * 1000.0,
    }


def concurrency_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[tuple[float, int]] = []
    for sample in samples:
        events.append((float(sample["started"]), 1))
        events.append((float(sample["finished"]), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        maximum = max(maximum, active)

    llama = [sample for sample in samples if sample["backend"] == "llama_cpp"]
    vllm = [sample for sample in samples if sample["backend"] == "vllm"]
    cross_backend_overlap = any(
        left["started"] < right["finished"] and right["started"] < left["finished"]
        for left in llama
        for right in vllm
    )
    return {
        "max_in_flight": maximum,
        "cross_backend_overlap": cross_backend_overlap,
    }


def backend_summary(samples: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    selected = [sample for sample in samples if sample["backend"] == backend]
    latencies = [float(sample["latency_ms"]) for sample in selected]
    return {
        "requests": len(selected),
        "http_successes": sum(sample["error"] is None for sample in selected),
        "semantic_successes": sum(sample["semantically_valid"] for sample in selected),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
    }


def _values_by_device(
    metrics,
    name: str,
    labels: dict[str, str] | None = None,
    *,
    integral: bool,
) -> dict[str, float | int]:
    selected: dict[str, float | int] = {}
    for sample in metrics.get(name, []):
        if labels and any(sample.labels.get(key) != value for key, value in labels.items()):
            continue
        device = sample.labels.get("device")
        if device is None or not device.isdigit() or str(int(device)) != device:
            raise ValueError(f"metric {name} has no canonical device label")
        if device in selected:
            raise ValueError(f"metric {name} repeats device {device}")
        if integral and not sample.value.is_integer():
            raise ValueError(f"metric {name} is not integral: {sample.value}")
        selected[device] = int(sample.value) if integral else float(sample.value)
    if not selected:
        raise ValueError(f"metric {name} has no matching samples")
    return selected


def _integral_by_device(
    metrics, name: str, labels: dict[str, str] | None = None
) -> dict[str, int]:
    return _values_by_device(
        metrics, name, labels, integral=True
    )  # type: ignore[return-value]


def _numeric_by_device(metrics, name: str) -> dict[str, float]:
    return _values_by_device(
        metrics, name, integral=False
    )  # type: ignore[return-value]


def validate_mixed_memory(
    metrics_text: str, *, vllm_model: str, llama_model_id: int
) -> dict[str, Any]:
    metrics = parse_prometheus(metrics_text)
    vllm_labels = {"model": vllm_model}
    backing = _integral_by_device(
        metrics, "kapsl_managed_vllm_kv_backing_bytes", vllm_labels
    )
    quarantine = _integral_by_device(
        metrics, "kapsl_managed_vllm_kv_quarantine_bytes", vllm_labels
    )
    owner = f"gguf_kv:{llama_model_id}"
    llama_admitted = _integral_by_device(
        metrics, "kapsl_gpu_device_pool_owner_admitted", {"owner": owner}
    )
    llama_usage = _integral_by_device(
        metrics, "kapsl_gpu_device_pool_owner_usage_bytes", {"owner": owner}
    )
    authority_available = _integral_by_device(
        metrics, "kapsl_device_memory_available_bytes"
    )
    pool_allocated = _integral_by_device(
        metrics, "kapsl_gpu_device_pool_allocated_bytes"
    )
    pool_free = _integral_by_device(metrics, "kapsl_gpu_device_pool_free_bytes")
    pool_free_ranges = _integral_by_device(
        metrics, "kapsl_gpu_device_pool_free_ranges"
    )
    largest_free_ranges = _integral_by_device(
        metrics, "kapsl_gpu_device_pool_largest_free_range_bytes"
    )
    fragmentation = _numeric_by_device(
        metrics, "kapsl_gpu_device_pool_fragmentation_ratio"
    )
    if set(backing) != set(quarantine):
        raise ValueError(
            f"managed-vLLM backing/quarantine devices diverged: {backing} != {quarantine}"
        )
    if any(value <= 0 for value in backing.values()):
        raise ValueError(f"managed-vLLM backing is not active: {backing}")
    if any(quarantine.values()):
        raise ValueError(f"managed-vLLM has quarantined memory: {quarantine}")
    if set(llama_admitted) != set(llama_usage) or any(
        value != 1 for value in llama_admitted.values()
    ) or any(value <= 0 for value in llama_usage.values()):
        raise ValueError(
            f"llama.cpp shared-pool owner is not active: admitted={llama_admitted} usage={llama_usage}"
        )
    missing_authority = set(backing) - set(authority_available)
    if missing_authority or any(
        authority_available[device] <= 0 for device in backing
    ):
        raise ValueError(
            "mixed load consumed every authority byte instead of leaving ungranted VRAM: "
            f"{authority_available}"
        )
    owner_devices = set(llama_usage)
    pool_maps = (
        pool_allocated,
        pool_free,
        pool_free_ranges,
        largest_free_ranges,
        fragmentation,
    )
    if any(not owner_devices.issubset(values) for values in pool_maps) or any(
        pool_allocated[device] <= 0
        or pool_free[device] <= 0
        or pool_free_ranges[device] <= 0
        or largest_free_ranges[device] <= 0
        or largest_free_ranges[device] > pool_free[device]
        or not 0.0 <= fragmentation[device] <= 1.0
        for device in owner_devices
    ):
        raise ValueError(
            "mixed load did not retain a valid reusable general-pool range: "
            f"free={pool_free} ranges={pool_free_ranges} largest={largest_free_ranges} "
            f"fragmentation={fragmentation}"
        )
    return {
        "vllm_backing_bytes": backing,
        "vllm_quarantine_bytes": quarantine,
        "llama_owner": owner,
        "llama_owner_usage_bytes": llama_usage,
        "authority_available_bytes": authority_available,
        "general_pool_allocated_bytes": pool_allocated,
        "general_pool_free_bytes": pool_free,
        "general_pool_free_ranges": pool_free_ranges,
        "general_pool_largest_free_range_bytes": largest_free_ranges,
        "general_pool_fragmentation_ratio": fragmentation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metrics-url")
    parser.add_argument("--llama-model", required=True)
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--requests-per-backend", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.requests_per_backend <= 0 or args.max_tokens <= 0:
        parser.error("request and token counts must be positive")

    work = []
    for backend, model in (
        ("llama_cpp", args.llama_model),
        ("vllm", args.vllm_model),
    ):
        for index in range(args.requests_per_backend):
            work.append((backend, model, CASES[index % len(CASES)]))

    barrier = threading.Barrier(len(work))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(work)) as executor:
        futures = [
            executor.submit(
                request_once,
                args.base_url,
                backend,
                model,
                case,
                barrier,
                args.max_tokens,
                args.timeout_seconds,
            )
            for backend, model, case in work
        ]
        samples = [future.result(timeout=args.timeout_seconds + 30) for future in futures]

    concurrency = concurrency_summary(samples)
    summaries = {
        backend: backend_summary(samples, backend) for backend in ("llama_cpp", "vllm")
    }
    failures = []
    for backend, summary in summaries.items():
        if summary["http_successes"] != summary["requests"]:
            failures.append(f"{backend} had failed HTTP requests")
        if summary["semantic_successes"] != summary["requests"]:
            failures.append(f"{backend} returned semantically invalid answers")
    if concurrency["max_in_flight"] < 2:
        failures.append("requests did not overlap")
    if not concurrency["cross_backend_overlap"]:
        failures.append("llama.cpp and vLLM requests did not overlap")

    memory = None
    if args.metrics_url:
        try:
            with urllib.request.urlopen(
                f"{args.base_url.rstrip('/')}/api/models", timeout=args.timeout_seconds
            ) as response:
                models = json.load(response)
            llama_primary = [
                row
                for row in models
                if row.get("name") == args.llama_model
                and int(row.get("replica_id", -1)) == 0
            ]
            if len(llama_primary) != 1:
                raise ValueError(
                    f"expected one llama.cpp primary model row, found {llama_primary!r}"
                )
            with urllib.request.urlopen(
                args.metrics_url, timeout=args.timeout_seconds
            ) as response:
                metrics_text = response.read().decode("utf-8")
            memory = validate_mixed_memory(
                metrics_text,
                vllm_model=args.vllm_model,
                llama_model_id=int(llama_primary[0]["base_model_id"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"mixed memory accounting failed: {error}")

    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "concurrency": concurrency,
        "backends": summaries,
        "memory": memory,
        "failures": failures,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
