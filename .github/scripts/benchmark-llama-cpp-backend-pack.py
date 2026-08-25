#!/usr/bin/env python3
"""Measure and compare eager vs lazy llama.cpp streaming performance."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any


PROMPT = (
    "Explain in a precise technical paragraph how paged key-value caches reduce "
    "memory fragmentation during batched transformer inference. Continue until "
    "the requested token budget is exhausted."
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def discover_model(base_url: str, requested: str | None, timeout: float) -> str:
    if requested:
        return requested
    payload = read_json(f"{base_url.rstrip('/')}/v1/models", timeout)
    models = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(models, list) or not models:
        raise RuntimeError("runtime exposes no model through /v1/models")
    model = models[0]
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        raise RuntimeError("runtime returned an invalid /v1/models payload")
    return model["id"]


def stream_once(
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1234,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    content_times: list[float] = []
    completion_tokens: int | None = None
    finished: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"chat stream returned HTTP {response.status}")
            for raw_line in response:
                observed = time.perf_counter()
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = observed
                    break
                if not data:
                    continue
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"stream reported an error: {event['error']}")
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = event.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        content_times.append(observed)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat stream returned HTTP {error.code}: {body[:500]}") from error

    if finished is None:
        finished = time.perf_counter()
    if not content_times:
        raise RuntimeError("chat stream produced no non-empty content delta")
    if completion_tokens is None:
        raise RuntimeError("chat stream omitted requested completion-token usage")
    if completion_tokens < 2:
        raise RuntimeError(
            f"chat stream produced only {completion_tokens} completion token(s); need at least two"
        )

    first = content_times[0]
    inter_token_seconds = [
        later - earlier for earlier, later in zip(content_times, content_times[1:])
    ]
    if not inter_token_seconds:
        inter_token_seconds = [(finished - first) / max(completion_tokens - 1, 1)]
    decode_seconds = max(finished - first, 1e-9)
    return {
        "ttft_ms": (first - started) * 1000.0,
        "inter_token_ms": [gap * 1000.0 for gap in inter_token_seconds],
        "completion_tokens": completion_tokens,
        "decode_seconds": decode_seconds,
        "content_events": len(content_times),
    }


def benchmark(args: argparse.Namespace) -> int:
    model = discover_model(args.base_url, args.model, args.timeout_seconds)
    for _ in range(args.warmup):
        stream_once(args.base_url, model, args.max_tokens, args.timeout_seconds)

    samples = [
        stream_once(args.base_url, model, args.max_tokens, args.timeout_seconds)
        for _ in range(args.requests)
    ]
    ttft = [float(sample["ttft_ms"]) for sample in samples]
    inter_token = [
        float(gap) for sample in samples for gap in sample["inter_token_ms"]
    ]
    completion_tokens = sum(int(sample["completion_tokens"]) for sample in samples)
    decode_seconds = sum(float(sample["decode_seconds"]) for sample in samples)
    result = {
        "schema_version": 1,
        "label": args.label,
        "model": model,
        "requests": args.requests,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens_per_second": completion_tokens / decode_seconds,
        "p50_ttft_ms": percentile(ttft, 0.50),
        "p95_ttft_ms": percentile(ttft, 0.95),
        "p50_inter_token_ms": percentile(inter_token, 0.50),
        "p95_inter_token_ms": percentile(inter_token, 0.95),
        "mean_content_events": statistics.fmean(
            int(sample["content_events"]) for sample in samples
        ),
        "samples": samples,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    return 0


def compare_results(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    throughput_regression: float,
    latency_regression: float,
) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    failures: list[str] = []

    reference_throughput = float(reference["decode_tokens_per_second"])
    candidate_throughput = float(candidate["decode_tokens_per_second"])
    minimum_throughput = reference_throughput * (1.0 - throughput_regression)
    checks["decode_throughput"] = {
        "reference": reference_throughput,
        "candidate": candidate_throughput,
        "minimum": minimum_throughput,
        "passed": candidate_throughput >= minimum_throughput,
    }
    if not checks["decode_throughput"]["passed"]:
        failures.append(
            f"decode throughput {candidate_throughput:.3f} is below {minimum_throughput:.3f}"
        )

    for metric in ("p95_ttft_ms", "p95_inter_token_ms"):
        reference_latency = float(reference[metric])
        candidate_latency = float(candidate[metric])
        maximum_latency = reference_latency * (1.0 + latency_regression)
        checks[metric] = {
            "reference": reference_latency,
            "candidate": candidate_latency,
            "maximum": maximum_latency,
            "passed": candidate_latency <= maximum_latency,
        }
        if not checks[metric]["passed"]:
            failures.append(
                f"{metric} {candidate_latency:.3f} exceeds {maximum_latency:.3f}"
            )
    return checks, failures


def compare(args: argparse.Namespace) -> int:
    reference = json.loads(pathlib.Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(pathlib.Path(args.candidate).read_text(encoding="utf-8"))
    checks, failures = compare_results(
        reference,
        candidate,
        args.max_throughput_regression_percent / 100.0,
        args.max_latency_regression_percent / 100.0,
    )
    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "thresholds": {
            "max_throughput_regression_percent": args.max_throughput_regression_percent,
            "max_latency_regression_percent": args.max_latency_regression_percent,
        },
        "checks": checks,
        "failures": failures,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


def self_test() -> int:
    assert percentile([1.0, 2.0, 3.0], 0.50) == 2.0
    reference = {
        "decode_tokens_per_second": 100.0,
        "p95_ttft_ms": 100.0,
        "p95_inter_token_ms": 10.0,
    }
    passing = {
        "decode_tokens_per_second": 98.0,
        "p95_ttft_ms": 105.0,
        "p95_inter_token_ms": 10.5,
    }
    _, failures = compare_results(reference, passing, 0.02, 0.05)
    assert not failures
    failing = dict(passing)
    failing["decode_tokens_per_second"] = 97.99
    failing["p95_ttft_ms"] = 105.01
    _, failures = compare_results(reference, failing, 0.02, 0.05)
    assert len(failures) == 2
    print("llama.cpp backend-pack benchmark self-test passed")
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="measure one running Kapsl instance")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model")
    run.add_argument("--label", required=True)
    run.add_argument("--requests", type=positive_int, default=20)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--max-tokens", type=positive_int, default=64)
    run.add_argument("--timeout-seconds", type=float, default=180.0)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=benchmark)

    compare_parser = subparsers.add_parser("compare", help="enforce regression budgets")
    compare_parser.add_argument("--reference", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--max-throughput-regression-percent", type=float, default=2.0)
    compare_parser.add_argument("--max-latency-regression-percent", type=float, default=5.0)
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(handler=compare)

    test = subparsers.add_parser("self-test", help="exercise comparison math")
    test.set_defaults(handler=lambda _args: self_test())
    return root


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "warmup", 0) < 0:
        raise SystemExit("--warmup must be non-negative")
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
