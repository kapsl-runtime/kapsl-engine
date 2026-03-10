#!/usr/bin/env python3
"""
Memory fragmentation probe for kapsl-runtime.

This script repeatedly:
1. Sends concurrent infer requests across one or more models.
2. Waits until all tested model queues are idle.
3. Samples kapsl process RSS.

Fragmentation is inferred when idle RSS keeps drifting upward over cycles.

Example:
  python3 test_memory_fragmentation.py \
    --model-ids 0,1 \
    --payload-by-model 0=payloads/mnist.json \
    --payload-by-model 1=payloads/squeezenet.json \
    --cycles 20 \
    --concurrency 64 \
    --requests-per-worker 200 \
    --force-cpu
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import random
import string
import struct
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CycleResult:
    cycle: int
    burst_elapsed_s: float
    successes: int
    failures: int
    p50_ms: float
    p95_ms: float
    idle_rss_kb: int
    idle_total_model_memory_bytes: int
    idle_model_memory_by_id: Dict[int, int]
    idle_active_replicas_by_base_id: Dict[int, int]
    per_model_successes: Dict[int, int]
    per_model_failures: Dict[int, int]


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


def build_default_payload() -> Dict[str, Any]:
    # MNIST-like tensor [1,1,28,28], float32(0.5)
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


def random_ascii_bytes(rng: random.Random, min_bytes: int, max_bytes: int) -> bytes:
    # Keep prompts ASCII so bytes length == character length.
    if min_bytes <= 0:
        min_bytes = 1
    if max_bytes < min_bytes:
        max_bytes = min_bytes

    length = rng.randint(min_bytes, max_bytes)
    alphabet = string.ascii_letters + string.digits + " ,.;:?!-_/()[]{}"
    if length <= 0:
        length = 1
    prompt = "".join(rng.choice(alphabet) for _ in range(length))
    if not prompt.strip():
        prompt = "a" * length
    return prompt.encode("ascii", errors="ignore")


def maybe_randomize_string_payload(
    base_payload: Dict[str, Any],
    rng: random.Random,
    min_bytes: int,
    max_bytes: int,
) -> Dict[str, Any]:
    input_obj = base_payload.get("input")
    if not isinstance(input_obj, dict):
        return base_payload
    dtype = str(input_obj.get("dtype", "")).lower()
    if dtype != "string":
        return base_payload

    payload = dict(base_payload)
    new_input = dict(input_obj)
    data = random_ascii_bytes(rng, min_bytes, max_bytes)
    new_input.pop("data", None)
    new_input["data_base64"] = base64.b64encode(data).decode("ascii")
    # Keep shape consistent with byte payload (dtype Utf8 uses 1 byte per element).
    new_input["shape"] = [1, len(data)]
    payload["input"] = new_input
    return payload


def parse_model_ids(
    model_id: Optional[int], model_ids_text: Optional[str]
) -> List[int]:
    ids: List[int] = []
    if model_ids_text:
        for raw in model_ids_text.replace(",", " ").split():
            try:
                ids.append(int(raw))
            except ValueError as err:
                raise ValueError(f"Invalid model id '{raw}' in --model-ids") from err

    if model_id is not None:
        ids.insert(0, model_id)

    if not ids:
        ids = [0]

    seen = set()
    deduped: List[int] = []
    for mid in ids:
        if mid in seen:
            continue
        seen.add(mid)
        deduped.append(mid)
    return deduped


def parse_payload_by_model(entries: List[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --payload-by-model '{raw}'. Expected format MODEL_ID=PATH"
            )

        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(
                f"Invalid --payload-by-model '{raw}'. Missing MODEL_ID before '='"
            )
        if not value:
            raise ValueError(
                f"Invalid --payload-by-model '{raw}'. Missing PATH after '='"
            )

        try:
            model_id = int(key)
        except ValueError as err:
            raise ValueError(
                f"Invalid --payload-by-model '{raw}'. MODEL_ID must be an integer"
            ) from err

        if model_id in mapping:
            raise ValueError(f"Duplicate --payload-by-model entry for model {model_id}")

        mapping[model_id] = value

    return mapping


def load_payloads_for_models(
    model_ids: List[int],
    default_payload_file: Optional[str],
    payload_files_by_model: Dict[int, str],
) -> Dict[int, Dict[str, Any]]:
    payloads: Dict[int, Dict[str, Any]] = {}
    default_payload = load_payload(default_payload_file)

    for model_id in model_ids:
        payload_file = payload_files_by_model.get(model_id)
        if payload_file:
            payloads[model_id] = load_payload(payload_file)
        else:
            payloads[model_id] = default_payload

    return payloads


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def http_get_json(url: str, token: Optional[str], timeout: float) -> Any:
    req = urllib.request.Request(url, headers=auth_headers(token), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post_bytes(url: str, token: Optional[str], body: bytes, timeout: float) -> Any:
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


def http_post_json(url: str, token: Optional[str], payload: Any, timeout: float) -> Any:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return http_post_bytes(url, token, body, timeout)


def list_models(
    base_url: str, token: Optional[str], timeout: float
) -> List[Dict[str, Any]]:
    payload = http_get_json(f"{base_url.rstrip('/')}/api/models", token, timeout)
    if not isinstance(payload, list):
        raise RuntimeError(f"Invalid /api/models response: {type(payload).__name__}")
    return [entry for entry in payload if isinstance(entry, dict)]


def count_active_replicas(models: List[Dict[str, Any]], base_model_id: int) -> int:
    count = 0
    for model in models:
        try:
            mid = int(model.get("base_model_id", -1))
        except (TypeError, ValueError):
            continue
        if mid != base_model_id:
            continue
        status = str(model.get("status", "")).lower()
        if status == "active":
            count += 1
    return count


def build_payload_body(
    base_payload: Dict[str, Any],
    force_cpu: bool,
    request_id: Optional[str] = None,
) -> bytes:
    payload = dict(base_payload)
    metadata = dict(payload.get("metadata") or {})
    if request_id is not None:
        metadata["request_id"] = request_id
    if force_cpu:
        metadata["force_cpu"] = True
    if metadata:
        payload["metadata"] = metadata
    else:
        payload.pop("metadata", None)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def read_rss_kb(pid: int) -> int:
    output = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(pid)],
        text=True,
    ).strip()
    if not output:
        raise RuntimeError(f"Process {pid} not found")
    return int(output)


def resolve_kapsl_pid(pid: Optional[int], process_pattern: str) -> int:
    if pid is not None:
        _ = read_rss_kb(pid)
        return pid

    output = subprocess.check_output(
        ["pgrep", "-n", "-f", process_pattern],
        text=True,
    ).strip()
    if not output:
        raise RuntimeError(
            f"Could not find kapsl process with pattern '{process_pattern}'. "
            "Use --pid to specify explicitly."
        )
    return int(output)


def send_single_infer(
    infer_url: str,
    token: Optional[str],
    payload_body: bytes,
    timeout_seconds: float,
) -> Tuple[bool, float, str]:
    started = time.perf_counter()
    try:
        http_post_bytes(infer_url, token, payload_body, timeout_seconds)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return True, latency_ms, ""
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace").strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return False, latency_ms, f"HTTP {err.code} {err.reason} {body}"
    except Exception as err:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000.0
        return False, latency_ms, str(err)


def run_burst(
    infer_urls: Dict[int, str],
    payloads_by_model: Dict[int, Dict[str, Any]],
    payload_bodies_by_model: Dict[int, bytes],
    model_ids: List[int],
    token: Optional[str],
    timeout_seconds: float,
    concurrency: int,
    requests_per_worker: int,
    force_cpu: bool,
    unique_request_id_per_request: bool,
    randomize_string_input: bool,
    string_rng: Optional[random.Random],
    string_min_bytes: int,
    string_max_bytes: int,
) -> Tuple[float, int, int, float, float, List[str], Dict[int, int], Dict[int, int]]:
    if not model_ids:
        raise ValueError("model_ids must not be empty")
    if randomize_string_input and string_rng is None:
        raise ValueError(
            "string_rng must be provided when randomize_string_input is enabled"
        )

    total_requests = concurrency * requests_per_worker
    latencies: List[float] = []
    errors: List[str] = []
    successes_by_model = {mid: 0 for mid in model_ids}
    failures_by_model = {mid: 0 for mid in model_ids}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        future_model_map: Dict[Any, int] = {}
        for worker in range(concurrency):
            for req_idx in range(requests_per_worker):
                target_idx = (worker * requests_per_worker + req_idx) % len(model_ids)
                model_id = model_ids[target_idx]
                infer_url = infer_urls[model_id]
                if unique_request_id_per_request or randomize_string_input:
                    request_id = (
                        f"frag-m{model_id}-{worker}-{req_idx}-{time.time_ns()}"
                        if unique_request_id_per_request
                        else None
                    )
                    base_payload = payloads_by_model[model_id]
                    if randomize_string_input and string_rng is not None:
                        base_payload = maybe_randomize_string_payload(
                            base_payload,
                            rng=string_rng,
                            min_bytes=string_min_bytes,
                            max_bytes=string_max_bytes,
                        )
                    payload_body = build_payload_body(
                        base_payload,
                        force_cpu=force_cpu,
                        request_id=request_id,
                    )
                else:
                    payload_body = payload_bodies_by_model[model_id]
                fut = pool.submit(
                    send_single_infer,
                    infer_url,
                    token,
                    payload_body,
                    timeout_seconds,
                )
                futures.append(fut)
                future_model_map[fut] = model_id

        for fut in as_completed(futures):
            ok, latency_ms, err = fut.result()
            model_id = future_model_map[fut]
            latencies.append(latency_ms)
            if ok:
                successes_by_model[model_id] += 1
            else:
                failures_by_model[model_id] += 1
                errors.append(err)

    elapsed = time.perf_counter() - started
    successes = total_requests - len(errors)
    failures = len(errors)
    latencies_sorted = sorted(latencies)
    p50 = percentile(latencies_sorted, 0.50)
    p95 = percentile(latencies_sorted, 0.95)
    return (
        elapsed,
        successes,
        failures,
        p50,
        p95,
        errors,
        successes_by_model,
        failures_by_model,
    )


def wait_for_idle_models(
    model_urls: Dict[int, str],
    model_ids: List[int],
    token: Optional[str],
    timeout_seconds: float,
    poll_interval: float,
    stable_samples: int,
) -> Dict[int, Dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    last_states: Dict[int, Dict[str, Any]] = {}

    while time.monotonic() < deadline:
        current_states: Dict[int, Dict[str, Any]] = {}
        all_idle = True
        for model_id in model_ids:
            state = http_get_json(model_urls[model_id], token, timeout_seconds)
            parsed = state if isinstance(state, dict) else {}
            current_states[model_id] = parsed
            queue = parsed.get("queue_depth", [0, 0])
            active = int(parsed.get("active_inferences", 0))
            q0 = int(queue[0]) if isinstance(queue, list) and len(queue) > 0 else 0
            q1 = int(queue[1]) if isinstance(queue, list) and len(queue) > 1 else 0
            idle = active == 0 and q0 == 0 and q1 == 0
            if not idle:
                all_idle = False

        last_states = current_states
        if all_idle:
            stable += 1
            if stable >= stable_samples:
                return last_states
        else:
            stable = 0

        time.sleep(poll_interval)

    last_bits: List[str] = []
    for model_id in model_ids:
        state = last_states.get(model_id, {})
        last_bits.append(
            f"model={model_id} active={state.get('active_inferences')} queue={state.get('queue_depth')}"
        )

    raise TimeoutError(
        "Timed out waiting for all models idle (last states: "
        + "; ".join(last_bits)
        + ")"
    )


def write_csv(path: str, rows: List[CycleResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cycle",
                "burst_elapsed_s",
                "successes",
                "failures",
                "p50_ms",
                "p95_ms",
                "idle_rss_kb",
                "idle_total_model_memory_bytes",
                "idle_model_memory_by_id_json",
                "idle_active_replicas_by_base_id_json",
                "per_model_successes_json",
                "per_model_failures_json",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.cycle,
                    f"{row.burst_elapsed_s:.6f}",
                    row.successes,
                    row.failures,
                    f"{row.p50_ms:.6f}",
                    f"{row.p95_ms:.6f}",
                    row.idle_rss_kb,
                    row.idle_total_model_memory_bytes,
                    json.dumps(row.idle_model_memory_by_id, sort_keys=True),
                    json.dumps(row.idle_active_replicas_by_base_id, sort_keys=True),
                    json.dumps(row.per_model_successes, sort_keys=True),
                    json.dumps(row.per_model_failures, sort_keys=True),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated inference bursts and detect idle RSS drift."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9095")
    parser.add_argument(
        "--model-id",
        type=int,
        default=None,
        help="Single model id to test. If omitted, defaults to 0 unless --model-ids is provided.",
    )
    parser.add_argument(
        "--model-ids",
        default=None,
        help="Comma/space-separated model ids to test together (example: 0,1,2).",
    )
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests-per-worker", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--idle-poll-interval", type=float, default=0.2)
    parser.add_argument("--idle-stable-samples", type=int, default=3)
    parser.add_argument(
        "--payload-file",
        default=None,
        help="Default payload JSON file for all models without a per-model override.",
    )
    parser.add_argument(
        "--payload-by-model",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Per-model payload override; repeatable (example: --payload-by-model 1=payloads/squeezenet.json).",
    )
    parser.add_argument("--pid", type=int, default=None, help="Optional kapsl PID")
    parser.add_argument(
        "--process-pattern",
        default="(^|/)kapsl( |$)",
        help="Pattern used by pgrep -f when --pid is not set",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional API token (falls back to KAPSL_API_TOKEN / KAPSL_DESKTOP_API_TOKEN; legacy KAPSL_* also works)",
    )
    parser.add_argument(
        "--force-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set request.metadata.force_cpu=true for all burst requests.",
    )
    parser.add_argument(
        "--unique-request-id-per-request",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate a unique metadata.request_id for each request. "
            "This adds client-side JSON overhead and can skew latency metrics."
        ),
    )
    parser.add_argument(
        "--randomize-string-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If input.dtype is 'string', randomize input.data per request to better mimic production fragmentation.",
    )
    parser.add_argument(
        "--string-min-bytes",
        type=int,
        default=64,
        help="Minimum random string size in bytes (ASCII) when --randomize-string-input is enabled.",
    )
    parser.add_argument(
        "--string-max-bytes",
        type=int,
        default=2048,
        help="Maximum random string size in bytes (ASCII) when --randomize-string-input is enabled.",
    )
    parser.add_argument(
        "--string-seed",
        type=int,
        default=None,
        help="Optional RNG seed used for --randomize-string-input.",
    )
    parser.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=15.0,
        help="Fail if max idle RSS drift from baseline exceeds this percentage.",
    )
    parser.add_argument(
        "--freeze-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily lock scaling policy during the run.",
    )
    parser.add_argument(
        "--freeze-scaling-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if scaling policy freeze/restore cannot be completed.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to write cycle metrics as CSV.",
    )
    args = parser.parse_args()

    if args.cycles <= 0:
        raise SystemExit("--cycles must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.requests_per_worker <= 0:
        raise SystemExit("--requests-per-worker must be > 0")
    if args.idle_stable_samples <= 0:
        raise SystemExit("--idle-stable-samples must be > 0")
    if args.idle_poll_interval <= 0:
        raise SystemExit("--idle-poll-interval must be > 0")
    if args.randomize_string_input:
        if args.string_min_bytes <= 0:
            raise SystemExit(
                "--string-min-bytes must be > 0 when --randomize-string-input is enabled"
            )
        if args.string_max_bytes < args.string_min_bytes:
            raise SystemExit(
                "--string-max-bytes must be >= --string-min-bytes when --randomize-string-input is enabled"
            )

    try:
        model_ids = parse_model_ids(args.model_id, args.model_ids)
    except ValueError as err:
        raise SystemExit(str(err)) from err

    try:
        payload_files_by_model = parse_payload_by_model(args.payload_by_model)
    except ValueError as err:
        raise SystemExit(str(err)) from err

    extra_payload_models = sorted(set(payload_files_by_model) - set(model_ids))
    if extra_payload_models:
        raise SystemExit(
            "--payload-by-model contains model ids not in test set: "
            + ", ".join(str(mid) for mid in extra_payload_models)
        )

    try:
        payloads_by_model = load_payloads_for_models(
            model_ids,
            args.payload_file,
            payload_files_by_model,
        )
    except Exception as err:  # noqa: BLE001
        raise SystemExit(f"Failed to load payload(s): {err}") from err

    token = (
        args.token
        or os.getenv("KAPSL_API_TOKEN_ADMIN")
        or os.getenv("KAPSL_API_TOKEN_WRITER")
        or os.getenv("KAPSL_API_TOKEN_READER")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN_ADMIN")
        or os.getenv("KAPSL_API_TOKEN_WRITER")
        or os.getenv("KAPSL_API_TOKEN_READER")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
    )

    base_url = args.base_url.rstrip("/")
    infer_urls = {mid: f"{base_url}/api/models/{mid}/infer" for mid in model_ids}
    model_urls = {mid: f"{base_url}/api/models/{mid}" for mid in model_ids}
    total_requests = args.concurrency * args.requests_per_worker
    string_rng = (
        random.Random(args.string_seed) if args.randomize_string_input else None
    )

    for model_id in model_ids:
        model_url = model_urls[model_id]
        try:
            _ = http_get_json(model_url, token, args.timeout_seconds)
        except Exception as err:  # noqa: BLE001
            print(f"Preflight failed: cannot reach {model_url}: {err}")
            return 2

    original_scaling: Dict[int, Dict[str, Any]] = {}
    frozen_replica_targets: Dict[int, int] = {}
    if args.freeze_scaling:
        try:
            models_snapshot = list_models(base_url, token, args.timeout_seconds)
            for model_id in model_ids:
                scaling_url = f"{base_url}/api/models/{model_id}/scaling"
                policy = http_get_json(scaling_url, token, args.timeout_seconds)
                if not isinstance(policy, dict):
                    raise RuntimeError(
                        f"Invalid scaling policy response for model {model_id}: {type(policy).__name__}"
                    )
                original_scaling[model_id] = dict(policy)

                active_replicas = count_active_replicas(models_snapshot, model_id)
                if active_replicas <= 0:
                    active_replicas = 1
                frozen_replica_targets[model_id] = active_replicas

                frozen_policy = dict(policy)
                frozen_policy["min_replicas"] = active_replicas
                frozen_policy["max_replicas"] = active_replicas
                http_post_json(scaling_url, token, frozen_policy, args.timeout_seconds)
        except Exception as err:  # noqa: BLE001
            if original_scaling:
                try:
                    for mid, policy in original_scaling.items():
                        scaling_url = f"{base_url}/api/models/{mid}/scaling"
                        http_post_json(scaling_url, token, policy, args.timeout_seconds)
                except Exception:
                    pass
            msg = f"Failed to freeze scaling policy: {err}"
            if args.freeze_scaling_strict:
                print(msg)
                return 2
            print(f"Warning: {msg}")
            original_scaling = {}
            frozen_replica_targets = {}

    def run_probe() -> int:
        try:
            pid = resolve_kapsl_pid(args.pid, args.process_pattern)
        except Exception as err:  # noqa: BLE001
            print(f"Failed to resolve kapsl PID: {err}")
            return 2

        payload_desc_by_model = {
            mid: payload_files_by_model.get(
                mid, args.payload_file or "<built-in-mnist-default>"
            )
            for mid in model_ids
        }
        payload_bodies_by_model = {
            mid: build_payload_body(
                payloads_by_model[mid],
                force_cpu=args.force_cpu,
                request_id=(
                    "frag-static-request-id"
                    if args.unique_request_id_per_request
                    else None
                ),
            )
            for mid in model_ids
        }
        payload_bytes_by_model = {
            mid: len(payload_bodies_by_model[mid]) for mid in model_ids
        }

        print("Memory Fragmentation Probe")
        print(f"  model_ids: {model_ids}")
        print(f"  infer_urls: {infer_urls}")
        print(f"  model_urls: {model_urls}")
        print(f"  payload_by_model: {payload_desc_by_model}")
        print(f"  payload_bytes_by_model: {payload_bytes_by_model}")
        print(f"  pid: {pid}")
        print(f"  cycles: {args.cycles}")
        print(f"  concurrency: {args.concurrency}")
        print(f"  requests_per_worker: {args.requests_per_worker}")
        print(f"  total_requests_per_cycle: {total_requests}")
        print(f"  force_cpu: {args.force_cpu}")
        print(f"  unique_request_id_per_request: {args.unique_request_id_per_request}")
        print(f"  randomize_string_input: {args.randomize_string_input}")
        if args.randomize_string_input:
            print(
                f"  string_bytes_range: [{args.string_min_bytes}, {args.string_max_bytes}]"
            )
            print(f"  string_seed: {args.string_seed}")
        print(f"  fail_threshold_pct: {args.fail_threshold_pct}")
        print(f"  freeze_scaling: {args.freeze_scaling}")
        if frozen_replica_targets:
            print(f"  frozen_replicas_by_model: {frozen_replica_targets}")
        print(f"  token: {'set' if token else 'not set'}")

        results: List[CycleResult] = []
        all_errors: List[str] = []

        for cycle in range(1, args.cycles + 1):
            (
                elapsed,
                ok,
                failed,
                p50,
                p95,
                errors,
                per_model_successes,
                per_model_failures,
            ) = run_burst(
                infer_urls=infer_urls,
                payloads_by_model=payloads_by_model,
                payload_bodies_by_model=payload_bodies_by_model,
                model_ids=model_ids,
                token=token,
                timeout_seconds=args.timeout_seconds,
                concurrency=args.concurrency,
                requests_per_worker=args.requests_per_worker,
                force_cpu=args.force_cpu,
                unique_request_id_per_request=args.unique_request_id_per_request,
                randomize_string_input=args.randomize_string_input,
                string_rng=string_rng,
                string_min_bytes=args.string_min_bytes,
                string_max_bytes=args.string_max_bytes,
            )
            all_errors.extend(errors)

            try:
                idle_states = wait_for_idle_models(
                    model_urls=model_urls,
                    model_ids=model_ids,
                    token=token,
                    timeout_seconds=args.idle_timeout_seconds,
                    poll_interval=args.idle_poll_interval,
                    stable_samples=args.idle_stable_samples,
                )
            except Exception as err:  # noqa: BLE001
                print(f"\nCycle {cycle}: failed while waiting for idle: {err}")
                return 2

            try:
                models_snapshot = list_models(base_url, token, args.timeout_seconds)
                idle_replicas_by_id = {
                    mid: count_active_replicas(models_snapshot, mid)
                    for mid in model_ids
                }
            except Exception:
                idle_replicas_by_id = {mid: 0 for mid in model_ids}

            try:
                rss_kb = read_rss_kb(pid)
            except Exception as err:  # noqa: BLE001
                print(f"\nCycle {cycle}: failed to read RSS for PID {pid}: {err}")
                return 2

            memory_by_id = {
                model_id: int(idle_states.get(model_id, {}).get("memory_usage", 0))
                for model_id in model_ids
            }
            memory_usage_total = sum(memory_by_id.values())

            row = CycleResult(
                cycle=cycle,
                burst_elapsed_s=elapsed,
                successes=ok,
                failures=failed,
                p50_ms=p50,
                p95_ms=p95,
                idle_rss_kb=rss_kb,
                idle_total_model_memory_bytes=memory_usage_total,
                idle_model_memory_by_id=memory_by_id,
                idle_active_replicas_by_base_id=idle_replicas_by_id,
                per_model_successes=per_model_successes,
                per_model_failures=per_model_failures,
            )
            results.append(row)

            per_model_ok_str = ",".join(
                f"{mid}:{per_model_successes[mid]}/{per_model_successes[mid] + per_model_failures[mid]}"
                for mid in model_ids
            )
            per_model_mem_str = ",".join(
                f"{mid}:{memory_by_id[mid]}" for mid in model_ids
            )
            per_model_replica_str = ",".join(
                f"{mid}:{idle_replicas_by_id.get(mid, 0)}" for mid in model_ids
            )

            print(
                f"  cycle={cycle:02d} "
                f"ok={ok}/{total_requests} "
                f"burst_s={elapsed:.2f} "
                f"p50={p50:.2f}ms "
                f"idle_rss={rss_kb}KB "
                f"idle_model_mem_total={memory_usage_total}B "
                f"replicas=[{per_model_replica_str}] "
                f"per_model_ok=[{per_model_ok_str}] "
                f"per_model_mem=[{per_model_mem_str}]"
            )

        if args.csv_output:
            write_csv(args.csv_output, results)
            print(f"\nWrote CSV: {args.csv_output}")

        baseline = results[0].idle_rss_kb
        max_idle = max(r.idle_rss_kb for r in results)
        last_idle = results[-1].idle_rss_kb
        max_drift_pct = (
            ((max_idle - baseline) / baseline * 100.0) if baseline > 0 else 0.0
        )
        final_drift_pct = (
            ((last_idle - baseline) / baseline * 100.0) if baseline > 0 else 0.0
        )

        baseline_model_mem = results[0].idle_total_model_memory_bytes
        max_model_mem = max(r.idle_total_model_memory_bytes for r in results)
        last_model_mem = results[-1].idle_total_model_memory_bytes

        print("\nSummary")
        print(f"  baseline_idle_rss_kb: {baseline}")
        print(f"  max_idle_rss_kb: {max_idle}")
        print(f"  last_idle_rss_kb: {last_idle}")
        print(f"  max_idle_drift_pct: {max_drift_pct:.2f}")
        print(f"  final_idle_drift_pct: {final_drift_pct:.2f}")
        print(f"  baseline_idle_model_mem_bytes: {baseline_model_mem}")
        print(f"  max_idle_model_mem_bytes: {max_model_mem}")
        print(f"  last_idle_model_mem_bytes: {last_model_mem}")
        print(f"  total_failures: {sum(r.failures for r in results)}")

        if all_errors:
            print("\nSample request errors:")
            for err in all_errors[:5]:
                print(f"  - {err}")

        if max_drift_pct > args.fail_threshold_pct:
            print(
                f"\nFAIL: max idle RSS drift {max_drift_pct:.2f}% exceeds threshold "
                f"{args.fail_threshold_pct:.2f}%."
            )
            return 1

        print("\nPASS: idle RSS drift is within threshold.")
        return 0

    try:
        return run_probe()
    finally:
        if args.freeze_scaling and original_scaling:
            try:
                for mid, policy in original_scaling.items():
                    scaling_url = f"{base_url}/api/models/{mid}/scaling"
                    http_post_json(scaling_url, token, policy, args.timeout_seconds)
            except Exception as err:  # noqa: BLE001
                msg = f"Failed to restore scaling policy: {err}"
                if args.freeze_scaling_strict:
                    print(msg)
                    raise SystemExit(2) from err
                print(f"Warning: {msg}")


if __name__ == "__main__":
    raise SystemExit(main())
