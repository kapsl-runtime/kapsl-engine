#!/usr/bin/env python3
import argparse
import base64
import copy
import json
import math
import random
import statistics
import struct
import time
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import requests  # type: ignore

    HAS_REQUESTS = True
except Exception:
    import urllib.error
    import urllib.request

    HAS_REQUESTS = False


def http_get(url, timeout):
    if HAS_REQUESTS:
        try:
            resp = requests.get(url, timeout=timeout)
            return resp.status_code, resp.text
        except Exception as exc:
            return None, str(exc)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")
    except Exception as exc:
        return None, str(exc)


def http_post(url, payload, timeout):
    if HAS_REQUESTS:
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            return resp.status_code, resp.text
        except Exception as exc:
            return None, str(exc)

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")
    except Exception as exc:
        return None, str(exc)


def parse_shape(shape_text):
    if not shape_text:
        return None
    parts = [p.strip() for p in shape_text.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_string_payload(prompt, max_tokens=None, temperature=None):
    payload = {
        "input": {
            "shape": [1, 1],
            "dtype": "string",
            "data_base64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        },
        "session_id": None,
    }
    metadata = {}
    if max_tokens is not None:
        metadata["max_tokens"] = int(max_tokens)
    if temperature is not None:
        metadata["temperature"] = float(temperature)
    if metadata:
        payload["metadata"] = metadata
    return payload


def build_float_payload(shape, rng):
    count = 1
    for dim in shape:
        count *= dim

    values = [rng.uniform(-1.0, 1.0) for _ in range(count)]
    packed = b"".join(struct.pack("<f", v) for v in values)
    return {
        "input": {
            "shape": shape,
            "dtype": "float32",
            "data_base64": base64.b64encode(packed).decode("ascii"),
        },
        "session_id": None,
    }


def with_kapsl_llm_metadata(payload, max_tokens=None, temperature=None):
    input_obj = payload.get("input") if isinstance(payload, dict) else None
    if not isinstance(input_obj, dict):
        return payload
    if input_obj.get("dtype") != "string":
        return payload

    metadata = {}
    if max_tokens is not None:
        metadata["max_tokens"] = int(max_tokens)
    if temperature is not None:
        metadata["temperature"] = float(temperature)
    if not metadata:
        return payload

    out = copy.deepcopy(payload)
    existing_metadata = out.get("metadata")
    if not isinstance(existing_metadata, dict):
        existing_metadata = {}
    existing_metadata.update(metadata)
    out["metadata"] = existing_metadata
    return out


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize_latencies(latencies_ms):
    if not latencies_ms:
        return {}
    values = sorted(latencies_ms)
    return {
        "min": values[0],
        "max": values[-1],
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def parse_prom_metrics(text):
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(" ")
        if len(parts) < 2:
            continue

        name_with_labels = parts[0]
        value_str = parts[-1]
        try:
            value = float(value_str)
        except ValueError:
            continue

        if "{" in name_with_labels:
            name, label_part = name_with_labels.split("{", 1)
            label_part = label_part.rstrip("}")
            labels = {}
            if label_part:
                for item in label_part.split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        labels[k.strip()] = v.strip().strip('"')
        else:
            name = name_with_labels
            labels = {}

        metrics.setdefault(name, []).append({"labels": labels, "value": value})
    return metrics


def sum_metric(metrics, names):
    for name in names:
        if name in metrics:
            return sum(item["value"] for item in metrics[name])
    return None


def collect_kapsl_snapshot(base_url, timeout):
    status, body = http_get(f"{base_url}/api/models", timeout)
    if status != 200:
        return {"error": f"Failed to fetch models: {status} {body}"}

    try:
        models = json.loads(body)
    except Exception as exc:
        return {"error": f"Invalid /api/models JSON: {exc}"}

    runtime_rss_bytes = None
    gpu_memory_bytes = None
    system_gpu_utilization = None
    stats_status, stats_body = http_get(f"{base_url}/api/system/stats", timeout)
    if stats_status == 200:
        try:
            stats = json.loads(stats_body)
            runtime_rss_bytes = int(stats.get("process_memory_bytes", 0))
            gpu_memory_bytes = stats.get("gpu_memory_bytes", None)
            if gpu_memory_bytes is not None:
                gpu_memory_bytes = int(gpu_memory_bytes)
            system_gpu_utilization = float(stats.get("gpu_utilization", 0.0))
        except Exception:
            # Keep snapshot usable even if system stats payload is missing/old runtime.
            runtime_rss_bytes = None
            gpu_memory_bytes = None
            system_gpu_utilization = None

    total_memory = 0
    max_memory = 0
    total_throughput = 0.0
    avg_gpu = 0.0
    for model in models:
        mem = int(model.get("memory_usage", 0))
        total_memory += mem
        max_memory = max(max_memory, mem)
        total_throughput += float(model.get("throughput", 0.0))
        avg_gpu += float(model.get("gpu_utilization", 0.0))

    avg_gpu = avg_gpu / len(models) if models else 0.0
    fragmentation_est = None
    if total_memory > 0:
        fragmentation_est = 1.0 - (max_memory / total_memory)

    return {
        "models": len(models),
        "total_memory_bytes": total_memory,
        "max_model_memory_bytes": max_memory,
        "fragmentation_estimate": fragmentation_est,
        "runtime_rss_bytes": runtime_rss_bytes,
        "gpu_memory_bytes": gpu_memory_bytes,
        "system_gpu_utilization": system_gpu_utilization,
        "total_throughput": total_throughput,
        "avg_gpu_utilization": avg_gpu,
    }


def collect_vllm_snapshot(metrics_url, timeout, mem_used_names, mem_total_names):
    status, body = http_get(metrics_url, timeout)
    if status != 200:
        return {"error": f"Failed to fetch metrics: {status} {body}"}

    metrics = parse_prom_metrics(body)
    mem_used = sum_metric(metrics, mem_used_names)
    mem_total = sum_metric(metrics, mem_total_names)

    fragmentation_proxy = None
    if mem_used is not None and mem_total:
        fragmentation_proxy = 1.0 - (mem_used / mem_total)

    return {
        "memory_used_bytes": mem_used,
        "memory_total_bytes": mem_total,
        "fragmentation_proxy": fragmentation_proxy,
    }


def run_requests(url, payload_factory, total_requests, concurrency, timeout):
    latencies = []
    errors = 0

    def run_one():
        payload = payload_factory()
        start = time.perf_counter()
        status, body = http_post(url, payload, timeout)
        elapsed = (time.perf_counter() - start) * 1000.0
        if status != 200:
            return False, elapsed, status, body
        if is_kapsl_error_payload(body):
            return False, elapsed, status, body
        return True, elapsed, status, body

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_one) for _ in range(total_requests)]
        for future in as_completed(futures):
            ok, latency_ms, _, _ = future.result()
            if ok:
                latencies.append(latency_ms)
            else:
                errors += 1
    elapsed_total = time.perf_counter() - start_total

    ok_count = total_requests - errors
    throughput = ok_count / elapsed_total if elapsed_total > 0 else 0.0

    return {
        "requests": total_requests,
        "ok": ok_count,
        "errors": errors,
        "duration_sec": elapsed_total,
        "throughput_rps": throughput,
        "latency_ms": summarize_latencies(latencies),
    }


def run_kapsl_sessions(url, prompt, num_sessions, turns_per_session, max_tokens, temperature, timeout):
    """
    Multi-turn session benchmark against kapsl.

    Each session sends `turns_per_session` sequential requests with the same session_id.
    Kapsl reuses the KV cache across turns — only new tokens are processed after turn 1.
    Sessions run concurrently; turns within each session are sequential.
    """
    import uuid

    per_turn_latencies = [[] for _ in range(turns_per_session)]
    all_latencies = []
    total_errors = 0

    def run_session(session_idx):
        session_id = f"bench-{uuid.uuid4().hex[:8]}-{session_idx}"
        sess_latencies = []
        sess_errors = 0
        for turn in range(turns_per_session):
            payload = build_string_payload(prompt, max_tokens, temperature)
            payload["session_id"] = session_id
            start = time.perf_counter()
            status, body = http_post(url, payload, timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if status != 200 or is_kapsl_error_payload(body):
                sess_errors += 1
                break
            sess_latencies.append((turn, elapsed_ms))
        return sess_latencies, sess_errors

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = [executor.submit(run_session, i) for i in range(num_sessions)]
        for future in as_completed(futures):
            sess_latencies, sess_errors = future.result()
            total_errors += sess_errors
            for turn_idx, lat_ms in sess_latencies:
                per_turn_latencies[turn_idx].append(lat_ms)
                all_latencies.append(lat_ms)
    elapsed_total = time.perf_counter() - start_total

    total_requests = num_sessions * turns_per_session
    ok_count = total_requests - total_errors
    return {
        "requests": total_requests,
        "ok": ok_count,
        "errors": total_errors,
        "duration_sec": elapsed_total,
        "throughput_rps": ok_count / elapsed_total if elapsed_total > 0 else 0.0,
        "latency_ms": summarize_latencies(sorted(all_latencies)),
        "per_turn_latency_ms": {
            f"turn_{i + 1}": summarize_latencies(sorted(lats))
            for i, lats in enumerate(per_turn_latencies)
            if lats
        },
    }


def run_vllm_sessions(url, model_name, prompt, num_sessions, turns_per_session, max_tokens, temperature, timeout):
    """
    Multi-turn session benchmark against vLLM / llama-cpp-python.

    Simulates a real chat client: the full message history (user + assistant turns) is
    re-sent on every request. This is the standard OpenAI chat completions pattern and
    means the server must re-encode the growing context each turn.
    """
    per_turn_latencies = [[] for _ in range(turns_per_session)]
    all_latencies = []
    total_errors = 0

    def run_session(_session_idx):
        messages = []
        sess_latencies = []
        sess_errors = 0
        for turn in range(turns_per_session):
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            start = time.perf_counter()
            status, body = http_post(url, payload, timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if status != 200:
                sess_errors += 1
                break
            try:
                resp = json.loads(body)
                assistant_text = resp["choices"][0]["message"]["content"]
            except Exception:
                assistant_text = "..."
            messages.append({"role": "assistant", "content": assistant_text})
            sess_latencies.append((turn, elapsed_ms))
        return sess_latencies, sess_errors

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = [executor.submit(run_session, i) for i in range(num_sessions)]
        for future in as_completed(futures):
            sess_latencies, sess_errors = future.result()
            total_errors += sess_errors
            for turn_idx, lat_ms in sess_latencies:
                per_turn_latencies[turn_idx].append(lat_ms)
                all_latencies.append(lat_ms)
    elapsed_total = time.perf_counter() - start_total

    total_requests = num_sessions * turns_per_session
    ok_count = total_requests - total_errors
    return {
        "requests": total_requests,
        "ok": ok_count,
        "errors": total_errors,
        "duration_sec": elapsed_total,
        "throughput_rps": ok_count / elapsed_total if elapsed_total > 0 else 0.0,
        "latency_ms": summarize_latencies(sorted(all_latencies)),
        "per_turn_latency_ms": {
            f"turn_{i + 1}": summarize_latencies(sorted(lats))
            for i, lats in enumerate(per_turn_latencies)
            if lats
        },
    }


def print_session_summary(label, result):
    latency = result.get("latency_ms", {})
    per_turn = result.get("per_turn_latency_ms", {})
    print(f"\n{label} session results")
    print(
        f"  sessions x turns: {result.get('requests')} total"
        f"  ok: {result.get('ok')}  errors: {result.get('errors')}"
    )
    print(f"  throughput: {result.get('throughput_rps'):.3f} req/s")
    if latency:
        print(
            f"  overall latency ms: p50={latency.get('p50'):.1f}"
            f"  p90={latency.get('p90'):.1f}"
            f"  mean={latency.get('mean'):.1f}"
        )
    if per_turn:
        print("  per-turn p50 latency ms:")
        for turn_key in sorted(per_turn.keys()):
            t = per_turn[turn_key]
            if t:
                print(f"    {turn_key}: p50={t.get('p50'):.1f}  mean={t.get('mean'):.1f}")


def is_kapsl_error_payload(body_text):
    try:
        payload = json.loads(body_text)
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False

    if payload.get("dtype") != "string":
        return False
    data = payload.get("data")
    if not isinstance(data, list):
        return False

    try:
        snippet = bytes(int(x) & 0xFF for x in data[:1024]).decode(
            "utf-8", errors="ignore"
        )
    except Exception:
        return False

    return snippet.lstrip().lower().startswith("llm execution error:")


def read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def format_bytes(value):
    if value is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def print_summary(label, result, snapshot):
    latency = result.get("latency_ms", {})
    print(f"\n{label} results")
    print(
        f"  requests: {result.get('requests')} ok: {result.get('ok')} errors: {result.get('errors')}"
    )
    print(f"  throughput: {result.get('throughput_rps'):.2f} req/s")
    if latency:
        print(
            "  latency ms: "
            f"p50={latency.get('p50'):.2f} p90={latency.get('p90'):.2f} "
            f"p99={latency.get('p99'):.2f} mean={latency.get('mean'):.2f}"
        )
    if snapshot:
        if "total_memory_bytes" in snapshot:
            print(f"  memory used: {format_bytes(snapshot.get('total_memory_bytes'))}")
            frag = snapshot.get("fragmentation_estimate")
            if frag is not None:
                print(f"  fragmentation est.: {frag * 100:.1f}%")
        if "memory_used_bytes" in snapshot:
            print(f"  memory used: {format_bytes(snapshot.get('memory_used_bytes'))}")
            frag = snapshot.get("fragmentation_proxy")
            if frag is not None:
                print(f"  fragmentation proxy: {frag * 100:.1f}%")


def print_comparison(kapsl_result, vllm_result):
    if not kapsl_result or not vllm_result:
        return

    kapsl_tp = kapsl_result.get("throughput_rps", 0.0)
    vllm_tp = vllm_result.get("throughput_rps", 0.0)
    tp_ratio = (kapsl_tp / vllm_tp) if vllm_tp else None

    kapsl_lat = kapsl_result.get("latency_ms", {})
    vllm_lat = vllm_result.get("latency_ms", {})

    def ratio(a, b):
        return (a / b) if a is not None and b else None

    print("\nComparison (kapsl vs vllm)")
    if tp_ratio is not None:
        print(f"  throughput ratio: {tp_ratio:.2f}x")
    if kapsl_lat and vllm_lat:
        p50_ratio = ratio(kapsl_lat.get("p50"), vllm_lat.get("p50"))
        p90_ratio = ratio(kapsl_lat.get("p90"), vllm_lat.get("p90"))
        p99_ratio = ratio(kapsl_lat.get("p99"), vllm_lat.get("p99"))
        if p50_ratio is not None:
            print(f"  p50 latency ratio: {p50_ratio:.2f}x")
        if p90_ratio is not None:
            print(f"  p90 latency ratio: {p90_ratio:.2f}x")
        if p99_ratio is not None:
            print(f"  p99 latency ratio: {p99_ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark kapsl-runtime and vLLM with comparable request loads."
    )
    parser.add_argument("--kapsl-url", default="http://localhost:9095")
    parser.add_argument("--kapsl-model-id", type=int, action="append")
    parser.add_argument("--kapsl-model-ids")
    parser.add_argument("--kapsl-payload-file")
    parser.add_argument("--tensor-shape", default="1,1,28,28")

    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--vllm-model", action="append")
    parser.add_argument("--vllm-models")
    parser.add_argument(
        "--vllm-endpoint",
        choices=["chat", "completions", "generate"],
        default="chat",
    )
    parser.add_argument("--vllm-payload-file")

    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)

    # Session / multi-turn mode
    # When --num-sessions > 0, runs concurrent multi-turn sessions instead of
    # independent requests. kapsl uses session_id for KV cache reuse; the
    # comparison backend re-sends the full message history each turn.
    parser.add_argument("--num-sessions", type=int, default=0,
                        help="Number of concurrent multi-turn sessions (0 = flat mode)")
    parser.add_argument("--turns-per-session", type=int, default=5,
                        help="Sequential turns per session (default: 5)")

    parser.add_argument(
        "--vllm-mem-used-metric",
        action="append",
        default=[
            "vllm_gpu_memory_usage_bytes",
            "vllm_gpu_memory_used_bytes",
            "vllm:gpu_memory_used_bytes",
        ],
    )
    parser.add_argument(
        "--vllm-mem-total-metric",
        action="append",
        default=[
            "vllm_gpu_memory_total_bytes",
            "vllm:gpu_memory_total_bytes",
        ],
    )

    args = parser.parse_args()

    prompt = None
    if args.prompt_file:
        prompt = read_text(args.prompt_file)
    elif args.prompt:
        prompt = args.prompt

    rng = random.Random(args.seed)
    results = {}

    def vllm_endpoint_path(kind):
        if kind == "completions":
            return "v1/completions"
        if kind == "generate":
            return "generate"
        return "v1/chat/completions"

    kapsl_ids = []
    if args.kapsl_model_id:
        kapsl_ids.extend(args.kapsl_model_id)
    if args.kapsl_model_ids:
        for item in parse_list(args.kapsl_model_ids):
            kapsl_ids.append(int(item))
    if kapsl_ids:
        if args.kapsl_payload_file:
            kapsl_payload = with_kapsl_llm_metadata(
                load_json(args.kapsl_payload_file), args.max_tokens, args.temperature
            )
            payload_factory = lambda: copy.deepcopy(kapsl_payload)
        elif prompt:
            payload_factory = lambda: build_string_payload(
                prompt, args.max_tokens, args.temperature
            )
        else:
            shape = parse_shape(args.tensor_shape) or [1, 1, 28, 28]
            float_payload = build_float_payload(shape, rng)
            payload_factory = lambda: float_payload

        kapsl_snapshot = collect_kapsl_snapshot(args.kapsl_url, args.timeout)
        results["kapsl"] = {"models": {}, "snapshot": kapsl_snapshot}
        for model_id in kapsl_ids:
            kapsl_url = f"{args.kapsl_url}/api/models/{model_id}/infer"

            if args.num_sessions > 0 and prompt:
                # Multi-turn session mode: warmup with one flat request first
                for _ in range(max(args.warmup, 0)):
                    http_post(kapsl_url, payload_factory(), args.timeout)
                kapsl_result = run_kapsl_sessions(
                    kapsl_url,
                    prompt,
                    args.num_sessions,
                    args.turns_per_session,
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )
                results["kapsl"]["models"][str(model_id)] = {"benchmark": kapsl_result}
                print_session_summary(f"kapsl-runtime (model {model_id})", kapsl_result)
            else:
                for _ in range(max(args.warmup, 0)):
                    http_post(kapsl_url, payload_factory(), args.timeout)
                kapsl_result = run_requests(
                    kapsl_url,
                    payload_factory,
                    args.requests,
                    args.concurrency,
                    args.timeout,
                )
                results["kapsl"]["models"][str(model_id)] = {"benchmark": kapsl_result}
                print_summary(
                    f"kapsl-runtime (model {model_id})", kapsl_result, kapsl_snapshot
                )

    vllm_models = []
    if args.vllm_model:
        vllm_models.extend(args.vllm_model)
    if args.vllm_models:
        vllm_models.extend(parse_list(args.vllm_models))

    if vllm_models or args.vllm_payload_file:
        results["vllm"] = {"models": {}}
        endpoint = vllm_endpoint_path(args.vllm_endpoint)
        if args.vllm_payload_file:
            vllm_payload = load_json(args.vllm_payload_file)
            payload_factory = lambda: vllm_payload
            vllm_url = f"{args.vllm_url}/{endpoint}"

            for _ in range(max(args.warmup, 0)):
                http_post(vllm_url, payload_factory(), args.timeout)

            vllm_result = run_requests(
                vllm_url,
                payload_factory,
                args.requests,
                args.concurrency,
                args.timeout,
            )
            metrics_url = f"{args.vllm_url}/metrics"
            vllm_snapshot = collect_vllm_snapshot(
                metrics_url,
                args.timeout,
                args.vllm_mem_used_metric,
                args.vllm_mem_total_metric,
            )
            results["vllm"]["models"]["payload_file"] = {
                "benchmark": vllm_result,
                "snapshot": vllm_snapshot,
            }
            print_summary("vllm (payload file)", vllm_result, vllm_snapshot)
        else:
            prompt_value = prompt or "Write a short summary about runtime benchmarking."
            if args.vllm_endpoint == "generate" and len(vllm_models) > 1:
                print("Note: vLLM generate endpoint ignores model; running once.")
                vllm_models = [vllm_models[0]]

            for model_name in vllm_models:
                vllm_url = f"{args.vllm_url}/{endpoint}"

                if args.num_sessions > 0 and args.vllm_endpoint == "chat":
                    # Multi-turn session mode: full message history re-sent each turn
                    single_payload = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt_value}],
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                    }
                    for _ in range(max(args.warmup, 0)):
                        http_post(vllm_url, single_payload, args.timeout)
                    vllm_result = run_vllm_sessions(
                        vllm_url,
                        model_name,
                        prompt_value,
                        args.num_sessions,
                        args.turns_per_session,
                        args.max_tokens,
                        args.temperature,
                        args.timeout,
                    )
                    metrics_url = f"{args.vllm_url}/metrics"
                    vllm_snapshot = collect_vllm_snapshot(
                        metrics_url, args.timeout,
                        args.vllm_mem_used_metric, args.vllm_mem_total_metric,
                    )
                    results["vllm"]["models"][model_name] = {
                        "benchmark": vllm_result,
                        "snapshot": vllm_snapshot,
                    }
                    print_session_summary(f"vllm ({model_name})", vllm_result)
                else:
                    if args.vllm_endpoint == "completions":
                        vllm_payload = {
                            "model": model_name,
                            "prompt": prompt_value,
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                        }
                    elif args.vllm_endpoint == "generate":
                        vllm_payload = {
                            "prompt": prompt_value,
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                        }
                    else:
                        vllm_payload = {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt_value}],
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                        }

                    payload_factory = lambda: vllm_payload
                    for _ in range(max(args.warmup, 0)):
                        http_post(vllm_url, payload_factory(), args.timeout)

                    vllm_result = run_requests(
                        vllm_url,
                        payload_factory,
                        args.requests,
                        args.concurrency,
                        args.timeout,
                    )
                    metrics_url = f"{args.vllm_url}/metrics"
                    vllm_snapshot = collect_vllm_snapshot(
                        metrics_url,
                        args.timeout,
                        args.vllm_mem_used_metric,
                        args.vllm_mem_total_metric,
                    )
                    results["vllm"]["models"][model_name] = {
                        "benchmark": vllm_result,
                        "snapshot": vllm_snapshot,
                    }
                    print_summary(f"vllm ({model_name})", vllm_result, vllm_snapshot)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    if "kapsl" in results and "vllm" in results:
        kapsl_models = results.get("kapsl", {}).get("models", {})
        vllm_models = results.get("vllm", {}).get("models", {})
        if len(kapsl_models) == 1 and len(vllm_models) == 1:
            kapsl_key = next(iter(kapsl_models))
            vllm_key = next(iter(vllm_models))
            print_comparison(
                kapsl_models[kapsl_key]["benchmark"],
                vllm_models[vllm_key]["benchmark"],
            )
        else:
            print(
                "\nComparison skipped (multiple models). See output JSON for details."
            )


if __name__ == "__main__":
    main()
