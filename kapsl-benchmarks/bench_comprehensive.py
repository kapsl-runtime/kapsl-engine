#!/usr/bin/env python3
"""Comprehensive kapsl vs vLLM benchmark with tok/s, TTFT, and multi-concurrency."""

import argparse
import base64
import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as req_lib

# ---------------------------------------------------------------------------
# Prompts — diverse workload
# ---------------------------------------------------------------------------
PROMPTS = [
    "Explain the difference between TCP and UDP in simple terms.",
    "Write a Python function that checks if a string is a palindrome.",
    "Summarize the key principles of object-oriented programming.",
    "What are the main causes of climate change? List them briefly.",
    "Describe how a hash table works and its time complexity.",
    "Write a haiku about machine learning.",
    "Explain the CAP theorem in distributed systems.",
    "What is the difference between a process and a thread?",
    "List three advantages of using Rust over C++ for systems programming.",
    "Explain what a neural network is to a 10-year-old.",
]

FAIR_TASKS = [
    {
        "prompt": 'Return exactly this lowercase word and nothing else: "udp".',
        "validator": "exact",
        "expected": "udp",
    },
    {
        "prompt": 'Return exactly this JSON object and nothing else: {"answer":"tcp"}',
        "validator": "json_exact",
        "expected": {"answer": "tcp"},
    },
    {
        "prompt": "Reply with exactly three comma-separated colors in this order: red,green,blue",
        "validator": "exact",
        "expected": "red,green,blue",
    },
    {
        "prompt": "Return only the integer result of 17+25.",
        "validator": "exact",
        "expected": "42",
    },
    {
        "prompt": "Return only the uppercase word RUST.",
        "validator": "exact",
        "expected": "RUST",
    },
    {
        "prompt": "Answer with yes or no only: is UDP connectionless?",
        "validator": "exact_ci",
        "expected": "yes",
    },
    {
        "prompt": 'Return exactly this JSON object and nothing else: {"palindrome":true}',
        "validator": "json_exact",
        "expected": {"palindrome": True},
    },
    {
        "prompt": "Return only this sequence with no spaces: 2,4,6,8",
        "validator": "exact",
        "expected": "2,4,6,8",
    },
    {
        "prompt": "Return exactly one word: thread",
        "validator": "exact_ci",
        "expected": "thread",
    },
    {
        "prompt": "Return only this text: CAP",
        "validator": "exact",
        "expected": "CAP",
    },
]


def build_prompts(normalize_output_words):
    if not normalize_output_words:
        return PROMPTS
    instruction = (
        f"Answer in exactly {normalize_output_words} words. "
        "Use plain prose. Do not use bullets or numbering."
    )
    return [f"{instruction}\n\n{prompt}" for prompt in PROMPTS]


def build_workload(task_suite, normalize_output_words):
    if task_suite == "fair":
        return FAIR_TASKS
    return [{"prompt": prompt} for prompt in build_prompts(normalize_output_words)]


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summarize(values):
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": s[0],
        "max": s[-1],
        "mean": statistics.mean(s),
        "median": percentile(s, 50),
        "p50": percentile(s, 50),
        "p90": percentile(s, 90),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
    }


def estimate_output_units(text):
    if not text:
        return 0
    return max(1, len(text.split()))


def text_words(text):
    return re.findall(r"[A-Za-z0-9\+\#\-]+", (text or "").lower())


def prompt_keywords(prompt):
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "what",
        "how",
        "is",
        "are",
        "be",
        "as",
        "it",
        "its",
        "if",
        "that",
        "this",
        "them",
        "they",
        "you",
        "your",
        "about",
        "between",
        "three",
        "list",
        "write",
        "explain",
        "describe",
        "difference",
        "simple",
        "terms",
        "briefly",
        "main",
        "key",
    }
    seen = []
    for word in text_words(prompt):
        if len(word) < 4 or word in stop:
            continue
        if word not in seen:
            seen.append(word)
        if len(seen) >= 5:
            break
    return seen


def fidelity_metrics(ok_results, prompts, normalize_output_words):
    prompt_hits = []
    within_target = []
    normalized_outputs = []
    for idx, result in enumerate(ok_results):
        output = (result.get("output_text") or "").strip().lower()
        normalized_outputs.append(output)
        kws = prompt_keywords(prompts[idx % len(prompts)])
        if kws:
            hits = sum(1 for kw in kws if kw in output)
            prompt_hits.append(hits / len(kws))
        if normalize_output_words:
            units = result.get("output_tokens", 0)
            within_target.append(abs(units - normalize_output_words) <= 5)

    repeated = 0
    if normalized_outputs:
        counts = {}
        for output in normalized_outputs:
            counts[output] = counts.get(output, 0) + 1
        repeated = sum(count - 1 for count in counts.values() if count > 1)

    return {
        "prompt_overlap_mean": statistics.mean(prompt_hits) if prompt_hits else None,
        "target_word_compliance_rate": (
            sum(1 for ok in within_target if ok) / len(within_target)
            if within_target
            else None
        ),
        "repeated_output_rate": repeated / len(normalized_outputs)
        if normalized_outputs
        else None,
    }


def validate_fair_output(output_text, task):
    text = (output_text or "").strip()
    validator = task["validator"]
    expected = task["expected"]
    if validator == "exact":
        return text == expected
    if validator == "exact_ci":
        return text.lower() == str(expected).lower()
    if validator == "json_exact":
        try:
            parsed = json.loads(text)
        except Exception:
            return False
        return parsed == expected
    return False


def fair_metrics(ok_results, tasks):
    if not ok_results:
        return {
            "pass_rate": 0.0,
            "repeat_fail_rate": None,
            "exact_match_rate": 0.0,
        }
    passes = []
    normalized_outputs = []
    for idx, result in enumerate(ok_results):
        output = (result.get("output_text") or "").strip()
        normalized_outputs.append(output)
        passes.append(validate_fair_output(output, tasks[idx % len(tasks)]))
    repeated = 0
    counts = {}
    for output in normalized_outputs:
        counts[output] = counts.get(output, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    pass_count = sum(1 for ok in passes if ok)
    return {
        "pass_rate": pass_count / len(ok_results),
        "exact_match_rate": pass_count / len(ok_results),
        "repeat_fail_rate": repeated / len(ok_results),
    }


# ---------------------------------------------------------------------------
# vLLM: streaming request for TTFT + ITL + token counting
# ---------------------------------------------------------------------------
def vllm_streaming_request(url, model, prompt, max_tokens, temperature, timeout):
    """Send a streaming chat completion and measure TTFT, ITL, total latency, token count."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t_start = time.perf_counter()
    ttft = None
    token_times = []
    output_text = []
    error = None

    try:
        resp = req_lib.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content is not None:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = (now - t_start) * 1000.0
                    token_times.append(now)
                    output_text.append(content)
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        error = str(exc)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0
    output_text_joined = "".join(output_text)
    output_tokens = estimate_output_units(output_text_joined)

    # Inter-token latencies
    itl_values = []
    for i in range(1, len(token_times)):
        itl_values.append((token_times[i] - token_times[i - 1]) * 1000.0)

    if error:
        return {"ok": False, "error": error}

    return {
        "ok": True,
        "total_ms": total_ms,
        "ttft_ms": ttft,
        "output_tokens": output_tokens,
        "itl_ms": itl_values,
        "tok_per_sec": output_tokens / (total_ms / 1000.0) if total_ms > 0 else 0,
        "output_text": output_text_joined,
    }


# ---------------------------------------------------------------------------
# vLLM: non-streaming request (for token count from usage field)
# ---------------------------------------------------------------------------
def vllm_request(url, model, prompt, max_tokens, temperature, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t_start = time.perf_counter()
    try:
        resp = req_lib.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
        total_ms = (time.perf_counter() - t_start) * 1000.0
        if resp.status_code != 200:
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}",
                "total_ms": total_ms,
            }
        body = resp.json()
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        output_text = ""
        try:
            output_text = body["choices"][0]["message"]["content"]
        except Exception:
            output_text = ""
        completion_tokens = estimate_output_units(output_text)
        return {
            "ok": True,
            "total_ms": total_ms,
            "prompt_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "tok_per_sec": completion_tokens / (total_ms / 1000.0)
            if total_ms > 0
            else 0,
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return {"ok": False, "error": str(exc), "total_ms": total_ms}


# ---------------------------------------------------------------------------
# kapsl: request with token estimation
# ---------------------------------------------------------------------------
def kapsl_request(url, model_id, prompt, max_tokens, temperature, timeout):
    prompt_bytes = prompt.encode()
    payload = {
        "input": {
            "shape": [1, len(prompt_bytes)],
            "dtype": "string",
            "data_base64": base64.b64encode(prompt_bytes).decode(),
        },
        "session_id": None,
        "metadata": {"max_new_tokens": max_tokens, "temperature": temperature},
    }
    t_start = time.perf_counter()
    try:
        resp = req_lib.post(
            f"{url}/api/models/{model_id}/infer", json=payload, timeout=timeout
        )
        total_ms = (time.perf_counter() - t_start) * 1000.0
        if resp.status_code != 200:
            return {
                "ok": False,
                "error": f"HTTP {resp.status_code}",
                "total_ms": total_ms,
            }
        body = resp.json()
        # Extract output text from kapsl response
        output_text = ""
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list):
                try:
                    output_text = bytes(int(x) & 0xFF for x in data).decode(
                        "utf-8", errors="ignore"
                    )
                except Exception:
                    pass
            elif isinstance(data, str):
                output_text = data
            # Check data_base64
            data_b64 = body.get("data_base64")
            if data_b64 and not output_text:
                try:
                    output_text = base64.b64decode(data_b64).decode(
                        "utf-8", errors="ignore"
                    )
                except Exception:
                    pass
        est_tokens = estimate_output_units(output_text)
        return {
            "ok": True,
            "total_ms": total_ms,
            "output_tokens": est_tokens,
            "tok_per_sec": est_tokens / (total_ms / 1000.0) if total_ms > 0 else 0,
            "output_text": output_text[:200],
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return {"ok": False, "error": str(exc), "total_ms": total_ms}


# ---------------------------------------------------------------------------
# kapsl streaming: measure TTFT via SSE /infer/stream endpoint
# ---------------------------------------------------------------------------
def kapsl_streaming_request(url, model_id, prompt, max_tokens, temperature, timeout):
    prompt_bytes = prompt.encode()
    payload = {
        "input": {
            "shape": [1, len(prompt_bytes)],
            "dtype": "string",
            "data_base64": base64.b64encode(prompt_bytes).decode(),
        },
        "session_id": None,
        "metadata": {"max_new_tokens": max_tokens, "temperature": temperature},
    }
    t_start = time.perf_counter()
    ttft = None
    token_times = []
    output_text = []

    try:
        resp = req_lib.post(
            f"{url}/api/models/{model_id}/infer/stream",
            json=payload,
            stream=True,
            timeout=timeout,
        )
        if resp.status_code != 200:
            # Fallback: streaming not supported, use non-streaming
            return kapsl_request(
                url, model_id, prompt, max_tokens, temperature, timeout
            )

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = (now - t_start) * 1000.0
                    token_times.append(now)
                    text = ""
                    if isinstance(chunk, dict):
                        text = chunk.get("text", chunk.get("token", ""))
                    output_text.append(str(text))
                except json.JSONDecodeError:
                    continue
    except Exception:
        # Streaming not supported, fallback
        return kapsl_request(url, model_id, prompt, max_tokens, temperature, timeout)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0
    output_text_joined = "".join(output_text)
    output_tokens = estimate_output_units(output_text_joined)

    if output_tokens == 0:
        return kapsl_request(url, model_id, prompt, max_tokens, temperature, timeout)

    itl_values = []
    for i in range(1, len(token_times)):
        itl_values.append((token_times[i] - token_times[i - 1]) * 1000.0)

    return {
        "ok": True,
        "total_ms": total_ms,
        "ttft_ms": ttft,
        "output_tokens": output_tokens,
        "itl_ms": itl_values,
        "tok_per_sec": output_tokens / (total_ms / 1000.0) if total_ms > 0 else 0,
        "output_text": output_text_joined[:200],
    }


# ---------------------------------------------------------------------------
# Run benchmark at a given concurrency
# ---------------------------------------------------------------------------
def run_bench(
    request_fn, total_requests, concurrency, tasks, normalize_output_words, task_suite
):
    results = []

    def do_one(idx):
        task = tasks[idx % len(tasks)]
        return request_fn(task["prompt"])

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(do_one, i): i for i in range(total_requests)}
        for f in as_completed(futs):
            results.append(f.result())
    wall_time = time.perf_counter() - t_start

    ok_results = [r for r in results if r.get("ok")]
    errors = [r for r in results if not r.get("ok")]

    latencies = [r["total_ms"] for r in ok_results]
    tok_per_sec_values = [r["tok_per_sec"] for r in ok_results if r.get("tok_per_sec")]
    ttft_values = [r["ttft_ms"] for r in ok_results if r.get("ttft_ms") is not None]
    total_output_tokens = sum(r.get("output_tokens", 0) for r in ok_results)

    all_itl = []
    for r in ok_results:
        all_itl.extend(r.get("itl_ms", []))

    sample_outputs = []
    for r in ok_results:
        text = (r.get("output_text") or "").strip()
        if text:
            sample_outputs.append(text[:200])
        if len(sample_outputs) >= 3:
            break

    fidelity = (
        fair_metrics(ok_results, tasks)
        if task_suite == "fair"
        else fidelity_metrics(
            ok_results, [task["prompt"] for task in tasks], normalize_output_words
        )
    )
    pass_rate = fidelity.get("pass_rate")
    goodput_rps = (
        (len(ok_results) * pass_rate) / wall_time
        if pass_rate is not None and wall_time > 0
        else None
    )

    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "ok": len(ok_results),
        "errors": len(errors),
        "wall_time_sec": wall_time,
        "throughput_rps": len(ok_results) / wall_time if wall_time > 0 else 0,
        "total_output_tokens": total_output_tokens,
        "avg_output_units": total_output_tokens / len(ok_results) if ok_results else 0,
        "overall_tok_per_sec": total_output_tokens / wall_time if wall_time > 0 else 0,
        "goodput_rps": goodput_rps,
        "latency_ms": summarize(latencies),
        "tok_per_sec": summarize(tok_per_sec_values),
        "ttft_ms": summarize(ttft_values) if ttft_values else None,
        "itl_ms": summarize(all_itl) if all_itl else None,
        "error_details": [e.get("error", "unknown") for e in errors[:5]],
        "sample_outputs": sample_outputs,
        "fidelity": fidelity,
    }


def collect_memory(kapsl_url, vllm_url, timeout):
    """Collect memory stats from both engines."""
    mem = {}
    try:
        resp = req_lib.get(f"{kapsl_url}/api/system/stats", timeout=timeout)
        if resp.status_code == 200:
            stats = resp.json()
            mem["kapsl_rss_bytes"] = stats.get("process_memory_bytes", 0)
            mem["kapsl_gpu_bytes"] = stats.get("gpu_memory_bytes")
    except Exception:
        pass
    try:
        resp = req_lib.get(f"{vllm_url}/metrics", timeout=timeout)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.startswith("vllm_gpu_memory_usage_bytes") or line.startswith(
                    "vllm:gpu_memory_used_bytes"
                ):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mem["vllm_gpu_bytes"] = float(parts[-1])
                        except ValueError:
                            pass
                if "process_resident_memory_bytes" in line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mem["vllm_rss_bytes"] = float(parts[-1])
                        except ValueError:
                            pass
    except Exception:
        pass
    return mem


def fmt_bytes(b):
    if b is None:
        return "n/a"
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024.0
    return f"{b:.1f} TiB"


def print_result(label, r):
    lat = r.get("latency_ms", {})
    tok = r.get("tok_per_sec", {})
    ttft = r.get("ttft_ms")
    itl = r.get("itl_ms")
    print(f"\n  [{label}] concurrency={r['concurrency']}")
    print(f"    requests: {r['ok']}/{r['total_requests']} ok, {r['errors']} errors")
    print(f"    wall time: {r['wall_time_sec']:.1f}s")
    print(f"    throughput: {r['throughput_rps']:.2f} req/s")
    print(
        f"    output units: {r['total_output_tokens']}  ({r['overall_tok_per_sec']:.1f} units/s overall)"
    )
    print(f"    avg output units/request: {r['avg_output_units']:.1f}")
    if r.get("goodput_rps") is not None:
        print(f"    goodput: {r['goodput_rps']:.2f} req/s")
    if lat:
        print(
            f"    latency (ms): p50={lat['p50']:.0f}  p90={lat['p90']:.0f}  p99={lat['p99']:.0f}  mean={lat['mean']:.0f}"
        )
    if tok:
        print(
            f"    units/s per req: p50={tok['p50']:.1f}  p90={tok['p90']:.1f}  mean={tok['mean']:.1f}"
        )
    if ttft:
        print(
            f"    TTFT (ms): p50={ttft['p50']:.0f}  p90={ttft['p90']:.0f}  p99={ttft['p99']:.0f}  mean={ttft['mean']:.0f}"
        )
    if itl:
        print(
            f"    ITL (ms): p50={itl['p50']:.1f}  p90={itl['p90']:.1f}  mean={itl['mean']:.1f}"
        )
    fidelity = r.get("fidelity") or {}
    pass_rate = fidelity.get("pass_rate")
    if pass_rate is not None:
        print(f"    pass rate: {pass_rate:.0%}")
    overlap = fidelity.get("prompt_overlap_mean")
    compliance = fidelity.get("target_word_compliance_rate")
    repeated = fidelity.get("repeated_output_rate")
    repeat_fail = fidelity.get("repeat_fail_rate")
    if overlap is not None:
        print(f"    prompt overlap mean: {overlap:.2f}")
    if compliance is not None:
        print(f"    target compliance: {compliance:.0%}")
    if repeated is not None:
        print(f"    repeated output rate: {repeated:.0%}")
    if repeat_fail is not None:
        print(f"    repeated output rate: {repeat_fail:.0%}")
    sample_outputs = r.get("sample_outputs") or []
    if sample_outputs:
        print(f"    sample output: {sample_outputs[0]!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive kapsl vs vLLM benchmark"
    )
    parser.add_argument("--kapsl-url", default="http://127.0.0.1:9195")
    parser.add_argument("--kapsl-model-id", type=int, default=0)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--task-suite",
        choices=["open", "fair"],
        default="open",
        help="Use open-ended prompts or validator-backed fair tasks.",
    )
    parser.add_argument(
        "--normalize-output-words",
        type=int,
        default=0,
        help="Wrap each prompt with an instruction to answer in exactly N words.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--kapsl-only",
        action="store_true",
        help="Run kapsl benchmark only, skip vLLM entirely",
    )
    args = parser.parse_args()

    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]
    tasks = build_workload(args.task_suite, args.normalize_output_words)

    print("=" * 60)
    print("Comprehensive kapsl vs vLLM Benchmark")
    print(f"Model: {args.vllm_model}")
    print(f"Requests per config: {args.requests}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Workload items: {len(tasks)}")
    print(f"Task suite: {args.task_suite}")
    if args.normalize_output_words:
        print(f"Normalized output target: {args.normalize_output_words} words")
    print("=" * 60)

    # Warmup
    print("\nWarming up...")
    for _ in range(args.warmup):
        kapsl_request(
            args.kapsl_url,
            args.kapsl_model_id,
            tasks[0]["prompt"],
            args.max_tokens,
            args.temperature,
            args.timeout,
        )
        if not args.kapsl_only:
            vllm_request(
                args.vllm_url,
                args.vllm_model,
                tasks[0]["prompt"],
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
    print("Warmup done.")

    # Collect pre-benchmark memory
    mem_before = collect_memory(args.kapsl_url, args.vllm_url, args.timeout)

    all_results = {
        "meta": {
            "model": args.vllm_model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "requests_per_config": args.requests,
            "concurrency_levels": concurrency_levels,
            "num_prompts": len(tasks),
            "normalize_output_words": args.normalize_output_words,
            "task_suite": args.task_suite,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "kapsl": {},
        "vllm": {},
        "memory": mem_before,
    }

    for conc in concurrency_levels:
        print(f"\n{'─' * 50}")
        print(f"Running concurrency={conc}, {args.requests} requests each")
        print(f"{'─' * 50}")

        # kapsl — try streaming first for TTFT
        def kapsl_fn(prompt):
            return kapsl_streaming_request(
                args.kapsl_url,
                args.kapsl_model_id,
                prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
            )

        kapsl_result = run_bench(
            kapsl_fn,
            args.requests,
            conc,
            tasks,
            args.normalize_output_words,
            args.task_suite,
        )
        print_result("kapsl", kapsl_result)
        all_results["kapsl"][f"c{conc}"] = kapsl_result

        if not args.kapsl_only:
            # vLLM — streaming for TTFT/ITL
            def vllm_fn(prompt):
                return vllm_streaming_request(
                    args.vllm_url,
                    args.vllm_model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )

            vllm_result = run_bench(
                vllm_fn,
                args.requests,
                conc,
                tasks,
                args.normalize_output_words,
                args.task_suite,
            )
            print_result("vLLM", vllm_result)
            all_results["vllm"][f"c{conc}"] = vllm_result

    # Collect post-benchmark memory
    mem_after = collect_memory(args.kapsl_url, args.vllm_url, args.timeout)
    all_results["memory_after"] = mem_after

    # Summary
    print(f"\n{'=' * 60}")
    if args.kapsl_only:
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Concurrency':<12} {'Metric':<20} {'kapsl':<15}")
        print("-" * 48)
        for conc in concurrency_levels:
            k = all_results["kapsl"][f"c{conc}"]
            print(
                f"c={conc:<10} {'throughput (req/s)':<20} {k['throughput_rps']:<15.2f}"
            )
            if k.get("goodput_rps") is not None:
                print(f"{'':12} {'goodput (req/s)':<20} {k['goodput_rps']:<15.2f}")
            print(
                f"{'':12} {'units/s (overall)':<20} {k['overall_tok_per_sec']:<15.1f}"
            )
            print(f"{'':12} {'avg units/request':<20} {k['avg_output_units']:<15.1f}")
            print(
                f"{'':12} {'p50 latency (ms)':<20} {k['latency_ms'].get('p50', 0):<15.0f}"
            )
            kttft = (k.get("ttft_ms") or {}).get("p50")
            if kttft:
                print(f"{'':12} {'TTFT p50 (ms)':<20} {kttft:<15.0f}")
            print()
    else:
        print("SUMMARY COMPARISON")
        print(f"{'=' * 60}")
        print(
            f"{'Concurrency':<12} {'Metric':<20} {'kapsl':<15} {'vLLM':<15} {'Ratio':<10}"
        )
        print("-" * 72)
        for conc in concurrency_levels:
            k = all_results["kapsl"][f"c{conc}"]
            v = all_results["vllm"][f"c{conc}"]
            ktp = k["throughput_rps"]
            vtp = v["throughput_rps"]
            ratio_tp = ktp / vtp if vtp else 0
            print(
                f"c={conc:<10} {'throughput (req/s)':<20} {ktp:<15.2f} {vtp:<15.2f} {ratio_tp:<10.2f}x"
            )
            kg = k.get("goodput_rps")
            vg = v.get("goodput_rps")
            if kg is not None and vg is not None:
                ratio_g = kg / vg if vg else 0
                print(
                    f"{'':12} {'goodput (req/s)':<20} {kg:<15.2f} {vg:<15.2f} {ratio_g:<10.2f}x"
                )

            ktok = k["overall_tok_per_sec"]
            vtok = v["overall_tok_per_sec"]
            ratio_tok = ktok / vtok if vtok else 0
            print(
                f"{'':12} {'units/s (overall)':<20} {ktok:<15.1f} {vtok:<15.1f} {ratio_tok:<10.2f}x"
            )
            kau = k["avg_output_units"]
            vau = v["avg_output_units"]
            ratio_au = kau / vau if vau else 0
            print(
                f"{'':12} {'avg units/request':<20} {kau:<15.1f} {vau:<15.1f} {ratio_au:<10.2f}x"
            )

            klat = k["latency_ms"].get("p50", 0)
            vlat = v["latency_ms"].get("p50", 0)
            ratio_lat = klat / vlat if vlat else 0
            print(
                f"{'':12} {'p50 latency (ms)':<20} {klat:<15.0f} {vlat:<15.0f} {ratio_lat:<10.2f}x"
            )

            kttft = (k.get("ttft_ms") or {}).get("p50")
            vttft = (v.get("ttft_ms") or {}).get("p50")
            if kttft and vttft:
                ratio_ttft = kttft / vttft if vttft else 0
                print(
                    f"{'':12} {'TTFT p50 (ms)':<20} {kttft:<15.0f} {vttft:<15.0f} {ratio_ttft:<10.2f}x"
                )
            print()

    # Memory
    print("Memory Usage:")
    if mem_after.get("kapsl_rss_bytes"):
        print(f"  kapsl RSS: {fmt_bytes(mem_after['kapsl_rss_bytes'])}")
    if mem_after.get("vllm_rss_bytes"):
        print(f"  vLLM  RSS: {fmt_bytes(mem_after['vllm_rss_bytes'])}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
