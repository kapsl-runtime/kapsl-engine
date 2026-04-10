#!/usr/bin/env python3
"""
Kapsl Runtime Benchmark
=======================
Measures throughput, latency distribution, and (for LLMs) tokens/sec on a
running kapsl-runtime instance.

Usage
-----
# LLM (string dtype) benchmark — auto-detects first loaded model:
  python3 benchmark.py --mode llm

# Tensor benchmark (ONNX / vision):
  python3 benchmark.py --mode tensor --shape 1,1,28,28 --dtype float32

# Custom concurrency sweep:
  python3 benchmark.py --mode llm --concurrency 1,4,8,16 --requests 200

# Target a specific model id and remote host:
  python3 benchmark.py --mode llm --model-id 0 --base-url http://10.0.0.5:9095

# Write JSON results to file:
  python3 benchmark.py --mode llm --output results.json

Environment variables
---------------------
  KAPSL_API_TOKEN          API bearer token (reader role is enough)
  KAPSL_BENCHMARK_TOKEN    overrides KAPSL_API_TOKEN for this script only
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://127.0.0.1:9095"
DEFAULT_REQUESTS = 100
DEFAULT_CONCURRENCY = "1,4,8"
DEFAULT_TIMEOUT = 120.0
DEFAULT_WARMUP = 5

LLM_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in two sentences.",
    "What is the capital of France and why is it significant?",
    "Write a one-sentence summary of the theory of relativity.",
    "List three benefits of regular exercise.",
    "What is a transformer model in the context of deep learning?",
    "Describe how a hash table works.",
    "What is the difference between TCP and UDP?",
    "Explain gradient descent in simple terms.",
    "What does REST stand for and what are its core principles?",
    "Why is the sky blue?",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    rank = (len(s) - 1) * p
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    return s[lo] * (1.0 - (rank - lo)) + s[hi] * (rank - lo)


def fmt_ms(ms: float) -> str:
    return f"{ms:.1f} ms"


def fmt_rps(rps: float) -> str:
    return f"{rps:.2f} req/s"


def get_token() -> Optional[str]:
    return (
        os.getenv("KAPSL_BENCHMARK_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
    )


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get_json(url: str, token: Optional[str], timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers=auth_headers(token), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url: str, token: Optional[str], payload: Any, timeout: float) -> Any:
    body = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=body, headers=auth_headers(token), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Hardware info collection
# ---------------------------------------------------------------------------

def collect_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "gpus": [],
    }

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                timeout=10,
                text=True,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                gpu: Dict[str, Any] = {}
                if len(parts) >= 1:
                    gpu["name"] = parts[0]
                if len(parts) >= 2:
                    gpu["memory_mb"] = parts[1]
                if len(parts) >= 3:
                    gpu["driver"] = parts[2]
                if len(parts) >= 4:
                    gpu["compute_cap"] = parts[3]
                info["gpus"].append(gpu)
        except Exception:
            pass

    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def build_tensor_payload(shape: List[int], dtype: str) -> Dict[str, Any]:
    n = 1
    for d in shape:
        n *= d
    fmt = {"float32": "f", "float64": "d", "int32": "i", "int64": "q", "uint8": "B"}.get(dtype, "f")
    data = struct.pack(f"@{n}{fmt}", *([1] * n))
    return {
        "input": {
            "shape": shape,
            "dtype": dtype,
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    }


def build_string_payload(prompt: str) -> Dict[str, Any]:
    return {
        "input": {
            "shape": [1, 1],
            "dtype": "string",
            "data_base64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        }
    }


def decode_string_response(result: Dict[str, Any]) -> Optional[str]:
    if result.get("dtype") != "string":
        return None
    data = result.get("data", [])
    if isinstance(data, list):
        return bytes(data).decode("utf-8", errors="replace")
    if isinstance(data, str):
        return base64.b64decode(data).decode("utf-8", errors="replace")
    return None


def count_tokens(text: str) -> int:
    """Rough token count: split on whitespace + punctuation (GPT-style ~0.75 words/token)."""
    words = text.split()
    return max(1, int(len(words) / 0.75))


# ---------------------------------------------------------------------------
# Single request runner
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    ok: bool
    latency_ms: float
    output_tokens: int = 0
    error: str = ""


def run_request(
    infer_url: str,
    token: Optional[str],
    payload: Dict[str, Any],
    timeout: float,
    mode: str,
    prompt_idx: int = 0,
) -> RequestResult:
    if mode == "llm":
        prompt = LLM_PROMPTS[prompt_idx % len(LLM_PROMPTS)]
        payload = build_string_payload(prompt)

    t0 = time.perf_counter()
    try:
        result = post_json(infer_url, token, payload, timeout)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        output_tokens = 0
        if mode == "llm":
            text = decode_string_response(result)
            if text:
                output_tokens = count_tokens(text)

        return RequestResult(ok=True, latency_ms=latency_ms, output_tokens=output_tokens)

    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        body = e.read().decode("utf-8", errors="replace").strip()
        return RequestResult(ok=False, latency_ms=latency_ms, error=f"HTTP {e.code}: {body[:120]}")
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RequestResult(ok=False, latency_ms=latency_ms, error=str(e)[:120])


# ---------------------------------------------------------------------------
# Concurrency sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    concurrency: int
    total_requests: int
    successes: int
    failures: int
    elapsed_s: float
    throughput_rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    avg_output_tokens: float = 0.0
    tokens_per_sec: float = 0.0
    errors: List[str] = field(default_factory=list)


def run_sweep(
    infer_url: str,
    token: Optional[str],
    concurrency: int,
    total_requests: int,
    payload: Dict[str, Any],
    timeout: float,
    mode: str,
    label: str,
) -> SweepResult:
    latencies: List[float] = []
    output_tokens_list: List[int] = []
    errors: List[str] = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(run_request, infer_url, token, payload, timeout, mode, i)
            for i in range(total_requests)
        ]
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            latencies.append(res.latency_ms)
            if res.ok:
                output_tokens_list.append(res.output_tokens)
            else:
                errors.append(res.error)
            done += 1
            pct = done / total_requests * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            print(
                f"\r  [{bar}] {done}/{total_requests} ({pct:.0f}%)  ",
                end="",
                flush=True,
            )
    print()

    elapsed = time.perf_counter() - t_start
    successes = total_requests - len(errors)
    throughput = successes / elapsed if elapsed > 0 else 0.0

    lat_sorted = sorted(latencies)
    p50 = percentile(lat_sorted, 0.50)
    p95 = percentile(lat_sorted, 0.95)
    p99 = percentile(lat_sorted, 0.99)
    lat_min = lat_sorted[0] if lat_sorted else 0.0
    lat_max = lat_sorted[-1] if lat_sorted else 0.0

    avg_tokens = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0.0
    tokens_per_sec = (sum(output_tokens_list) / elapsed) if (output_tokens_list and elapsed > 0) else 0.0

    return SweepResult(
        concurrency=concurrency,
        total_requests=total_requests,
        successes=successes,
        failures=len(errors),
        elapsed_s=elapsed,
        throughput_rps=throughput,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        latency_min_ms=lat_min,
        latency_max_ms=lat_max,
        avg_output_tokens=avg_tokens,
        tokens_per_sec=tokens_per_sec,
        errors=errors[:5],
    )


# ---------------------------------------------------------------------------
# GPU utilisation poller
# ---------------------------------------------------------------------------

class GpuPoller:
    def __init__(self) -> None:
        self._samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if not self._samples:
            return {}
        utils = [s["util"] for s in self._samples]
        mems = [s["mem_used"] for s in self._samples]
        return {
            "gpu_util_avg_pct": round(sum(utils) / len(utils), 1),
            "gpu_util_max_pct": round(max(utils), 1),
            "gpu_mem_used_max_mb": round(max(mems), 0),
        }

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    timeout=5, text=True,
                )
                for line in out.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) >= 2:
                        self._samples.append({
                            "util": float(parts[0].strip()),
                            "mem_used": float(parts[1].strip()),
                        })
            except Exception:
                pass
            self._stop.wait(0.5)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_model(base_url: str, token: Optional[str], model_id: Optional[int]) -> Tuple[int, str]:
    models_url = f"{base_url.rstrip('/')}/api/models"
    try:
        models = get_json(models_url, token, timeout=10.0)
    except Exception as e:
        print(f"ERROR: cannot reach {models_url}: {e}")
        sys.exit(1)

    if not models:
        print("ERROR: no models loaded in the runtime. Start kapsl with --model <path>.")
        sys.exit(1)

    if model_id is not None:
        for m in models:
            if int(m.get("id", -1)) == model_id:
                return model_id, m.get("name", str(model_id))
        print(f"ERROR: model id {model_id} not found. Available: {[m.get('id') for m in models]}")
        sys.exit(1)

    # Auto-pick first loaded model
    m = models[0]
    return int(m.get("id", 0)), m.get("name", "model-0")


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def print_sweep_table(results: List[SweepResult], mode: str) -> None:
    has_tokens = mode == "llm"
    if has_tokens:
        hdr = f"{'Conc':>6}  {'Req':>5}  {'OK':>5}  {'Fail':>5}  {'RPS':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'Tok/s':>8}"
        sep = "-" * 72
    else:
        hdr = f"{'Conc':>6}  {'Req':>5}  {'OK':>5}  {'Fail':>5}  {'RPS':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}"
        sep = "-" * 62

    print(hdr)
    print(sep)
    for r in results:
        if has_tokens:
            print(
                f"{r.concurrency:>6}  {r.total_requests:>5}  {r.successes:>5}  {r.failures:>5}"
                f"  {r.throughput_rps:>8.2f}"
                f"  {r.latency_p50_ms:>7.1f}m"
                f"  {r.latency_p95_ms:>7.1f}m"
                f"  {r.latency_p99_ms:>7.1f}m"
                f"  {r.tokens_per_sec:>8.1f}"
            )
        else:
            print(
                f"{r.concurrency:>6}  {r.total_requests:>5}  {r.successes:>5}  {r.failures:>5}"
                f"  {r.throughput_rps:>8.2f}"
                f"  {r.latency_p50_ms:>7.1f}m"
                f"  {r.latency_p95_ms:>7.1f}m"
                f"  {r.latency_p99_ms:>7.1f}m"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kapsl runtime benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Runtime base URL")
    p.add_argument("--model-id", type=int, default=None, help="Model id (auto-detect if omitted)")
    p.add_argument(
        "--mode",
        choices=["llm", "tensor"],
        default="llm",
        help="llm = string dtype (LLM / GGUF), tensor = float/int tensor (ONNX etc)",
    )
    p.add_argument(
        "--shape",
        default="1,1,28,28",
        help="Tensor shape for --mode tensor, comma-separated (default: 1,1,28,28)",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        help="Tensor dtype for --mode tensor (default: float32)",
    )
    p.add_argument(
        "--concurrency",
        default=DEFAULT_CONCURRENCY,
        help="Comma-separated concurrency levels to sweep (default: 1,4,8)",
    )
    p.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help="Total requests per concurrency level (default: 100)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup requests before each sweep (default: 5)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout seconds (default: 120)",
    )
    p.add_argument("--token", default=None, help="API bearer token")
    p.add_argument("--output", default=None, help="Write JSON results to this file")
    p.add_argument("--no-gpu-poll", action="store_true", help="Disable nvidia-smi polling")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    token = args.token or get_token()
    base_url = args.base_url.rstrip("/")

    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    if not concurrency_levels:
        print("ERROR: --concurrency must be a non-empty comma-separated list of integers")
        return 1

    shape = [int(x) for x in args.shape.split(",")]

    # -----------------------------------------------------------------------
    print_header("Kapsl Runtime Benchmark")
    print(f"  Runtime  : {base_url}")
    print(f"  Mode     : {args.mode}")
    print(f"  Requests : {args.requests} per concurrency level")
    print(f"  Warmup   : {args.warmup} requests")
    print(f"  Timeout  : {args.timeout}s")
    print(f"  Auth     : {'set' if token else 'none'}")

    # -----------------------------------------------------------------------
    print_header("Hardware")
    hw = collect_hardware_info()
    print(f"  OS      : {hw.get('os', 'unknown')}")
    print(f"  CPU     : {hw.get('cpu', 'unknown')}  ({hw.get('cpu_cores', '?')} cores)")
    print(f"  RAM     : {hw.get('ram_gb', '?')} GB")
    if hw.get("gpus"):
        for i, gpu in enumerate(hw["gpus"]):
            print(f"  GPU {i}   : {gpu.get('name', '?')}  {gpu.get('memory_mb', '?')} MB  "
                  f"(driver {gpu.get('driver', '?')}, cc {gpu.get('compute_cap', '?')})")
    else:
        print("  GPU     : none detected")

    # -----------------------------------------------------------------------
    print_header("Runtime info")
    model_id, model_name = discover_model(base_url, token, args.model_id)
    print(f"  Model   : {model_name}  (id={model_id})")

    try:
        stats = get_json(f"{base_url}/api/system/stats", token)
        print(f"  Stats   : {json.dumps(stats)}")
    except Exception:
        pass

    infer_url = f"{base_url}/api/models/{model_id}/infer"

    # -----------------------------------------------------------------------
    # Build base payload (tensor mode only; llm mode builds per-request)
    base_payload: Dict[str, Any] = {}
    if args.mode == "tensor":
        base_payload = build_tensor_payload(shape, args.dtype)
        print(f"  Shape   : {shape}  dtype={args.dtype}")

    # -----------------------------------------------------------------------
    all_results: List[SweepResult] = []
    all_gpu_stats: List[Dict[str, float]] = []

    for conc in concurrency_levels:
        print_header(f"Sweep: concurrency={conc}  requests={args.requests}")

        # Warmup
        if args.warmup > 0:
            print(f"  Warming up ({args.warmup} requests)...")
            with ThreadPoolExecutor(max_workers=min(conc, args.warmup)) as ex:
                wfutures = [
                    ex.submit(run_request, infer_url, token, base_payload, args.timeout, args.mode, i)
                    for i in range(args.warmup)
                ]
                for wf in as_completed(wfutures):
                    _ = wf.result()
            print("  Warmup done.")

        # GPU polling
        poller = GpuPoller()
        if not args.no_gpu_poll:
            poller.start()

        print(f"  Running {args.requests} requests at concurrency {conc}...")
        result = run_sweep(
            infer_url=infer_url,
            token=token,
            concurrency=conc,
            total_requests=args.requests,
            payload=base_payload,
            timeout=args.timeout,
            mode=args.mode,
            label=f"conc={conc}",
        )

        gpu_stats = poller.stop()

        all_results.append(result)
        all_gpu_stats.append(gpu_stats)

        print(f"  Elapsed    : {result.elapsed_s:.2f}s")
        print(f"  Successes  : {result.successes}/{result.total_requests}")
        print(f"  Throughput : {fmt_rps(result.throughput_rps)}")
        print(f"  Latency    : p50={fmt_ms(result.latency_p50_ms)}  "
              f"p95={fmt_ms(result.latency_p95_ms)}  "
              f"p99={fmt_ms(result.latency_p99_ms)}")
        print(f"  Lat range  : {fmt_ms(result.latency_min_ms)} – {fmt_ms(result.latency_max_ms)}")
        if args.mode == "llm" and result.tokens_per_sec > 0:
            print(f"  Tokens/s   : {result.tokens_per_sec:.1f}  (avg {result.avg_output_tokens:.0f} tok/req)")
        if gpu_stats:
            print(f"  GPU util   : avg {gpu_stats.get('gpu_util_avg_pct', '?')}%  "
                  f"max {gpu_stats.get('gpu_util_max_pct', '?')}%  "
                  f"mem {gpu_stats.get('gpu_mem_used_max_mb', '?')} MB")
        if result.errors:
            print(f"  Errors ({len(result.errors)} shown):")
            for e in result.errors:
                print(f"    - {e}")

    # -----------------------------------------------------------------------
    print_header("Summary")
    print_sweep_table(all_results, args.mode)

    best_rps = max(all_results, key=lambda r: r.throughput_rps)
    print()
    print(f"  Peak throughput : {fmt_rps(best_rps.throughput_rps)}  (concurrency={best_rps.concurrency})")
    if args.mode == "llm":
        best_tok = max(all_results, key=lambda r: r.tokens_per_sec)
        print(f"  Peak tokens/s   : {best_tok.tokens_per_sec:.1f}  (concurrency={best_tok.concurrency})")

    # -----------------------------------------------------------------------
    if args.output:
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "base_url": base_url,
                "model_id": model_id,
                "model_name": model_name,
                "mode": args.mode,
                "requests_per_level": args.requests,
                "warmup": args.warmup,
                "timeout_s": args.timeout,
                "shape": shape if args.mode == "tensor" else None,
                "dtype": args.dtype if args.mode == "tensor" else "string",
            },
            "hardware": hw,
            "results": [
                {
                    "concurrency": r.concurrency,
                    "total_requests": r.total_requests,
                    "successes": r.successes,
                    "failures": r.failures,
                    "elapsed_s": round(r.elapsed_s, 3),
                    "throughput_rps": round(r.throughput_rps, 3),
                    "latency_p50_ms": round(r.latency_p50_ms, 2),
                    "latency_p95_ms": round(r.latency_p95_ms, 2),
                    "latency_p99_ms": round(r.latency_p99_ms, 2),
                    "latency_min_ms": round(r.latency_min_ms, 2),
                    "latency_max_ms": round(r.latency_max_ms, 2),
                    "avg_output_tokens": round(r.avg_output_tokens, 1),
                    "tokens_per_sec": round(r.tokens_per_sec, 2),
                    "gpu_stats": gpu,
                    "sample_errors": r.errors,
                }
                for r, gpu in zip(all_results, all_gpu_stats)
            ],
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
