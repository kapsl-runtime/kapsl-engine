#!/usr/bin/env python3
"""
Concurrent inference test for Gemma (LLM) and Kokoro (TTS) via kapsl runtime HTTP API.

This script is meant to answer a simple question: when a Gemma request is in-flight,
can Kokoro still run (and vice-versa), or do they serialize?

Modes:
  - pair: launches 1 Gemma + 1 Kokoro request at the same time per iteration.
  - burst: launches many requests concurrently, alternating Gemma/Kokoro.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Copied from kapsl-runtime/models/kokoro/README.md (phoneme/token ids example).
DEFAULT_KOKORO_TOKENS: List[int] = [
    50,
    157,
    43,
    135,
    16,
    53,
    135,
    46,
    16,
    43,
    102,
    16,
    56,
    156,
    57,
    135,
    6,
    16,
    102,
    62,
    61,
    16,
    70,
    56,
    16,
    138,
    56,
    156,
    72,
    56,
    61,
    85,
    123,
    83,
    44,
    83,
    54,
    16,
    53,
    65,
    156,
    86,
    61,
    62,
    131,
    83,
    56,
    4,
    16,
    54,
    156,
    43,
    102,
    53,
    16,
    156,
    72,
    61,
    53,
    102,
    112,
    16,
    70,
    56,
    16,
    138,
    56,
    44,
    156,
    76,
    158,
    123,
    56,
    16,
    62,
    131,
    156,
    43,
    102,
    54,
    46,
    16,
    102,
    48,
    16,
    81,
    47,
    102,
    54,
    16,
    54,
    156,
    51,
    158,
    46,
    16,
    70,
    16,
    92,
    156,
    135,
    46,
    16,
    54,
    156,
    43,
    102,
    48,
    4,
    16,
    81,
    47,
    102,
    16,
    50,
    156,
    72,
    64,
    83,
    56,
    62,
    16,
    156,
    51,
    158,
    64,
    83,
    56,
    16,
    44,
    157,
    102,
    56,
    16,
    44,
    156,
    76,
    158,
    123,
    56,
    4,
]


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    return (
        cli_token
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


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def http_get_json(url: str, token: Optional[str], timeout: float) -> Any:
    req = urllib.request.Request(url, headers=auth_headers(token), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post_json(url: str, token: Optional[str], payload: Any, timeout: float) -> Any:
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


def list_models(
    base_url: str, token: Optional[str], timeout: float
) -> List[Dict[str, Any]]:
    payload = http_get_json(f"{base_url.rstrip('/')}/api/models", token, timeout)
    if not isinstance(payload, list):
        raise RuntimeError(f"Invalid /api/models response: {type(payload).__name__}")
    return [entry for entry in payload if isinstance(entry, dict)]


def model_is_active(model: Dict[str, Any]) -> bool:
    return str(model.get("status", "")).lower() == "active"


def model_is_base_model(model: Dict[str, Any]) -> bool:
    try:
        mid = int(model.get("id"))
    except (TypeError, ValueError):
        return False
    raw_base = model.get("base_model_id")
    if raw_base is None:
        return True
    try:
        base = int(raw_base)
    except (TypeError, ValueError):
        return True
    return mid == base


def choose_model_id(
    models: List[Dict[str, Any]],
    forced_model_id: Optional[int],
    name_contains: str,
) -> Tuple[int, str]:
    if forced_model_id is not None:
        for model in models:
            try:
                mid = int(model.get("id"))
            except (TypeError, ValueError):
                continue
            if mid == forced_model_id:
                if not model_is_active(model):
                    raise RuntimeError(
                        f"Model {forced_model_id} is not active (status={model.get('status')})"
                    )
                return forced_model_id, str(model.get("name") or forced_model_id)
        raise RuntimeError(f"Model id {forced_model_id} was not found in /api/models")

    needle = (name_contains or "").strip().lower()
    active = [m for m in models if model_is_active(m)]
    active_base = [m for m in active if model_is_base_model(m)]
    for model in active_base:
        name = str(model.get("name", "")).lower()
        if needle and needle in name:
            mid = int(model["id"])
            return mid, str(model.get("name") or mid)
    names = [str(m.get("name", "")) for m in active_base]
    raise RuntimeError(
        "Could not auto-select model. "
        "Use --gemma-id/--kokoro-id or --*-name-contains. "
        f"Active base models: {names}"
    )


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def pack_i64_native(values: Sequence[int]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"@{len(values)}q", *values)


def pack_f32_native(values: Sequence[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"@{len(values)}f", *values)


def read_style_vector_f32(voice_path: Path, index: int) -> List[float]:
    # Voice files are raw float32 vectors shaped (-1, 1, 256) in upstream examples.
    offset = index * 256 * 4
    with open(voice_path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(256 * 4)
    if len(raw) != 256 * 4:
        raise RuntimeError(
            f"Voice file too small for index={index}. "
            f"Need {offset + 256 * 4} bytes, got {voice_path.stat().st_size} bytes."
        )
    return list(struct.unpack("<256f", raw))


def resolve_voice_path(voice_file: Optional[str], voice_name: str) -> Path:
    if voice_file:
        return Path(voice_file)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root
        / "kapsl-runtime"
        / "models"
        / "kokoro"
        / "_slim_ctx"
        / "voices"
        / f"{voice_name}.bin",
        repo_root
        / "kapsl-runtime"
        / "models"
        / "kokoro"
        / "voices"
        / f"{voice_name}.bin",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def load_kokoro_tokens(tokens_file: Optional[str], token_count: int) -> List[int]:
    if tokens_file:
        raw = Path(tokens_file).read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, list) or not all(
            isinstance(x, int) for x in payload
        ):
            raise ValueError("--kokoro-tokens-file must be a JSON array of integers")
        tokens = [int(x) for x in payload]
    else:
        if token_count <= 0:
            token_count = 1
        tokens = list(DEFAULT_KOKORO_TOKENS[:token_count])
    if len(tokens) > 510:
        raise ValueError(f"kokoro token length must be <= 510 (got {len(tokens)})")
    return tokens


def build_gemma_payload(
    prompt: str,
    request_id: str,
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    timeout_ms: Optional[int] = None,
) -> Dict[str, Any]:
    data = prompt.encode("utf-8")
    metadata: Dict[str, Any] = {"request_id": request_id}
    if max_new_tokens is not None:
        metadata["max_new_tokens"] = int(max_new_tokens)
    if temperature is not None:
        metadata["temperature"] = float(temperature)
    if seed is not None:
        metadata["seed"] = int(seed)
    if timeout_ms is not None:
        metadata["timeout_ms"] = int(timeout_ms)
    return {
        "input": {
            "shape": [1, len(data)],
            "dtype": "string",
            "data_base64": base64.b64encode(data).decode("ascii"),
        },
        "metadata": metadata,
    }


def build_kokoro_payload(
    tokens: Sequence[int],
    voice_path: Path,
    speed: float,
    request_id: str,
) -> Dict[str, Any]:
    style_vec = read_style_vector_f32(voice_path, index=len(tokens))
    input_ids = [0] + list(tokens) + [0]

    input_bytes = pack_i64_native(input_ids)
    style_bytes = pack_f32_native(style_vec)
    speed_bytes = pack_f32_native([float(speed)])

    return {
        "input": {
            "shape": [1, len(input_ids)],
            "dtype": "int64",
            "data_base64": base64.b64encode(input_bytes).decode("ascii"),
        },
        "additional_inputs": [
            {
                "name": "style",
                "tensor": {
                    "shape": [1, 256],
                    "dtype": "float32",
                    "data_base64": base64.b64encode(style_bytes).decode("ascii"),
                },
            },
            {
                "name": "speed",
                "tensor": {
                    "shape": [1],
                    "dtype": "float32",
                    "data_base64": base64.b64encode(speed_bytes).decode("ascii"),
                },
            },
        ],
        "metadata": {
            "request_id": request_id,
        },
    }


def post_infer(
    infer_url: str,
    token: Optional[str],
    payload: Dict[str, Any],
    timeout_seconds: float,
) -> Tuple[bool, float, str, Optional[Dict[str, Any]]]:
    started = time.perf_counter()
    try:
        result = http_post_json(infer_url, token, payload, timeout_seconds)
        elapsed = time.perf_counter() - started
        if not isinstance(result, dict):
            return (
                False,
                elapsed,
                f"invalid response type={type(result).__name__}",
                None,
            )
        if "error" in result:
            return False, elapsed, str(result.get("error")), result
        return True, elapsed, "", result
    except urllib.error.HTTPError as err:
        try:
            body = err.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        elapsed = time.perf_counter() - started
        return False, elapsed, f"HTTP {err.code} {err.reason} {body}", None
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return False, elapsed, str(exc), None


def decode_string_output(result: Dict[str, Any]) -> str:
    dtype = str(result.get("dtype", "")).lower()
    if dtype != "string":
        raise ValueError(f"expected dtype=string, got {dtype}")
    data_field = result.get("data", [])
    if not isinstance(data_field, list):
        raise ValueError("invalid string output data; expected list")
    raw = bytes(int(x) & 0xFF for x in data_field)
    return raw.decode("utf-8", errors="replace")


def kokoro_audio_bytes_len(result: Dict[str, Any]) -> int:
    dtype = str(result.get("dtype", "")).lower()
    if dtype != "float32":
        raise ValueError(f"expected dtype=float32, got {dtype}")
    data_field = result.get("data", [])
    if not isinstance(data_field, list):
        raise ValueError("invalid float32 output data; expected list")
    return len(data_field)


def mode_pair(
    base_url: str,
    token: Optional[str],
    timeout_seconds: float,
    gemma_id: int,
    kokoro_id: int,
    gemma_prompt: str,
    kokoro_tokens: Sequence[int],
    voice_path: Path,
    kokoro_speed: float,
    iterations: int,
    poll_interval_s: float,
    gemma_max_new_tokens: Optional[int],
    gemma_temperature: Optional[float],
    gemma_seed: Optional[int],
    gemma_timeout_ms: Optional[int],
) -> int:
    gemma_url = f"{base_url.rstrip('/')}/api/models/{gemma_id}/infer"
    kokoro_url = f"{base_url.rstrip('/')}/api/models/{kokoro_id}/infer"
    gemma_state_url = f"{base_url.rstrip('/')}/api/models/{gemma_id}"
    kokoro_state_url = f"{base_url.rstrip('/')}/api/models/{kokoro_id}"

    gemma_lat: List[float] = []
    kokoro_lat: List[float] = []
    gemma_ok = 0
    kokoro_ok = 0
    saw_overlap_active = 0

    for i in range(1, iterations + 1):
        req_id_base = f"concurrent-gk-{time.time_ns()}-{i}"
        gemma_payload = build_gemma_payload(
            gemma_prompt,
            request_id=req_id_base + "-gemma",
            max_new_tokens=gemma_max_new_tokens,
            temperature=gemma_temperature,
            seed=gemma_seed,
            timeout_ms=gemma_timeout_ms,
        )
        kokoro_payload = build_kokoro_payload(
            kokoro_tokens,
            voice_path=voice_path,
            speed=kokoro_speed,
            request_id=req_id_base + "-kokoro",
        )

        barrier = threading.Barrier(3)
        done = threading.Event()

        def call_one(name: str, url: str, payload: Dict[str, Any]):
            barrier.wait()
            ok, elapsed, err, result = post_infer(url, token, payload, timeout_seconds)
            return name, ok, elapsed, err, result

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_g = pool.submit(call_one, "gemma", gemma_url, gemma_payload)
            fut_k = pool.submit(call_one, "kokoro", kokoro_url, kokoro_payload)

            # Start both together, then poll model states while requests are in-flight.
            barrier.wait()
            active_overlap = False
            while True:
                if fut_g.done() and fut_k.done():
                    break
                try:
                    gs = http_get_json(
                        gemma_state_url, token, timeout=min(3.0, timeout_seconds)
                    )
                    ks = http_get_json(
                        kokoro_state_url, token, timeout=min(3.0, timeout_seconds)
                    )
                    g_active = int((gs or {}).get("active_inferences", 0))
                    k_active = int((ks or {}).get("active_inferences", 0))
                    if g_active > 0 and k_active > 0:
                        active_overlap = True
                except Exception:
                    pass
                time.sleep(max(0.05, poll_interval_s))

            done.set()
            g_name, g_ok, g_elapsed, g_err, g_result = fut_g.result()
            k_name, k_ok, k_elapsed, k_err, k_result = fut_k.result()

        wall_overlap = (
            True  # in this mode they always overlap in-flight by construction
        )
        if active_overlap:
            saw_overlap_active += 1

        gemma_lat.append(g_elapsed)
        kokoro_lat.append(k_elapsed)
        if g_ok:
            gemma_ok += 1
        if k_ok:
            kokoro_ok += 1

        g_extra = ""
        if g_ok and g_result is not None:
            try:
                text = decode_string_output(g_result)
                text = text.strip().replace("\n", " ")
                if len(text) > 120:
                    text = text[:120] + "..."
                g_extra = f" out='{text}'"
            except Exception:
                g_extra = ""

        k_extra = ""
        if k_ok and k_result is not None:
            try:
                b_len = kokoro_audio_bytes_len(k_result)
                samples = b_len // 4
                k_extra = f" audio_bytes={b_len} samples={samples}"
            except Exception:
                k_extra = ""

        print(
            f"iter={i:02d} overlap_wall={wall_overlap} overlap_active={active_overlap} | "
            f"gemma ok={g_ok} t={g_elapsed:.2f}s{g_extra} | "
            f"kokoro ok={k_ok} t={k_elapsed:.2f}s{k_extra}"
        )
        if (not g_ok) and g_err:
            print(f"  gemma_error: {g_err}")
        if (not k_ok) and k_err:
            print(f"  kokoro_error: {k_err}")

    gemma_sorted = sorted(gemma_lat)
    kokoro_sorted = sorted(kokoro_lat)
    print("\nSummary (pair)")
    print(
        f"  gemma: ok={gemma_ok}/{iterations} p50={percentile(gemma_sorted, 0.5):.2f}s p95={percentile(gemma_sorted, 0.95):.2f}s"
    )
    print(
        f"  kokoro: ok={kokoro_ok}/{iterations} p50={percentile(kokoro_sorted, 0.5):.2f}s p95={percentile(kokoro_sorted, 0.95):.2f}s"
    )
    print(f"  overlap_active_samples: {saw_overlap_active}/{iterations}")
    return 0 if gemma_ok > 0 and kokoro_ok > 0 else 1


def mode_burst(
    base_url: str,
    token: Optional[str],
    timeout_seconds: float,
    gemma_id: int,
    kokoro_id: int,
    gemma_prompt: str,
    kokoro_tokens: Sequence[int],
    voice_path: Path,
    kokoro_speed: float,
    cycles: int,
    concurrency: int,
    requests_per_worker: int,
    gemma_max_new_tokens: Optional[int],
    gemma_temperature: Optional[float],
    gemma_seed: Optional[int],
    gemma_timeout_ms: Optional[int],
) -> int:
    gemma_url = f"{base_url.rstrip('/')}/api/models/{gemma_id}/infer"
    kokoro_url = f"{base_url.rstrip('/')}/api/models/{kokoro_id}/infer"

    total_ok = 0
    total_req = 0

    for cycle in range(1, cycles + 1):
        started = time.perf_counter()
        ok_by_model = {"gemma": 0, "kokoro": 0}
        err_by_model = {"gemma": 0, "kokoro": 0}
        lat_by_model: Dict[str, List[float]] = {"gemma": [], "kokoro": []}
        errors: List[str] = []

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for worker in range(concurrency):
                for idx in range(requests_per_worker):
                    target = (
                        "gemma"
                        if ((worker * requests_per_worker + idx) % 2 == 0)
                        else "kokoro"
                    )
                    req_id = f"burst-gk-{cycle}-{worker}-{idx}-{time.time_ns()}"
                    if target == "gemma":
                        url = gemma_url
                        payload = build_gemma_payload(
                            gemma_prompt,
                            request_id=req_id,
                            max_new_tokens=gemma_max_new_tokens,
                            temperature=gemma_temperature,
                            seed=gemma_seed,
                            timeout_ms=gemma_timeout_ms,
                        )
                    else:
                        url = kokoro_url
                        payload = build_kokoro_payload(
                            kokoro_tokens,
                            voice_path=voice_path,
                            speed=kokoro_speed,
                            request_id=req_id,
                        )
                    futures.append(
                        (
                            target,
                            pool.submit(
                                post_infer, url, token, payload, timeout_seconds
                            ),
                        )
                    )

            for target, fut in futures:
                ok, elapsed, err, _ = fut.result()
                lat_by_model[target].append(elapsed)
                if ok:
                    ok_by_model[target] += 1
                else:
                    err_by_model[target] += 1
                    if err:
                        errors.append(f"{target}: {err}")

        elapsed_s = time.perf_counter() - started
        reqs = concurrency * requests_per_worker
        oks = ok_by_model["gemma"] + ok_by_model["kokoro"]
        total_ok += oks
        total_req += reqs

        print(
            f"cycle={cycle:02d} ok={oks}/{reqs} burst_s={elapsed_s:.2f} "
            f"gemma_ok={ok_by_model['gemma']} kokoro_ok={ok_by_model['kokoro']} "
            f"gemma_p50={percentile(sorted(lat_by_model['gemma']), 0.5):.2f}s "
            f"kokoro_p50={percentile(sorted(lat_by_model['kokoro']), 0.5):.2f}s"
        )
        if errors:
            print("  sample_errors:")
            for e in errors[:5]:
                print(f"    - {e}")

    print("\nSummary (burst)")
    print(f"  total_ok={total_ok}/{total_req}")
    return 0 if total_ok > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concurrent Gemma + Kokoro inference test via kapsl runtime HTTP API"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9095")
    parser.add_argument("--token", help="API token (or set KAPSL_API_TOKEN*)")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)

    parser.add_argument("--gemma-id", type=int, help="Gemma model id override")
    parser.add_argument("--kokoro-id", type=int, help="Kokoro model id override")
    parser.add_argument("--gemma-name-contains", default="gemma")
    parser.add_argument("--kokoro-name-contains", default="kokoro")

    parser.add_argument("--gemma-prompt", default="Reply with exactly: ok")
    parser.add_argument(
        "--gemma-max-new-tokens",
        type=int,
        default=None,
        help="Request metadata override for LLM max_new_tokens",
    )
    parser.add_argument(
        "--gemma-temperature",
        type=float,
        default=None,
        help="Request metadata override for LLM temperature (0 for greedy)",
    )
    parser.add_argument(
        "--gemma-seed",
        type=int,
        default=None,
        help="Request metadata override for LLM seed (stabilizes sampling across concurrency)",
    )
    parser.add_argument(
        "--gemma-timeout-ms",
        type=int,
        default=None,
        help="Server-side per-request timeout via request.metadata.timeout_ms",
    )

    parser.add_argument("--voice", default="af")
    parser.add_argument(
        "--voice-file", help="Path to Kokoro voice .bin (overrides --voice)"
    )
    parser.add_argument("--kokoro-speed", type=float, default=1.0)
    parser.add_argument("--kokoro-tokens-file", help="JSON array of token ids")
    parser.add_argument(
        "--kokoro-token-count",
        type=int,
        default=64,
        help="When --kokoro-tokens-file is not provided, use the first N default tokens",
    )

    parser.add_argument("--mode", choices=["pair", "burst"], default="pair")
    parser.add_argument(
        "--iterations", type=int, default=3, help="pair mode iterations"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="pair mode polling interval (s)",
    )
    parser.add_argument("--cycles", type=int, default=2, help="burst mode cycles")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="burst mode thread workers"
    )
    parser.add_argument(
        "--requests-per-worker",
        type=int,
        default=2,
        help="burst mode requests per worker",
    )
    args = parser.parse_args()

    token = resolve_token(args.token)

    try:
        models = list_models(
            args.base_url, token, timeout=min(10.0, args.timeout_seconds)
        )
        gemma_id, gemma_name = choose_model_id(
            models,
            forced_model_id=args.gemma_id,
            name_contains=args.gemma_name_contains,
        )
        kokoro_id, kokoro_name = choose_model_id(
            models,
            forced_model_id=args.kokoro_id,
            name_contains=args.kokoro_name_contains,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if gemma_id == kokoro_id:
        print(
            f"[error] gemma and kokoro resolved to the same model id={gemma_id}. "
            "Use --gemma-id/--kokoro-id to disambiguate.",
            file=sys.stderr,
        )
        return 1

    voice_path = resolve_voice_path(args.voice_file, args.voice)
    if not voice_path.exists():
        print(f"[error] voice file not found: {voice_path}", file=sys.stderr)
        return 1

    try:
        kokoro_tokens = load_kokoro_tokens(
            args.kokoro_tokens_file, args.kokoro_token_count
        )
    except Exception as exc:
        print(f"[error] invalid kokoro tokens: {exc}", file=sys.stderr)
        return 1

    print(f"Gemma:  id={gemma_id} name={gemma_name}")
    print(f"Kokoro: id={kokoro_id} name={kokoro_name}")
    print(f"Kokoro voice: {voice_path}")
    print(f"Kokoro tokens: {len(kokoro_tokens)} speed={args.kokoro_speed}")
    print(f"Mode: {args.mode}")

    if args.mode == "pair":
        return mode_pair(
            base_url=args.base_url,
            token=token,
            timeout_seconds=args.timeout_seconds,
            gemma_id=gemma_id,
            kokoro_id=kokoro_id,
            gemma_prompt=args.gemma_prompt,
            kokoro_tokens=kokoro_tokens,
            voice_path=voice_path,
            kokoro_speed=args.kokoro_speed,
            iterations=max(1, args.iterations),
            poll_interval_s=max(0.05, args.poll_interval),
            gemma_max_new_tokens=args.gemma_max_new_tokens,
            gemma_temperature=args.gemma_temperature,
            gemma_seed=args.gemma_seed,
            gemma_timeout_ms=args.gemma_timeout_ms,
        )

    return mode_burst(
        base_url=args.base_url,
        token=token,
        timeout_seconds=args.timeout_seconds,
        gemma_id=gemma_id,
        kokoro_id=kokoro_id,
        gemma_prompt=args.gemma_prompt,
        kokoro_tokens=kokoro_tokens,
        voice_path=voice_path,
        kokoro_speed=args.kokoro_speed,
        cycles=max(1, args.cycles),
        concurrency=max(1, args.concurrency),
        requests_per_worker=max(1, args.requests_per_worker),
        gemma_max_new_tokens=args.gemma_max_new_tokens,
        gemma_temperature=args.gemma_temperature,
        gemma_seed=args.gemma_seed,
        gemma_timeout_ms=args.gemma_timeout_ms,
    )


if __name__ == "__main__":
    raise SystemExit(main())
