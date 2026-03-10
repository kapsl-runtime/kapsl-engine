#!/usr/bin/env python3
"""
Smoke test for the Kokoro TTS ONNX model via kapsl runtime HTTP API.

This script sends a multi-input ONNX inference request:
  - input_ids: int64 tensor (primary input)
  - style: float32 tensor (additional input named "style")
  - speed: float32 tensor (additional input named "speed")

It writes the returned float32 audio waveform to a 24kHz mono WAV file (int16 PCM).

Notes:
  - This script sends tensor bytes as base64 (`data_base64`) and interprets results using
    native endianness on the server. It packs/unpacks with native endianness.
  - The default token sequence is copied from `kapsl-runtime/models/kokoro/README.md`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_TOKENS: List[int] = [
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
        "Use --model-id or --model-name-contains. "
        f"Active base models: {names}"
    )


def pack_i64_native(values: Sequence[int]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"@{len(values)}q", *values)


def pack_f32_native(values: Sequence[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"@{len(values)}f", *values)


def unpack_f32_native(raw: bytes) -> List[float]:
    if len(raw) % 4 != 0:
        raise ValueError(
            f"float32 buffer size must be divisible by 4, got {len(raw)} bytes"
        )
    count = len(raw) // 4
    if count == 0:
        return []
    return list(struct.unpack(f"@{count}f", raw))


def read_style_vector_f32(voice_path: Path, index: int) -> List[float]:
    # Voice files are raw float32 vectors shaped (-1, 1, 256) in the upstream Kokoro example.
    # Index selects the vector based on token length.
    offset = index * 256 * 4
    with open(voice_path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(256 * 4)
    if len(raw) != 256 * 4:
        raise RuntimeError(
            f"Voice file too small for index={index}. "
            f"Need {offset + 256 * 4} bytes, got {voice_path.stat().st_size} bytes."
        )
    # The voice bins are produced on little-endian platforms; decode explicitly.
    return list(struct.unpack("<256f", raw))


def wav_write_int16_mono(
    path: Path, samples: Sequence[float], sample_rate: int
) -> None:
    # Convert float32 [-1,1] to int16 PCM little-endian.
    frames = bytearray()
    for s in samples:
        if s > 1.0:
            s = 1.0
        elif s < -1.0:
            s = -1.0
        v = int(round(s * 32767.0))
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        frames += struct.pack("<h", v)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def load_tokens(tokens_file: Optional[str]) -> List[int]:
    if not tokens_file:
        return list(DEFAULT_TOKENS)
    path = Path(tokens_file)
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, list) or not all(isinstance(x, int) for x in payload):
        raise ValueError("--tokens-file must be a JSON array of integers")
    return [int(x) for x in payload]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kokoro TTS smoke test via kapsl runtime HTTP API"
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:9095", help="Runtime base URL"
    )
    parser.add_argument("--token", help="API token (or set KAPSL_API_TOKEN*)")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--model-id", type=int, help="Override auto-selected model id")
    parser.add_argument(
        "--model-name-contains",
        default="kokoro",
        help="Substring match for model selection when --model-id is not provided",
    )
    parser.add_argument(
        "--voice",
        default="af",
        help="Voice name (maps to <voice>.bin). Example: af, af_bella, am_adam",
    )
    parser.add_argument(
        "--voice-file", help="Path to voice .bin file (overrides --voice)"
    )
    parser.add_argument("--tokens-file", help="JSON file containing token IDs array")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--output-wav", default="kokoro_test.wav")
    args = parser.parse_args()

    token = resolve_token(args.token)

    try:
        models = list_models(args.base_url, token, args.timeout_seconds)
        model_id, model_name = choose_model_id(
            models,
            forced_model_id=args.model_id,
            name_contains=args.model_name_contains,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    tokens = load_tokens(args.tokens_file)
    if len(tokens) > 510:
        print(
            f"[error] token length must be <= 510 (got {len(tokens)}).",
            file=sys.stderr,
        )
        return 1

    if args.voice_file:
        voice_path = Path(args.voice_file)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root
            / "kapsl-runtime"
            / "models"
            / "kokoro"
            / "_slim_ctx"
            / "voices"
            / f"{args.voice}.bin",
            repo_root
            / "kapsl-runtime"
            / "models"
            / "kokoro"
            / "voices"
            / f"{args.voice}.bin",
        ]
        voice_path = next((p for p in candidates if p.exists()), candidates[0])

    if not voice_path.exists():
        print(
            f"[error] voice file not found: {voice_path}. Use --voice-file to override.",
            file=sys.stderr,
        )
        return 1

    try:
        style_vec = read_style_vector_f32(voice_path, index=len(tokens))
    except Exception as exc:
        print(
            f"[error] failed to read style vector from {voice_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    input_ids = [0] + tokens + [0]
    input_bytes = pack_i64_native(input_ids)
    style_bytes = pack_f32_native(style_vec)  # (256,) -> shape [1,256]
    speed_bytes = pack_f32_native([float(args.speed)])

    infer_url = f"{args.base_url.rstrip('/')}/api/models/{model_id}/infer"
    payload: Dict[str, Any] = {
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
            "request_id": f"kokoro-smoke-{time.time_ns()}",
        },
    }

    print(f"Using model {model_name} (id={model_id}) at {infer_url}")
    print(f"Voice: {voice_path}")
    print(f"Tokens: {len(tokens)} (+2 pads => {len(input_ids)}) speed={args.speed}")

    started = time.perf_counter()
    try:
        result = http_post_json(infer_url, token, payload, args.timeout_seconds)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace").strip()
        print(f"[error] HTTP {err.code} {err.reason}: {body}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[error] request failed: {exc}", file=sys.stderr)
        return 1

    elapsed_s = time.perf_counter() - started
    if isinstance(result, dict) and "error" in result:
        print(f"[error] runtime error: {result.get('error')}", file=sys.stderr)
        return 1
    if not isinstance(result, dict):
        print(
            f"[error] invalid infer response: {type(result).__name__}", file=sys.stderr
        )
        return 1

    dtype = str(result.get("dtype", "")).lower()
    if dtype != "float32":
        print(
            f"[error] unexpected output dtype={dtype} (want float32)", file=sys.stderr
        )
        return 1

    data_field = result.get("data", [])
    if not isinstance(data_field, list):
        print(
            "[error] invalid output data; expected a JSON byte array", file=sys.stderr
        )
        return 1

    raw = bytes(int(x) & 0xFF for x in data_field)
    samples = unpack_f32_native(raw)

    shape = result.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        print(f"Output tensor shape: {shape} ({len(samples)} samples)")
    else:
        print(f"Output tensor shape: {shape} ({len(samples)} samples)")

    out_path = Path(args.output_wav)
    try:
        wav_write_int16_mono(out_path, samples, sample_rate=int(args.sample_rate))
    except Exception as exc:
        print(f"[error] failed to write wav {out_path}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {out_path} ({args.sample_rate}Hz mono int16). infer_time={elapsed_s:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
