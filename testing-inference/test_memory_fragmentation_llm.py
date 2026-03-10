#!/usr/bin/env python3
"""
LLM-focused memory fragmentation probe for kapsl-runtime.

This wrapper builds a string payload for an LLM model and delegates execution to
`test_memory_fragmentation.py` so we reuse the existing measurement logic.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_LLM_NAME_KEYWORDS = [
    "llm",
    "llama",
    "gemma",
    "gpt",
    "deepseek",
    "qwen",
    "mistral",
]


def auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def http_get_json(url: str, token: Optional[str], timeout: float) -> Any:
    req = urllib.request.Request(url, headers=auth_headers(token), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


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


def model_is_active(model: Dict[str, Any]) -> bool:
    return str(model.get("status", "")).lower() == "active"


def model_id(model: Dict[str, Any]) -> Optional[int]:
    try:
        return int(model.get("id"))
    except (TypeError, ValueError):
        return None


def base_model_id(model: Dict[str, Any]) -> Optional[int]:
    raw = model.get("base_model_id")
    if raw is None:
        # Some payloads omit base_model_id; treat as standalone base model.
        return model_id(model)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def model_is_base_model(model: Dict[str, Any]) -> bool:
    mid = model_id(model)
    base = base_model_id(model)
    if mid is None:
        return False
    if base is None:
        return True
    return mid == base


def model_name(model: Dict[str, Any]) -> str:
    return str(model.get("name", ""))


def list_models(
    base_url: str, token: Optional[str], timeout_seconds: float
) -> List[Dict[str, Any]]:
    payload = http_get_json(
        f"{base_url.rstrip('/')}/api/models", token, timeout_seconds
    )
    if not isinstance(payload, list):
        raise RuntimeError(f"Invalid /api/models response: {type(payload).__name__}")
    return [entry for entry in payload if isinstance(entry, dict)]


def choose_model(
    models: List[Dict[str, Any]],
    forced_model_id: Optional[int],
    preferred_name_filters: List[str],
) -> Tuple[int, str]:
    if forced_model_id is not None:
        for model in models:
            try:
                model_id = int(model.get("id"))
            except (TypeError, ValueError):
                continue
            if model_id == forced_model_id:
                if not model_is_active(model):
                    raise RuntimeError(
                        f"Model {forced_model_id} is not active (status={model.get('status')})"
                    )
                return model_id, model_name(model) or str(model_id)
        raise RuntimeError(f"Model id {forced_model_id} was not found in /api/models")

    active_models = [m for m in models if model_is_active(m)]
    if not active_models:
        raise RuntimeError("No active models found")

    filters = [f.strip().lower() for f in preferred_name_filters if f.strip()]
    if filters:
        for model in active_models:
            name = model_name(model).lower()
            if all(keyword in name for keyword in filters):
                return int(model["id"]), model_name(model) or str(model["id"])

    for model in active_models:
        name = model_name(model).lower()
        if any(keyword in name for keyword in DEFAULT_LLM_NAME_KEYWORDS):
            return int(model["id"]), model_name(model) or str(model["id"])

    names = [model_name(m) for m in active_models]
    raise RuntimeError(
        "Could not auto-select an LLM model. "
        "Use --model-id or --model-name-contains. "
        f"Active models: {names}"
    )


def parse_model_ids_text(value: Optional[str]) -> List[int]:
    if not value:
        return []
    ids: List[int] = []
    for raw in value.replace(",", " ").split():
        try:
            ids.append(int(raw))
        except ValueError as err:
            raise ValueError(f"Invalid model id '{raw}' in --model-ids") from err
    # Dedup in order.
    seen = set()
    deduped: List[int] = []
    for mid in ids:
        if mid in seen:
            continue
        seen.add(mid)
        deduped.append(mid)
    return deduped


def choose_models(
    models: List[Dict[str, Any]],
    forced_model_ids: Sequence[int],
    preferred_name_filters: List[str],
    max_models: Optional[int],
) -> List[Tuple[int, str]]:
    if forced_model_ids:
        selected: List[Tuple[int, str]] = []
        for forced in forced_model_ids:
            found = None
            for model in models:
                if model_id(model) == forced:
                    found = model
                    break
            if not found:
                raise RuntimeError(f"Model id {forced} was not found in /api/models")
            if not model_is_active(found):
                raise RuntimeError(
                    f"Model {forced} is not active (status={found.get('status')})"
                )
            selected.append((forced, model_name(found) or str(forced)))
        return selected

    active_models = [m for m in models if model_is_active(m)]
    active_base_models = [m for m in active_models if model_is_base_model(m)]
    if not active_base_models:
        raise RuntimeError("No active base models found")

    filters = [f.strip().lower() for f in preferred_name_filters if f.strip()]
    selected_models: List[Dict[str, Any]] = []

    if filters:
        for model in active_base_models:
            name = model_name(model).lower()
            if all(keyword in name for keyword in filters):
                selected_models.append(model)
    else:
        for model in active_base_models:
            name = model_name(model).lower()
            if any(keyword in name for keyword in DEFAULT_LLM_NAME_KEYWORDS):
                selected_models.append(model)

    if not selected_models:
        names = [model_name(m) for m in active_base_models]
        raise RuntimeError(
            "Could not auto-select any LLM models. "
            "Use --model-ids/--model-id or --model-name-contains. "
            f"Active base models: {names}"
        )

    # Deterministic ordering.
    selected_models.sort(key=lambda m: (model_name(m).lower(), model_id(m) or 0))

    if max_models is not None and max_models > 0:
        selected_models = selected_models[:max_models]

    return [(int(m["id"]), model_name(m) or str(m["id"])) for m in selected_models]


def make_llm_payload(prompt: str) -> Dict[str, Any]:
    data = prompt.encode("utf-8")
    return {
        "input": {
            "shape": [1, len(data)],
            "dtype": "string",
            "data_base64": base64.b64encode(data).decode("ascii"),
        },
        "session_id": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run memory fragmentation test using an LLM string payload."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9095")
    parser.add_argument(
        "--model-id",
        type=int,
        default=None,
        help="Specific model id to test. If omitted, this script auto-selects an active LLM model.",
    )
    parser.add_argument(
        "--model-ids",
        default=None,
        help="Comma/space-separated model ids to test together (example: 0,1,2).",
    )
    parser.add_argument(
        "--model-name-contains",
        action="append",
        default=[],
        help=(
            "Name keyword filter for auto-selection. Repeatable; all provided keywords must match. "
            "(example: --model-name-contains llm --model-name-contains llama)"
        ),
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Limit the number of auto-selected models (0 means no limit).",
    )
    parser.add_argument(
        "--prompt",
        default="Write one short sentence about memory fragmentation in LLM inference.",
        help="Prompt encoded as string input payload.",
    )
    parser.add_argument(
        "--random-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Randomize prompt per request (delegated to test_memory_fragmentation.py) "
            "to better mimic production fragmentation."
        ),
    )
    parser.add_argument(
        "--prompt-min-bytes",
        type=int,
        default=64,
        help="Minimum random prompt size in bytes (ASCII).",
    )
    parser.add_argument(
        "--prompt-max-bytes",
        type=int,
        default=1024,
        help="Maximum random prompt size in bytes (ASCII).",
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic prompt generation.",
    )
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests-per-worker", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--idle-poll-interval", type=float, default=0.3)
    parser.add_argument("--idle-stable-samples", type=int, default=3)
    parser.add_argument(
        "--token",
        default=None,
        help="Optional API token (falls back to KAPSL_API_TOKEN / KAPSL_DESKTOP_API_TOKEN; legacy KAPSL_* also works).",
    )
    parser.add_argument("--pid", type=int, default=None, help="Optional kapsl PID.")
    parser.add_argument(
        "--process-pattern",
        default="(^|/)kapsl( |$)",
        help="Pattern used by pgrep -f when --pid is not set.",
    )
    parser.add_argument(
        "--force-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set request.metadata.force_cpu=true for all requests.",
    )
    parser.add_argument(
        "--unique-request-id-per-request",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate unique request_id for each request.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = resolve_token(args.token)
    base_url = args.base_url.rstrip("/")
    script_dir = Path(__file__).resolve().parent
    core_script = script_dir / "test_memory_fragmentation.py"

    if not core_script.exists():
        print(f"Required script not found: {core_script}")
        return 2

    try:
        models = list_models(base_url, token, args.timeout_seconds)
        forced_ids = []
        forced_ids.extend(parse_model_ids_text(args.model_ids))
        if args.model_id is not None:
            forced_ids.insert(0, int(args.model_id))
        forced_ids = list(dict.fromkeys(forced_ids))  # stable dedupe
        max_models = (
            args.max_models if args.max_models and args.max_models > 0 else None
        )

        selected = choose_models(
            models=models,
            forced_model_ids=forced_ids,
            preferred_name_filters=args.model_name_contains,
            max_models=max_models,
        )
    except Exception as err:  # noqa: BLE001
        print(f"Failed to select LLM model(s): {err}")
        return 2

    payload_files: List[str] = []
    payload_files_by_model: Dict[int, str] = {}
    try:
        for model_id, _model_name in selected:
            # Base payload: if --random-prompt is enabled, the core probe will
            # replace input.data per request. We still need a valid string payload
            # template on disk.
            payload = make_llm_payload(args.prompt)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"frag_llm_payload_m{model_id}_",
                delete=False,
                encoding="utf-8",
            ) as handle:
                json.dump(payload, handle, separators=(",", ":"))
                payload_files.append(handle.name)
                payload_files_by_model[model_id] = handle.name

        model_ids = [mid for mid, _ in selected]
        cmd = [
            sys.executable,
            str(core_script),
            "--base-url",
            base_url,
            "--model-ids",
            ",".join(str(mid) for mid in model_ids),
            "--cycles",
            str(args.cycles),
            "--concurrency",
            str(args.concurrency),
            "--requests-per-worker",
            str(args.requests_per_worker),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--idle-timeout-seconds",
            str(args.idle_timeout_seconds),
            "--idle-poll-interval",
            str(args.idle_poll_interval),
            "--idle-stable-samples",
            str(args.idle_stable_samples),
            "--process-pattern",
            args.process_pattern,
            "--fail-threshold-pct",
            str(args.fail_threshold_pct),
        ]
        if args.random_prompt:
            cmd.append("--randomize-string-input")
            cmd.extend(["--string-min-bytes", str(args.prompt_min_bytes)])
            cmd.extend(["--string-max-bytes", str(args.prompt_max_bytes)])
            if args.prompt_seed is not None:
                cmd.extend(["--string-seed", str(args.prompt_seed)])
        for model_id in model_ids:
            cmd.extend(
                ["--payload-by-model", f"{model_id}={payload_files_by_model[model_id]}"]
            )
        if args.csv_output:
            cmd.extend(["--csv-output", args.csv_output])
        if token:
            cmd.extend(["--token", token])
        if args.pid is not None:
            cmd.extend(["--pid", str(args.pid)])

        cmd.append("--force-cpu" if args.force_cpu else "--no-force-cpu")
        cmd.append(
            "--unique-request-id-per-request"
            if args.unique_request_id_per_request
            else "--no-unique-request-id-per-request"
        )
        cmd.append("--freeze-scaling" if args.freeze_scaling else "--no-freeze-scaling")
        cmd.append(
            "--freeze-scaling-strict"
            if args.freeze_scaling_strict
            else "--no-freeze-scaling-strict"
        )

        print("LLM Fragmentation Wrapper")
        print(f"  selected_model_ids: {model_ids}")
        print(
            "  selected_model_names: "
            + ", ".join(f"{mid}={name}" for mid, name in selected)
        )
        if args.random_prompt:
            print(f"  random_prompt: true (seed={args.prompt_seed})")
            print(
                f"  prompt_bytes_range: [{args.prompt_min_bytes}, {args.prompt_max_bytes}]"
            )
        else:
            print(f"  random_prompt: false")
            print(f"  prompt_chars: {len(args.prompt)}")
        print(f"  payload_files_by_model: {payload_files_by_model}")
        print(f"  freeze_scaling: {args.freeze_scaling}")
        print(f"  command: {' '.join(cmd)}")
        print()

        completed = subprocess.run(cmd)
        return int(completed.returncode)
    finally:
        for path in payload_files:
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
