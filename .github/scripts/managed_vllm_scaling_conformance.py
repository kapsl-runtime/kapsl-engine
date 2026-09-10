#!/usr/bin/env python3
"""Drive a real managed-vLLM 1 -> 2 -> 1 scale cycle through public APIs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from managed_vllm_gpu_conformance import MEMORY_METRICS, parse_prometheus


class ScalingConformanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class KvRow:
    replica: int
    device: int
    requested_bytes: int
    granted_bytes: int
    minimum_bytes: int
    backing_bytes: int
    total_blocks: int
    allocated_blocks: int
    active_blocks: int
    idle_blocks: int
    quarantine_bytes: int


def _get_json(url: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _get_text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, body: dict[str, Any], timeout: float = 30.0) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        raise ScalingConformanceError(
            f"POST {url} failed with HTTP {error.code}: {raw[:1000]}"
        ) from error


def _integral_sample(sample, metric: str) -> int:
    if not sample.value.is_integer():
        raise ScalingConformanceError(
            f"metric {metric} is not integral: {sample.value}"
        )
    return int(sample.value)


def active_kv_rows(metrics_text: str, model: str) -> list[KvRow]:
    metrics = parse_prometheus(metrics_text)
    by_field: dict[str, dict[tuple[int, int], int]] = {}
    for field, metric in MEMORY_METRICS.items():
        rows: dict[tuple[int, int], int] = {}
        for sample in metrics.get(metric, []):
            if sample.labels.get("model") != model:
                continue
            try:
                raw_replica = sample.labels["replica"]
                raw_device = sample.labels["device"]
                if (
                    not raw_replica.isdigit()
                    or not raw_device.isdigit()
                    or str(int(raw_replica)) != raw_replica
                    or str(int(raw_device)) != raw_device
                ):
                    raise ValueError("noncanonical identity")
                key = (int(raw_replica), int(raw_device))
            except (KeyError, ValueError) as error:
                raise ScalingConformanceError(
                    f"metric {metric} omitted canonical replica/device labels"
                ) from error
            if key in rows:
                raise ScalingConformanceError(
                    f"metric {metric} duplicated replica/device row {key}"
                )
            rows[key] = _integral_sample(sample, metric)
        by_field[field] = rows

    active_keys = {
        key
        for key, value in by_field["backing_bytes"].items()
        if value > 0
    }
    active_keys.update(
        key
        for key, value in by_field["quarantine_bytes"].items()
        if value > 0
    )
    rows = []
    for key in sorted(active_keys):
        values: dict[str, int] = {}
        for field in MEMORY_METRICS:
            try:
                values[field] = by_field[field][key]
            except KeyError as error:
                raise ScalingConformanceError(
                    f"managed-vLLM metric {field} omitted active row {key}"
                ) from error
        rows.append(KvRow(replica=key[0], device=key[1], **values))
    return rows


def validate_exact_rows(rows: list[KvRow], expected_replicas: int) -> None:
    if not rows:
        raise ScalingConformanceError("managed-vLLM exposed no active KV rows")
    replica_ids = {row.replica for row in rows}
    if replica_ids != set(range(expected_replicas)):
        raise ScalingConformanceError(
            f"active KV replica labels {sorted(replica_ids)} != expected range {expected_replicas}"
        )
    devices_per_replica = {
        replica: {row.device for row in rows if row.replica == replica}
        for replica in replica_ids
    }
    if len({tuple(sorted(devices)) for devices in devices_per_replica.values()}) != 1:
        raise ScalingConformanceError(
            f"replica device topology diverged: {devices_per_replica}"
        )
    for row in rows:
        if row.quarantine_bytes != 0:
            raise ScalingConformanceError(f"replica has quarantined bytes: {row}")
        if not (0 < row.minimum_bytes <= row.granted_bytes == row.backing_bytes):
            raise ScalingConformanceError(f"replica exact grant mismatch: {row}")
        if row.requested_bytes < row.granted_bytes:
            raise ScalingConformanceError(f"replica grant exceeds request: {row}")
        if not (0 < row.allocated_blocks <= row.total_blocks):
            raise ScalingConformanceError(f"replica block accounting is invalid: {row}")
        if row.active_blocks > row.allocated_blocks:
            raise ScalingConformanceError(f"replica active blocks exceed allocation: {row}")
        if row.idle_blocks != row.allocated_blocks - row.active_blocks:
            raise ScalingConformanceError(f"replica idle block accounting is invalid: {row}")


def active_replicas(models: list[dict[str, Any]], model: str) -> tuple[int, list[int]]:
    primaries = [
        row
        for row in models
        if row.get("name") == model and int(row.get("replica_id", -1)) == 0
    ]
    if len(primaries) != 1:
        raise ScalingConformanceError(
            f"expected one primary registry row for {model}, found {len(primaries)}"
        )
    base_id = int(primaries[0]["base_model_id"])
    replicas = sorted(
        int(row["replica_id"])
        for row in models
        if int(row.get("base_model_id", -1)) == base_id
        and row.get("status") == "active"
    )
    return base_id, replicas


def _wait_until(
    description: str,
    predicate: Callable[[], Any],
    *,
    timeout: float,
    interval: float = 1.0,
) -> Any:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except BaseException as error:
            last_error = error
        time.sleep(interval)
    suffix = f"; last error: {last_error}" if last_error else ""
    raise ScalingConformanceError(f"timed out waiting for {description}{suffix}")


def _completion(base_url: str, model: str, index: int) -> str:
    payload = _post_json(
        base_url.rstrip("/") + "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Reply with the integer {index} and no other text.",
                }
            ],
            "max_tokens": 16,
            "temperature": 0,
            "seed": index,
            "stream": False,
        },
        timeout=180,
    )
    if not str(payload.get("id", "")).startswith("chatcmpl-"):
        raise ScalingConformanceError(f"completion ID was not preserved: {payload!r}")
    return str(payload["id"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    models_url = args.base_url.rstrip("/") + "/api/models"
    metrics_url = args.metrics_url
    base_id, initial_replicas = active_replicas(_get_json(models_url), args.model)
    if initial_replicas != [0]:
        raise ScalingConformanceError(
            f"scale test must start with one primary replica, found {initial_replicas}"
        )
    initial_rows = active_kv_rows(_get_text(metrics_url), args.model)
    validate_exact_rows(initial_rows, 1)
    scaling_url = args.base_url.rstrip("/") + f"/api/models/{base_id}/scaling"

    scale_up_policy = {
        "min_replicas": 2,
        "max_replicas": 2,
        "target_queue_depth": 1,
        "scale_down_threshold": 0,
        "cooldown_seconds": 1,
    }
    _post_json(scaling_url, scale_up_policy)

    def two_replicas():
        _, replicas = active_replicas(_get_json(models_url), args.model)
        if replicas != [0, 1]:
            return None
        rows = active_kv_rows(_get_text(metrics_url), args.model)
        validate_exact_rows(rows, 2)
        return rows

    scaled_rows = _wait_until(
        "two independently admitted managed-vLLM replicas",
        two_replicas,
        timeout=args.scale_timeout,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        completion_ids = list(
            executor.map(lambda index: _completion(args.base_url, args.model, index), range(8))
        )
    if len(set(completion_ids)) != len(completion_ids):
        raise ScalingConformanceError("scaled replicas returned duplicate completion IDs")

    scale_down_policy = {
        "min_replicas": 1,
        "max_replicas": 1,
        "target_queue_depth": 1,
        "scale_down_threshold": 100,
        "cooldown_seconds": 1,
    }
    _post_json(scaling_url, scale_down_policy)

    def one_replica():
        _, replicas = active_replicas(_get_json(models_url), args.model)
        if replicas != [0]:
            return None
        rows = active_kv_rows(_get_text(metrics_url), args.model)
        validate_exact_rows(rows, 1)
        return rows

    final_rows = _wait_until(
        "scale-down to one replica with released backing",
        one_replica,
        timeout=args.scale_timeout,
    )
    final_id = _completion(args.base_url, args.model, 99)
    return {
        "schema_version": 1,
        "status": "passed",
        "model": args.model,
        "model_id": base_id,
        "gates": {
            "one_to_two_to_one": True,
            "per_replica_exact_admission": True,
            "scaled_pool_inference": True,
            "scale_down_released_backing": True,
            "no_quarantine": True,
        },
        "initial_rows": [row.__dict__ for row in initial_rows],
        "scaled_rows": [row.__dict__ for row in scaled_rows],
        "final_rows": [row.__dict__ for row in final_rows],
        "scaled_completion_ids": completion_ids,
        "final_completion_id": final_id,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-timeout", type=float, default=900)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run(args)
    except BaseException as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "report": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
