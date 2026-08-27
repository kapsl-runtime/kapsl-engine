#!/usr/bin/env python3
"""End-to-end exact-memory, VMM-resize, and wire-path GPU conformance.

The runtime and managed vLLM child are launched by the workflow. This probe
drives only public HTTP traffic, reads Prometheus evidence, and deliberately
crashes the private vLLM generation once. It never reaches into the KV control
socket or fabricates resize acknowledgements.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


MEMORY_METRICS = {
    "requested_bytes": "kapsl_managed_vllm_kv_requested_bytes",
    "granted_bytes": "kapsl_managed_vllm_kv_granted_bytes",
    "minimum_bytes": "kapsl_managed_vllm_kv_minimum_bytes",
    "backing_bytes": "kapsl_managed_vllm_kv_backing_bytes",
    "total_blocks": "kapsl_managed_vllm_kv_blocks_total",
    "allocated_blocks": "kapsl_managed_vllm_kv_blocks_allocated",
    "active_blocks": "kapsl_managed_vllm_kv_blocks_active",
    "idle_blocks": "kapsl_managed_vllm_kv_blocks_idle",
    "quarantine_bytes": "kapsl_managed_vllm_kv_quarantine_bytes",
}
GENERATION_METRIC = "kapsl_managed_vllm_restart_generation"
CANCELLATION_METRIC = "kapsl_managed_vllm_bridge_cancellations_total"
WIRE_REQUEST_METRIC = "kapsl_managed_vllm_bridge_requests_total"
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
_VMM_EVIDENCE_RE = re.compile(
    r"KAPSL_VMM_CONFORMANCE stable_address=(0x[0-9a-f]+) "
    r"mapped_bytes=([0-9]+) virtual_bytes=([0-9]+) phase=(initial|grow|shrink)"
)
_MANAGED_VLLM_LOG_RE = re.compile(
    r"Managed vLLM process started:.*?\blog=([^\r\n]+)"
)
_MANAGED_VLLM_ENDPOINT_RE = re.compile(
    r"Managed vLLM process started:.*?\bendpoint=(http://\S+?)\s+log="
)


class ConformanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetricSample:
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class MemorySnapshot:
    row_keys: list[str]
    requested_bytes: list[int]
    granted_bytes: list[int]
    minimum_bytes: list[int]
    backing_bytes: list[int]
    total_blocks: list[int]
    allocated_blocks: list[int]
    active_blocks: list[int]
    idle_blocks: list[int]
    quarantine_bytes: list[int]
    restart_generation: int


def parse_prometheus(text: str) -> dict[str, list[MetricSample]]:
    parsed: dict[str, list[MetricSample]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        left, separator, raw_value = line.rpartition(" ")
        if not separator:
            raise ConformanceError(f"invalid Prometheus sample: {line!r}")
        if "{" in left:
            name, raw_labels = left.split("{", 1)
            if not raw_labels.endswith("}"):
                raise ConformanceError(f"invalid Prometheus labels: {line!r}")
            labels = {
                key: bytes(value, "utf-8").decode("unicode_escape")
                for key, value in _LABEL_RE.findall(raw_labels[:-1])
            }
        else:
            name, labels = left, {}
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ConformanceError(f"invalid Prometheus value: {line!r}") from error
        parsed.setdefault(name, []).append(MetricSample(labels, value))
    return parsed


def _get_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _metric_integer_map(
    metrics: dict[str, list[MetricSample]],
    name: str,
    *,
    labels: dict[str, str],
    key_labels: tuple[str, ...],
) -> dict[tuple[str, ...], int]:
    samples = [
        sample
        for sample in metrics.get(name, [])
        if all(sample.labels.get(key) == value for key, value in labels.items())
    ]
    if not samples:
        raise ConformanceError(f"Prometheus metric {name!r} has no matching samples")
    values: dict[tuple[str, ...], int] = {}
    for sample in samples:
        if not sample.value.is_integer():
            raise ConformanceError(f"metric {name!r} is not integral: {sample.value}")
        try:
            key = tuple(sample.labels[label] for label in key_labels)
        except KeyError as error:
            raise ConformanceError(
                f"metric {name!r} is missing identity label {error.args[0]!r}"
            ) from error
        if key in values:
            raise ConformanceError(f"metric {name!r} repeats identity {key!r}")
        values[key] = int(sample.value)
    return values


def collect_memory_snapshot(metrics_url: str, model: str) -> MemorySnapshot:
    metrics = parse_prometheus(_get_text(metrics_url))
    keyed_rows = {
        field: _metric_integer_map(
            metrics,
            metric,
            labels={"model": model},
            key_labels=("replica", "device"),
        )
        for field, metric in MEMORY_METRICS.items()
    }
    row_keys = set(next(iter(keyed_rows.values())))
    for field, values in keyed_rows.items():
        if set(values) != row_keys:
            raise ConformanceError(
                f"managed-vLLM memory metric {field} identities differ: "
                f"{sorted(values)} != {sorted(row_keys)}"
            )
    ordered_keys = sorted(row_keys)
    generations = _metric_integer_map(
        metrics,
        GENERATION_METRIC,
        labels={"model": model},
        key_labels=("replica",),
    )
    expected_replicas = {(key[0],) for key in ordered_keys}
    if set(generations) != expected_replicas:
        raise ConformanceError(
            "managed-vLLM restart-generation identities differ from memory rows: "
            f"{sorted(generations)} != {sorted(expected_replicas)}"
        )
    generation_values = list(generations.values())
    if len(set(generation_values)) != 1:
        raise ConformanceError(
            "managed-vLLM replicas disagree on restart generation: "
            f"{generation_values}"
        )
    rows = {
        field: [values[key] for key in ordered_keys]
        for field, values in keyed_rows.items()
    }
    return MemorySnapshot(
        row_keys=[f"replica={replica},device={device}" for replica, device in ordered_keys],
        **rows,
        restart_generation=generation_values[0],
    )


def validate_memory_snapshot(
    snapshot: MemorySnapshot, *, elastic: bool, exact_initial_grant: bool = False
) -> int:
    row_count = len(snapshot.backing_bytes)
    if row_count == 0:
        raise ConformanceError("managed-vLLM emitted no device memory rows")
    if len(snapshot.row_keys) != row_count or len(set(snapshot.row_keys)) != row_count:
        raise ConformanceError(
            f"managed-vLLM memory row identities are incomplete or repeated: {snapshot.row_keys}"
        )
    for field in MEMORY_METRICS:
        values = getattr(snapshot, field)
        if len(values) != row_count:
            raise ConformanceError(
                f"managed-vLLM memory metric {field} has {len(values)} rows; expected {row_count}"
            )
    if any(value != 0 for value in snapshot.quarantine_bytes):
        raise ConformanceError(
            f"managed-vLLM quarantined clean conformance memory: {snapshot.quarantine_bytes}"
        )
    strides: list[int] = []
    for index in range(row_count):
        granted = snapshot.granted_bytes[index]
        minimum = snapshot.minimum_bytes[index]
        backing = snapshot.backing_bytes[index]
        allocated = snapshot.allocated_blocks[index]
        total = snapshot.total_blocks[index]
        active = snapshot.active_blocks[index]
        idle = snapshot.idle_blocks[index]
        if not (0 < minimum <= granted) or backing < minimum:
            raise ConformanceError(
                f"row {index} violates exact byte accounting: minimum={minimum} granted={granted} backing={backing}"
            )
        if exact_initial_grant and granted != backing:
            raise ConformanceError(
                f"row {index} initial physical backing does not equal its exact grant: granted={granted} backing={backing}"
            )
        if snapshot.requested_bytes[index] < granted:
            raise ConformanceError(
                f"row {index} grant exceeds its requested bytes"
            )
        if not (0 < allocated <= total) or active > allocated or idle != allocated - active:
            raise ConformanceError(
                f"row {index} has inconsistent block accounting: total={total} allocated={allocated} active={active} idle={idle}"
            )
        if backing % allocated:
            raise ConformanceError(
                f"row {index} backing is not an integral physical block count"
            )
        stride = backing // allocated
        if minimum % stride:
            raise ConformanceError(
                f"row {index} minimum is not block aligned"
            )
        if elastic and allocated >= total:
            raise ConformanceError(
                f"row {index} did not reserve virtual live-resize headroom"
            )
        strides.append(stride)
    if len(set(strides)) != 1:
        raise ConformanceError(f"tensor-parallel block strides diverged: {strides}")
    return strides[0]


def _post_json(url: str, body: dict[str, Any], timeout: float = 120.0) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers.items()), error.read()


def _completion_body(model: str, *, max_tokens: int, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Write a deterministic sequence of integers separated by spaces.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 17,
        "stream": stream,
        # This vLLM-compatible extension is intentionally kept in the
        # normalized wire body so the growth request lasts through a supervisor tick.
        "ignore_eos": True,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _require_completion(
    url: str,
    model: str,
    max_tokens: int = 32,
    *,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status, _, raw = _post_json(
        url,
        body or _completion_body(model, max_tokens=max_tokens, stream=False),
    )
    if status != 200:
        raise ConformanceError(
            f"one-shot completion failed with HTTP {status}: {raw[:1000]!r}"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConformanceError("one-shot completion returned invalid JSON") from error
    if (
        not str(payload.get("id", "")).startswith("chatcmpl-")
        or payload.get("object") != "chat.completion"
        or not isinstance(payload.get("choices"), list)
        or not payload["choices"]
        or not isinstance(
            (payload["choices"][0].get("message") or {}).get("content"), str
        )
        or not isinstance(payload.get("usage"), dict)
        or int(payload["usage"].get("completion_tokens", 0)) <= 0
    ):
        raise ConformanceError(f"one-shot OpenAI semantics are incomplete: {payload!r}")
    return payload


def _managed_vllm_endpoint(runtime_log: Path) -> str:
    text = runtime_log.read_text(encoding="utf-8", errors="replace")
    endpoints = _MANAGED_VLLM_ENDPOINT_RE.findall(text)
    if not endpoints:
        raise ConformanceError("runtime log contains no managed vLLM endpoint")
    endpoint = endpoints[-1].rstrip("/")
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ConformanceError(
            f"managed vLLM endpoint is not a private loopback origin: {endpoint!r}"
        )
    return endpoint


def _vllm_counter(endpoint: str, name: str) -> int:
    metrics = parse_prometheus(_get_text(endpoint.rstrip("/") + "/metrics"))
    samples = metrics.get(name, [])
    if not samples or any(not sample.value.is_integer() for sample in samples):
        raise ConformanceError(f"vLLM counter {name!r} is missing or non-integral")
    return sum(int(sample.value) for sample in samples)


def _require_prefix_cache_hit(
    completion_url: str, model: str, runtime_log: Path
) -> dict[str, Any]:
    endpoint = _managed_vllm_endpoint(runtime_log)
    query_metric = "vllm:prefix_cache_queries"
    hit_metric = "vllm:prefix_cache_hits"
    queries_before = _vllm_counter(endpoint, query_metric)
    prompt = (
        "Preserve this deterministic validation prefix and reply with OK: "
        + "KAPSL_PREFIX_BLOCK " * 160
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 31,
        "stream": False,
    }
    first = _require_completion(completion_url, model, body=body)
    queries_after_first = _wait_until(
        "the first native prefix-cache query",
        lambda: (
            value
            if (value := _vllm_counter(endpoint, query_metric)) > queries_before
            else None
        ),
        timeout=30,
    )
    hits_after_first = _vllm_counter(endpoint, hit_metric)
    second = _require_completion(completion_url, model, body=body)
    hits_after_second = _wait_until(
        "a native prefix-cache hit for an identical request",
        lambda: (
            value
            if (value := _vllm_counter(endpoint, hit_metric)) > hits_after_first
            else None
        ),
        timeout=30,
    )
    return {
        "endpoint": endpoint,
        "queries_before": queries_before,
        "queries_after_first": queries_after_first,
        "hits_after_first": hits_after_first,
        "hits_after_second": hits_after_second,
        "first_completion_id": first["id"],
        "second_completion_id": second["id"],
    }


def _load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ConformanceError(
            "token-level conformance requires the pinned transformers package"
        ) from error
    return AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )


def _cross_token_stop_string(content: str, model_dir: Path) -> tuple[str, int, int]:
    tokenizer = _load_tokenizer(model_dir)
    encoded = tokenizer(
        content,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if not isinstance(offsets, list):
        raise ConformanceError("the certified tokenizer returned no offset mapping")
    for left, right in zip(offsets, offsets[1:]):
        if (
            not isinstance(left, tuple)
            or not isinstance(right, tuple)
            or len(left) != 2
            or len(right) != 2
        ):
            continue
        left_start, boundary = (int(left[0]), int(left[1]))
        right_start, right_end = (int(right[0]), int(right[1]))
        start = max(left_start, boundary - 1)
        end = min(right_end, max(boundary + 1, right_start + 1))
        candidate = content[start:end]
        if (
            left_start < boundary <= right_end
            and start < boundary < end
            and candidate
            and content.count(candidate) == 1
            and content[:start]
        ):
            return candidate, start, boundary
    raise ConformanceError(
        "deterministic completion exposed no unique stop string spanning two tokens"
    )


def _require_cross_token_stop(
    completion_url: str, model: str, model_dir: Path
) -> dict[str, Any]:
    baseline_body = _completion_body(model, max_tokens=128, stream=False)
    baseline = _require_completion(
        completion_url, model, body=baseline_body
    )
    baseline_content = str(baseline["choices"][0]["message"]["content"])
    stop, start, boundary = _cross_token_stop_string(baseline_content, model_dir)
    stopped_body = dict(baseline_body)
    stopped_body["stop"] = stop
    stopped = _require_completion(completion_url, model, body=stopped_body)
    choice = stopped["choices"][0]
    stopped_content = str((choice.get("message") or {}).get("content", ""))
    if choice.get("finish_reason") != "stop":
        raise ConformanceError(
            f"cross-token stop returned finish_reason={choice.get('finish_reason')!r}"
        )
    if stop in stopped_content or stopped_content != baseline_content[:start]:
        raise ConformanceError(
            "cross-token stop was not enforced at the exact deterministic boundary"
        )
    return {
        "baseline_completion_id": baseline["id"],
        "stopped_completion_id": stopped["id"],
        "stop_string": stop,
        "stop_start_character": start,
        "token_boundary_character": boundary,
        "stopped_characters": len(stopped_content),
        "finish_reason": choice.get("finish_reason"),
    }


def _require_full_context_request(
    completion_url: str,
    model: str,
    model_dir: Path,
    max_model_len: int,
) -> dict[str, Any]:
    if max_model_len < 128:
        raise ConformanceError("full-context conformance requires max_model_len >= 128")
    tokenizer = _load_tokenizer(model_dir)

    def messages(repetitions: int) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": "capacity " * repetitions
                + "Reply with short deterministic validation text.",
            }
        ]

    target_prompt = max_model_len - 32
    lower, upper = 1, max_model_len * 4
    selected_messages: list[dict[str, str]] | None = None
    selected_prompt_tokens = 0
    while lower <= upper:
        middle = (lower + upper) // 2
        candidate = messages(middle)
        token_ids = tokenizer.apply_chat_template(
            candidate,
            tokenize=True,
            add_generation_prompt=True,
        )
        count = len(token_ids)
        if count <= target_prompt:
            selected_messages = candidate
            selected_prompt_tokens = count
            lower = middle + 1
        else:
            upper = middle - 1
    if selected_messages is None or selected_prompt_tokens <= 0:
        raise ConformanceError("could not construct a bounded full-context prompt")
    completion_tokens = max_model_len - selected_prompt_tokens
    if completion_tokens < 32:
        raise ConformanceError(
            "full-context prompt left too little deterministic generation capacity"
        )
    body = {
        "model": model,
        "messages": selected_messages,
        "max_tokens": completion_tokens,
        "temperature": 0,
        "seed": 47,
        "stream": False,
        "ignore_eos": True,
    }
    payload = _require_completion(completion_url, model, body=body)
    usage = payload["usage"]
    actual_prompt = int(usage.get("prompt_tokens", -1))
    actual_completion = int(usage.get("completion_tokens", -1))
    actual_total = int(usage.get("total_tokens", -1))
    if (
        actual_prompt != selected_prompt_tokens
        or actual_completion != completion_tokens
        or actual_total != max_model_len
    ):
        raise ConformanceError(
            "one-sequence minimum did not serve the exact full context: "
            f"planned=({selected_prompt_tokens},{completion_tokens},{max_model_len}) "
            f"actual=({actual_prompt},{actual_completion},{actual_total})"
        )
    return {
        "completion_id": payload["id"],
        "prompt_tokens": actual_prompt,
        "completion_tokens": actual_completion,
        "total_tokens": actual_total,
    }


def _require_stream(url: str, model: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=120
    )
    body = json.dumps(
        _completion_body(model, max_tokens=64, stream=True), separators=(",", ":")
    )
    connection.request(
        "POST",
        parsed.path,
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    raw = response.read()
    headers = dict(response.getheaders())
    connection.close()
    if response.status != 200:
        raise ConformanceError(
            f"streaming completion failed with HTTP {response.status}: {raw[:1000]!r}"
        )
    events = []
    done = False
    for line in raw.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        events.append(json.loads(data))
    if not done or not events:
        raise ConformanceError("streaming response omitted data events or [DONE]")
    ids = {event.get("id") for event in events if event.get("id") is not None}
    usage_events = [event for event in events if isinstance(event.get("usage"), dict)]
    if len(ids) != 1 or not str(next(iter(ids))).startswith("chatcmpl-"):
        raise ConformanceError(f"vLLM streaming completion IDs were not preserved: {ids}")
    if not usage_events or int(usage_events[-1]["usage"].get("completion_tokens", 0)) <= 0:
        raise ConformanceError("vLLM exact streaming usage was not preserved")
    return {
        "completion_id": next(iter(ids)),
        "events": len(events),
        "bytes": len(raw),
        "content_type": headers.get("Content-Type") or headers.get("content-type"),
        "usage": usage_events[-1]["usage"],
    }


def _wait_until(
    description: str,
    predicate: Callable[[], Any],
    *,
    timeout: float,
    interval: float = 0.5,
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
    detail = f"; last error: {last_error}" if last_error else ""
    raise ConformanceError(f"timed out waiting for {description}{detail}")


def _bridge_counter(metrics_url: str, name: str) -> int:
    metrics = parse_prometheus(_get_text(metrics_url))
    samples = [
        sample
        for sample in metrics.get(name, [])
        if sample.labels.get("mode") == "wire"
    ]
    if not samples:
        return 0
    if any(not sample.value.is_integer() for sample in samples):
        raise ConformanceError(f"bridge counter {name!r} is not integral")
    return sum(int(sample.value) for sample in samples)


def _cancel_stream(url: str, model: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=30
    )
    body = json.dumps(
        _completion_body(model, max_tokens=512, stream=True), separators=(",", ":")
    )
    connection.request(
        "POST",
        parsed.path,
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raw = response.read()
        connection.close()
        raise ConformanceError(
            f"cancellation stream failed to start: HTTP {response.status} {raw[:1000]!r}"
        )
    first = response.read(256)
    if b"data:" not in first:
        connection.close()
        raise ConformanceError("cancellation stream returned no SSE data")
    response.close()
    connection.close()


def _managed_vllm_pid(runtime_pid: int) -> int:
    seen: set[int] = set()
    pending = [runtime_pid]
    candidates: list[int] = []
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in children_path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError):
            children = []
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            pending.append(child)
            try:
                command = Path(f"/proc/{child}/cmdline").read_bytes().replace(b"\0", b" ")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if b"vllm.entrypoints.openai.api_server" in command:
                candidates.append(child)
    if len(candidates) != 1:
        raise ConformanceError(
            f"expected one managed vLLM API child below runtime {runtime_pid}, found {candidates}"
        )
    return candidates[0]


def _managed_vllm_log_paths(runtime_log: Path, log_root: Path) -> list[Path]:
    root = log_root.resolve()
    runtime_text = runtime_log.read_text(encoding="utf-8", errors="replace")
    candidates = {path.resolve() for path in root.rglob("vllm.log") if path.is_file()}
    for raw_path in _MANAGED_VLLM_LOG_RE.findall(runtime_text):
        candidate = Path(raw_path.strip()).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ConformanceError(
                f"managed vLLM log escaped the declared state root: {candidate}"
            ) from error
        if candidate.is_file():
            candidates.add(candidate)
    if not candidates:
        raise ConformanceError(
            f"no managed vLLM child logs were found beneath {log_root}"
        )
    return sorted(candidates)


def _validate_vmm_logs(
    runtime_log: Path, log_root: Path, *, minimum_resize_cycles: int = 1
) -> dict[str, Any]:
    child_logs = _managed_vllm_log_paths(runtime_log, log_root)
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in child_logs
    )
    by_address: dict[str, set[str]] = {}
    phase_counts: dict[str, dict[str, int]] = {}
    virtual_sizes: set[int] = set()
    for address, _, virtual, phase in _VMM_EVIDENCE_RE.findall(text):
        by_address.setdefault(address, set()).add(phase)
        counts = phase_counts.setdefault(
            address, {"initial": 0, "grow": 0, "shrink": 0}
        )
        counts[phase] += 1
        virtual_sizes.add(int(virtual))
    if not by_address:
        raise ConformanceError("runtime log contains no CUDA VMM conformance evidence")
    incomplete_resize = {
        address: sorted(phases)
        for address, phases in by_address.items()
        if ("grow" in phases or "shrink" in phases)
        and not {"initial", "grow", "shrink"}.issubset(phases)
    }
    complete = sorted(
        address
        for address, phases in by_address.items()
        if {"initial", "grow", "shrink"}.issubset(phases)
    )
    if incomplete_resize or not complete:
        raise ConformanceError(
            "CUDA virtual addresses did not survive every resize phase: "
            f"incomplete={incomplete_resize}, complete={complete}"
        )
    insufficient_churn = {
        address: counts
        for address, counts in phase_counts.items()
        if (counts["grow"] or counts["shrink"])
        and (
            counts["grow"] < minimum_resize_cycles
            or counts["shrink"] < minimum_resize_cycles
        )
    }
    if insufficient_churn:
        raise ConformanceError(
            "CUDA VMM workers did not complete the required resize churn: "
            f"{insufficient_churn}"
        )
    allocator_evidence = text.count("KAPSL_VMM_CONFORMANCE allocator_delta_bytes=0")
    if allocator_evidence < len(by_address):
        raise ConformanceError(
            "not every CUDA VMM worker proved zero PyTorch KV allocation delta"
        )
    if "quarantined blocks" in text or "failed to release CUDA VMM" in text:
        raise ConformanceError("runtime log reports a CUDA VMM quarantine/release failure")
    return {
        "child_logs": [str(path) for path in child_logs],
        "worker_virtual_addresses": sorted(by_address),
        "resize_stable_addresses": complete,
        "phase_counts": phase_counts,
        "virtual_sizes": sorted(virtual_sizes),
        "allocator_delta_zero_count": allocator_evidence,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.resize_cycles < 2:
        raise ConformanceError("resize conformance requires at least two churn cycles")
    profile_fields = args.profile.split(",")
    if len(profile_fields) != 4 or any(
        not field or any(character in field for character in "\r\n,")
        for field in profile_fields
    ):
        raise ConformanceError("elastic profile must be one safe four-field tuple")
    for name, digest in (
        ("adapter", args.adapter_build_id),
        ("backend", args.backend_build_id),
        ("runtime", args.runtime_build_id),
    ):
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ConformanceError(f"{name} build ID is not a canonical SHA-256 digest")
    completion_url = args.base_url.rstrip("/") + "/v1/chat/completions"
    health_url = args.base_url.rstrip("/") + "/api/health"
    _wait_until(
        "a healthy managed model",
        lambda: json.loads(_get_text(health_url)).get("unhealthy_models") == 0
        and json.loads(_get_text(health_url)).get("healthy_models", 0) >= 1,
        timeout=args.startup_timeout,
        interval=1.0,
    )
    initial = _wait_until(
        "managed-vLLM exact memory metrics",
        lambda: collect_memory_snapshot(args.metrics_url, args.model),
        timeout=30,
    )
    stride = validate_memory_snapshot(
        initial, elastic=True, exact_initial_grant=True
    )
    one_shot = _require_completion(completion_url, args.model)
    stream = _require_stream(completion_url, args.model)
    prefix_cache = _require_prefix_cache_hit(
        completion_url, args.model, args.runtime_log
    )
    cross_token_stop = _require_cross_token_stop(
        completion_url, args.model, args.model_dir
    )
    minimum_blocks = [value // stride for value in initial.minimum_bytes]
    resize_cycles: list[dict[str, Any]] = []
    long_results: list[dict[str, Any]] = []
    post_shrink_ids: list[str] = []
    for cycle in range(args.resize_cycles):
        growth_baseline = collect_memory_snapshot(args.metrics_url, args.model)
        long_result: dict[str, Any] = {}
        long_failure: list[BaseException] = []

        def generate_long() -> None:
            try:
                status, _, raw = _post_json(
                    completion_url,
                    _completion_body(args.model, max_tokens=900, stream=False),
                    timeout=args.resize_timeout,
                )
                if status != 200:
                    raise ConformanceError(
                        f"growth-driving completion failed with HTTP {status}: {raw[:1000]!r}"
                    )
                long_result.update(json.loads(raw))
            except BaseException as error:
                long_failure.append(error)

        request_thread = threading.Thread(target=generate_long, daemon=True)
        request_thread.start()

        def grown() -> MemorySnapshot | None:
            snapshot = collect_memory_snapshot(args.metrics_url, args.model)
            if snapshot.row_keys != growth_baseline.row_keys:
                raise ConformanceError(
                    "managed-vLLM device identities changed during live growth: "
                    f"{snapshot.row_keys} != {growth_baseline.row_keys}"
                )
            if all(
                current > previous
                for current, previous in zip(
                    snapshot.backing_bytes, growth_baseline.backing_bytes
                )
            ):
                return snapshot
            if not request_thread.is_alive() and long_failure:
                raise long_failure[0]
            return None

        grown_snapshot = _wait_until(
            f"live CUDA VMM growth cycle {cycle + 1}",
            grown,
            timeout=args.resize_timeout,
        )
        validate_memory_snapshot(grown_snapshot, elastic=False)
        if grown_snapshot.total_blocks != initial.total_blocks:
            raise ConformanceError(
                "live growth changed the stable virtual block capacity"
            )
        request_thread.join(timeout=args.resize_timeout)
        if request_thread.is_alive():
            raise ConformanceError("growth-driving completion did not finish")
        if long_failure:
            raise long_failure[0]

        def shrunk_to_minimum() -> MemorySnapshot | None:
            snapshot = collect_memory_snapshot(args.metrics_url, args.model)
            if snapshot.row_keys != initial.row_keys:
                raise ConformanceError(
                    "managed-vLLM device identities changed during live shrink: "
                    f"{snapshot.row_keys} != {initial.row_keys}"
                )
            return snapshot if snapshot.allocated_blocks == minimum_blocks else None

        shrunk = _wait_until(
            f"live CUDA VMM shrink cycle {cycle + 1} to the one-sequence minimum",
            shrunk_to_minimum,
            timeout=args.resize_timeout,
        )
        validate_memory_snapshot(shrunk, elastic=True)
        if shrunk.total_blocks != initial.total_blocks:
            raise ConformanceError(
                "live shrink changed the stable virtual block capacity"
            )
        post_shrink = _require_completion(completion_url, args.model)
        long_results.append(long_result)
        post_shrink_ids.append(post_shrink["id"])
        resize_cycles.append(
            {
                "cycle": cycle + 1,
                "growth_baseline": asdict(growth_baseline),
                "grown": asdict(grown_snapshot),
                "shrunk": asdict(shrunk),
            }
        )

    full_context_baseline = collect_memory_snapshot(args.metrics_url, args.model)
    if full_context_baseline.allocated_blocks != minimum_blocks:
        raise ConformanceError(
            "full-context request did not start from the one-sequence physical minimum"
        )
    full_context = _require_full_context_request(
        completion_url,
        args.model,
        args.model_dir,
        args.max_model_len,
    )
    _wait_until(
        "post-full-context return to the one-sequence physical minimum",
        lambda: (
            snapshot
            if (snapshot := collect_memory_snapshot(args.metrics_url, args.model)).allocated_blocks
            == minimum_blocks
            and all(value == 0 for value in snapshot.active_blocks)
            else None
        ),
        timeout=args.resize_timeout,
    )

    cancellation_before = _bridge_counter(args.metrics_url, CANCELLATION_METRIC)
    _cancel_stream(completion_url, args.model)
    _wait_until(
        "client-disconnect cancellation",
        lambda: _bridge_counter(args.metrics_url, CANCELLATION_METRIC)
        > cancellation_before,
        timeout=30,
    )
    _wait_until(
        "cancellation lease release",
        lambda: all(
            value == 0
            for value in collect_memory_snapshot(
                args.metrics_url, args.model
            ).active_blocks
        ),
        timeout=30,
    )

    generation_before = collect_memory_snapshot(
        args.metrics_url, args.model
    ).restart_generation
    child_pid = _managed_vllm_pid(args.runtime_pid)
    os.kill(child_pid, signal.SIGKILL)

    def restarted() -> MemorySnapshot | None:
        snapshot = collect_memory_snapshot(args.metrics_url, args.model)
        if snapshot.row_keys != initial.row_keys:
            raise ConformanceError(
                "managed-vLLM device identities changed across a supervised restart: "
                f"{snapshot.row_keys} != {initial.row_keys}"
            )
        if snapshot.restart_generation > generation_before and all(
            value > 0 for value in snapshot.backing_bytes
        ):
            return snapshot
        return None

    restarted_snapshot = _wait_until(
        "a clean exact-grant restart after forced child crash",
        restarted,
        timeout=args.restart_timeout,
        interval=1.0,
    )
    validate_memory_snapshot(
        restarted_snapshot, elastic=True, exact_initial_grant=True
    )
    post_restart = _require_completion(completion_url, args.model)
    final = collect_memory_snapshot(args.metrics_url, args.model)
    if any(final.quarantine_bytes):
        raise ConformanceError("clean supervised crash/restart left quarantine bytes")
    wire_requests = _bridge_counter(args.metrics_url, WIRE_REQUEST_METRIC)
    expected_wire_requests = 9 + 2 * args.resize_cycles
    if wire_requests < expected_wire_requests:
        raise ConformanceError(
            "wire-path request metric undercounted conformance traffic: "
            f"{wire_requests} < {expected_wire_requests}"
        )
    log_evidence = _validate_vmm_logs(
        args.runtime_log,
        args.vllm_log_root,
        minimum_resize_cycles=args.resize_cycles,
    )

    return {
        "schema_version": 1,
        "status": "passed",
        "model": args.model,
        "profile": {
            key: value
            for key, value in zip(
                ("adapter_id", "adapter_version", "backend_version", "profile_id"),
                profile_fields,
            )
        },
        "environment": {
            "adapter_build_id": args.adapter_build_id,
            "backend_build_id": args.backend_build_id,
            "runtime_build_id": args.runtime_build_id,
        },
        "gates": {
            "exact_initial_memory": True,
            "virtual_headroom": True,
            "native_inference_before_and_after_resize": True,
            "live_growth": True,
            "live_shrink_to_minimum": True,
            "full_max_context_at_minimum": True,
            "repeated_resize_churn": True,
            "stable_virtual_address": True,
            "zeroed_new_segments": True,
            "no_second_pytorch_kv_allocation": True,
            "wire_one_shot_and_stream_usage": True,
            "native_prefix_cache_hit": True,
            "cross_token_stop_enforcement": True,
            "disconnect_cancellation": True,
            "forced_crash_exact_replan": True,
            "clean_release_without_quarantine": True,
        },
        "block_stride_bytes": stride,
        "snapshots": {
            "initial": asdict(initial),
            "resize_cycles": resize_cycles,
            "restarted": asdict(restarted_snapshot),
            "final": asdict(final),
        },
        "openai": {
            "one_shot_id": one_shot["id"],
            "stream": stream,
            "prefix_cache": prefix_cache,
            "cross_token_stop": cross_token_stop,
            "full_context": full_context,
            "long_completion_ids": [result.get("id") for result in long_results],
            "post_shrink_ids": post_shrink_ids,
            "post_restart_id": post_restart["id"],
            "wire_requests": wire_requests,
        },
        "vmm_log_evidence": log_evidence,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-pid", type=int, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--vllm-log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowlist-output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--adapter-build-id", required=True)
    parser.add_argument("--backend-build-id", required=True)
    parser.add_argument("--runtime-build-id", required=True)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--resize-timeout", type=float, default=180)
    parser.add_argument("--restart-timeout", type=float, default=900)
    parser.add_argument("--resize-cycles", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=1024)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.allowlist_output.unlink(missing_ok=True)
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
    args.allowlist_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.allowlist_output.with_name(args.allowlist_output.name + ".tmp")
    temporary.write_text(args.profile + "\n", encoding="utf-8")
    os.replace(temporary, args.allowlist_output)
    print(json.dumps({"status": "passed", "report": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
