#!/usr/bin/env python3
"""Capture live request KV ownership, then prove disconnect reclamation."""

import argparse
import base64
import json
import os
import pathlib
import re
import time
import urllib.request
import uuid


def owner_usage(metrics, owner):
    pattern = (
        r'^kapsl_gpu_device_pool_owner_usage_bytes\{[^\n]*owner="'
        + re.escape(owner)
        + r'"[^\n]*\}\s+(\S+)$'
    )
    values = re.findall(pattern, metrics, re.M)
    return sum(int(float(value)) for value in values)


def probe(base_url, token, model_id, owner, output, poll_seconds=30):
    output.mkdir(parents=True, exist_ok=False)
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}

    def metrics():
        request = urllib.request.Request(base_url + "/metrics", headers=headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read().decode()

    prompt = "Continue counting integers in order, starting with one."
    payload = {
        "session_id": "live-kv-" + uuid.uuid4().hex,
        "input": {
            "shape": [1, 1],
            "dtype": "string",
            "data_base64": base64.b64encode(prompt.encode()).decode(),
        },
        "metadata": {"max_tokens": 1024, "min_tokens": 1024, "temperature": 0.0},
    }
    request = urllib.request.Request(
        base_url + f"/api/models/{model_id}/infer/stream",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    # Reading an event proves generation has begun. Metrics are captured while
    # the response remains open; completed requests are expected to own no KV.
    with urllib.request.urlopen(request, timeout=60) as stream:
        for raw in stream:
            if not raw.startswith(b"data:"):
                continue
            data = raw.removeprefix(b"data:").strip()
            if data == b"[DONE]":
                raise AssertionError(
                    "generation ended before live KV ownership was observed"
                )
            event = json.loads(data)
            if isinstance(event, dict) and event.get("error"):
                raise AssertionError(
                    "generation returned an error before KV ownership was observed"
                )
            (output / "first-event.json").write_text(json.dumps(event, indent=2) + "\n")
            active = metrics()
            (output / "metrics-active.txt").write_text(active)
            assert owner_usage(active, owner) > 0, (
                "no live allocation for the expected model/replica KV owner"
            )
            break
        else:
            raise AssertionError("generation stream produced no event")

    # Closing the unfinished stream cancels it. A missing owner is the normal
    # idle state when cross-session prefix caching is disabled.
    for attempt in range(6):
        idle = metrics()
        (output / f"metrics-cancelled-{attempt}.txt").write_text(idle)
        if owner_usage(idle, owner) == 0:
            return
        if attempt < 5:
            time.sleep(poll_seconds)
    raise AssertionError("cancelled generation retained request KV allocations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("KAPSL_GPU_TEST_API_TOKEN")
    if not token:
        parser.error("KAPSL_GPU_TEST_API_TOKEN is required")
    probe(args.base_url, token, args.model_id, args.owner, args.output)


if __name__ == "__main__":
    main()
