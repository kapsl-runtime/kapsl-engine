#!/usr/bin/env python3
"""Greedy chat-completion parity between two running Kapsl servers.

`run` sends the same fixed prompts to one server, unary and streamed, and
records what came back. `compare` checks that the embedded llama.cpp route and
the signed llama-cpp/cpu pack produced the same text.

Both routes construct the same `kapsl_llm::GgufBackend` from the same locked
crates; the pack only adds the C ABI boundary. Greedy decoding on the same
host must therefore match token for token, so any difference is a defect in
the pack boundary (request translation, sampling parameters, streaming or
stop handling), not numerical noise.

Standard library only, so the check runs on a bare runner.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Iterable

SEED = 7

# Short, varied prompts: plain answers, arithmetic, translation, and a system
# message so the model's chat template is exercised end to end.
PROMPTS: list[dict] = [
    {"name": "colors", "messages": [{"role": "user", "content": "Name three primary colors."}]},
    {"name": "ocean", "messages": [{"role": "user", "content": "Write one sentence about the ocean."}]},
    {
        "name": "arithmetic",
        "messages": [{"role": "user", "content": "What is 12 times 7? Answer with the number only."}],
    },
    {
        "name": "translation",
        "messages": [{"role": "user", "content": "Translate 'good morning' into French."}],
    },
    {
        "name": "system-message",
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "List two prime numbers."},
        ],
    },
]


def request_body(model: str, messages: list[dict], max_tokens: int, stream: bool) -> dict:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": SEED,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def parse_sse(lines: Iterable[str]) -> tuple[str, str | None]:
    """Join the content deltas of an OpenAI chat-completion SSE stream."""
    text: list[str] = []
    finish_reason = None
    done = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            done = True
            break
        chunk = json.loads(payload)
        if "error" in chunk:
            raise ValueError(f"stream reported an error: {chunk['error']}")
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                text.append(content)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    if not done:
        raise ValueError("stream ended without a [DONE] event")
    return "".join(text), finish_reason


def check_run(results: dict) -> list[str]:
    """Problems within one server's results, independent of the other route."""
    problems = []
    for case in results["cases"]:
        name = case["name"]
        if not case["unary"]["content"].strip():
            problems.append(f"{name}: unary response has no text")
        if case["stream"]["content"] != case["unary"]["content"]:
            problems.append(
                f"{name}: streamed text {case['stream']['content']!r} "
                f"differs from unary text {case['unary']['content']!r}"
            )
        if case["stream"]["finish_reason"] != case["unary"]["finish_reason"]:
            problems.append(
                f"{name}: streamed finish_reason {case['stream']['finish_reason']!r} "
                f"differs from unary {case['unary']['finish_reason']!r}"
            )
    return problems


def compare_results(reference: dict, candidate: dict) -> list[str]:
    """Differences between the embedded reference and the pack candidate."""
    mismatches = []
    reference_cases = {case["name"]: case for case in reference["cases"]}
    candidate_cases = {case["name"]: case for case in candidate["cases"]}
    if sorted(reference_cases) != sorted(candidate_cases):
        return [f"prompt sets differ: {sorted(reference_cases)} vs {sorted(candidate_cases)}"]
    for name, expected in reference_cases.items():
        actual = candidate_cases[name]
        for mode in ("unary", "stream"):
            for field in ("content", "finish_reason"):
                if expected[mode][field] != actual[mode][field]:
                    mismatches.append(
                        f"{name} ({mode} {field}): embedded {expected[mode][field]!r}, "
                        f"pack {actual[mode][field]!r}"
                    )
        if expected["unary"]["completion_tokens"] != actual["unary"]["completion_tokens"]:
            mismatches.append(
                f"{name} (completion_tokens): embedded {expected['unary']['completion_tokens']}, "
                f"pack {actual['unary']['completion_tokens']}"
            )
    return mismatches


def http_json(url: str, body: dict | None, timeout: float) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{url} returned HTTP {error.code}: {error.read().decode(errors='replace')}")


def http_stream(url: str, body: dict, timeout: float) -> tuple[str, str | None]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return parse_sse(line.decode() for line in response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{url} returned HTTP {error.code}: {error.read().decode(errors='replace')}")


def run(arguments: argparse.Namespace) -> int:
    base = arguments.base_url.rstrip("/")
    models = http_json(f"{base}/v1/models", None, arguments.timeout_seconds)
    entries = models.get("data") if isinstance(models, dict) else models
    if not entries:
        raise SystemExit(f"{base}/v1/models listed no model")
    model = entries[0]["id"]
    url = f"{base}/v1/chat/completions"
    cases = []
    for prompt in PROMPTS:
        unary = http_json(
            url,
            request_body(model, prompt["messages"], arguments.max_tokens, stream=False),
            arguments.timeout_seconds,
        )
        choice = unary["choices"][0]
        stream_text, stream_finish = http_stream(
            url,
            request_body(model, prompt["messages"], arguments.max_tokens, stream=True),
            arguments.timeout_seconds,
        )
        cases.append(
            {
                "name": prompt["name"],
                "unary": {
                    "content": choice["message"].get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": (unary.get("usage") or {}).get("completion_tokens"),
                },
                "stream": {"content": stream_text, "finish_reason": stream_finish},
            }
        )
    results = {"label": arguments.label, "model": model, "seed": SEED,
               "max_tokens": arguments.max_tokens, "cases": cases}
    problems = check_run(results)
    results["problems"] = problems
    pathlib.Path(arguments.output).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    for case in cases:
        print(f"[{arguments.label}] {case['name']}: {case['unary']['content']!r}")
    if problems:
        print(f"FAIL: {arguments.label} results are inconsistent:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


def compare(arguments: argparse.Namespace) -> int:
    reference = json.loads(pathlib.Path(arguments.reference).read_text())
    candidate = json.loads(pathlib.Path(arguments.candidate).read_text())
    mismatches = compare_results(reference, candidate)
    report = {"reference": reference["label"], "candidate": candidate["label"],
              "cases": len(reference["cases"]), "mismatches": mismatches}
    pathlib.Path(arguments.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if mismatches:
        print("FAIL: the llama-cpp/cpu pack diverged from embedded llama.cpp:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  - {mismatch}", file=sys.stderr)
        return 1
    print(f"PASS: {len(reference['cases'])} greedy prompts match exactly (unary and streamed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--max-tokens", type=int, default=32)
    run_parser.add_argument("--timeout-seconds", type=float, default=300)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--reference", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    return run(arguments) if arguments.command == "run" else compare(arguments)


if __name__ == "__main__":
    sys.exit(main())
