#!/usr/bin/env python3
"""Tests for the llama.cpp CPU parity helper (no model or Kapsl build needed)."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import threading
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import TestCase, main

spec = importlib.util.spec_from_file_location(
    "llama_cpp_cpu_parity", Path(__file__).with_name("llama_cpp_cpu_parity.py")
)
assert spec and spec.loader
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def sse(*chunks: dict, done: bool = True) -> list[str]:
    lines = [": keep-alive"]
    lines += [f"data: {json.dumps(chunk)}" for chunk in chunks]
    if done:
        lines.append("data: [DONE]")
    return lines


def delta(content: str | None = None, finish_reason: str | None = None) -> dict:
    payload = {} if content is None else {"content": content}
    return {"choices": [{"index": 0, "delta": payload, "finish_reason": finish_reason}]}


def case(name: str, text: str, stream_text: str | None = None, finish: str = "stop",
         tokens: int = 4) -> dict:
    return {
        "name": name,
        "unary": {"content": text, "finish_reason": finish, "completion_tokens": tokens},
        "stream": {"content": text if stream_text is None else stream_text, "finish_reason": finish},
    }


class SseTests(TestCase):
    def test_deltas_are_joined_and_the_last_finish_reason_wins(self):
        text, finish = parity.parse_sse(
            sse(delta(None), delta("Red, "), delta("blue"), delta(None, "length"))
        )
        self.assertEqual(text, "Red, blue")
        self.assertEqual(finish, "length")

    def test_a_stream_without_done_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[DONE\]"):
            parity.parse_sse(sse(delta("partial"), done=False))

    def test_an_error_chunk_is_reported(self):
        with self.assertRaisesRegex(ValueError, "boom"):
            parity.parse_sse(sse({"error": {"message": "boom"}}))


class ComparisonTests(TestCase):
    def test_identical_runs_match(self):
        results = {"label": "x", "cases": [case("a", "Paris"), case("b", "42")]}
        self.assertEqual(parity.check_run(results), [])
        self.assertEqual(parity.compare_results(results, json.loads(json.dumps(results))), [])

    def test_a_run_whose_stream_differs_from_unary_is_inconsistent(self):
        problems = parity.check_run({"cases": [case("a", "Paris", stream_text="Pari")]})
        self.assertEqual(len(problems), 1)
        self.assertIn("streamed text", problems[0])

    def test_an_empty_answer_is_inconsistent(self):
        self.assertIn("no text", parity.check_run({"cases": [case("a", "  ")]})[0])

    def test_text_finish_reason_and_token_count_differences_are_reported(self):
        reference = {"cases": [case("a", "Paris", tokens=3)]}
        candidate = {"cases": [case("a", "Lyon", finish="length", tokens=5)]}
        mismatches = parity.compare_results(reference, candidate)
        self.assertTrue(any("unary content" in item for item in mismatches))
        self.assertTrue(any("stream finish_reason" in item for item in mismatches))
        self.assertTrue(any("completion_tokens" in item for item in mismatches))

    def test_different_prompt_sets_are_reported(self):
        mismatches = parity.compare_results({"cases": [case("a", "x")]}, {"cases": [case("b", "x")]})
        self.assertEqual(len(mismatches), 1)
        self.assertIn("prompt sets differ", mismatches[0])


class FakeOpenAiServer:
    """Answers each prompt deterministically, unary and as SSE."""

    def __init__(self, answer):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.reply(200, {"object": "list", "data": [{"id": "parity-model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                text = answer(body["messages"][-1]["content"])
                if body["stream"]:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for line in sse(delta(text[:3]), delta(text[3:]), delta(None, "stop")):
                        self.wfile.write((line + "\n\n").encode())
                else:
                    self.reply(200, {
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "finish_reason": "stop"}],
                        "usage": {"completion_tokens": len(text.split())},
                    })

            def reply(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()


class EndToEndTests(TestCase):
    def run_against(self, answer, directory: Path, label: str) -> tuple[int, Path, list[dict]]:
        output = directory / f"{label}.json"
        fake = FakeOpenAiServer(answer)
        with fake as base_url, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = parity.main(["run", "--base-url", base_url, "--label", label,
                                "--output", str(output), "--max-tokens", "16"])
        return code, output, fake.requests

    def test_requests_are_greedy_and_seeded_and_matching_servers_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, reference, requests = self.run_against(lambda prompt: f"echo {prompt}", root, "embedded")
            self.assertEqual(code, 0)
            self.assertEqual(len(requests), 2 * len(parity.PROMPTS))
            for body in requests:
                self.assertEqual(body["model"], "parity-model")
                self.assertEqual(body["temperature"], 0)
                self.assertEqual(body["top_p"], 1)
                self.assertEqual(body["seed"], parity.SEED)
                self.assertEqual(body["max_tokens"], 16)
            code, candidate, _ = self.run_against(lambda prompt: f"echo {prompt}", root, "pack")
            self.assertEqual(code, 0)
            report = root / "comparison.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(parity.main(["compare", "--reference", str(reference),
                                              "--candidate", str(candidate), "--output", str(report)]), 0)
            self.assertEqual(json.loads(report.read_text())["mismatches"], [])

    def test_a_diverging_pack_fails_the_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference, _ = self.run_against(lambda prompt: f"echo {prompt}", root, "embedded")
            _, candidate, _ = self.run_against(
                lambda prompt: "different" if "ocean" in prompt else f"echo {prompt}", root, "pack"
            )
            report = root / "comparison.json"
            with redirect_stderr(io.StringIO()) as errors:
                code = parity.main(["compare", "--reference", str(reference),
                                    "--candidate", str(candidate), "--output", str(report)])
            self.assertEqual(code, 1)
            self.assertIn("ocean (unary content)", errors.getvalue())
            self.assertEqual(len(json.loads(report.read_text())["mismatches"]), 3)


if __name__ == "__main__":
    main()
