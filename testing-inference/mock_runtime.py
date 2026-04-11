#!/usr/bin/env python3
"""
Minimal mock of the kapsl-runtime HTTP API for benchmark validation.
Simulates /api/health, /api/models, /api/models/:id/infer, /api/system/stats.

Usage:
  python3 mock_runtime.py                    # listens on 127.0.0.1:9095
  python3 mock_runtime.py --port 9095 --latency-ms 50 --mode llm
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import struct
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MOCK_RESPONSE_TEXT = (
    "The transformer architecture uses self-attention mechanisms to weigh "
    "the importance of different tokens in a sequence when producing each output token. "
    "This allows the model to capture long-range dependencies more effectively than RNNs."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock kapsl-runtime for benchmark testing")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9095)
    p.add_argument("--latency-ms", type=float, default=40.0,
                   help="Simulated inference latency in ms (default: 40)")
    p.add_argument("--jitter-ms", type=float, default=10.0,
                   help="Random jitter added to latency (default: 10)")
    p.add_argument("--mode", choices=["llm", "tensor"], default="llm",
                   help="Response format to return (default: llm)")
    return p.parse_args()


ARGS = parse_args()


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress per-request logs
        pass

    def send_json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/health":
            self.send_json(200, {"status": "ok"})

        elif path == "/api/models":
            self.send_json(200, [
                {"id": 0, "name": "mock-model-0", "status": "running",
                 "queue_depth": [0, 0], "active_inferences": 0}
            ])

        elif path.startswith("/api/models/") and not path.endswith("/infer"):
            self.send_json(200, {
                "id": 0, "name": "mock-model-0", "status": "running",
                "queue_depth": [0, 0], "active_inferences": 0,
            })

        elif path == "/api/system/stats":
            self.send_json(200, {"queue_depth": 0, "active_requests": 0, "throughput_rps": 0})

        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        if path.endswith("/infer"):
            # Read body (we don't validate it)
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)

            # Simulate latency
            delay = (ARGS.latency_ms + random.uniform(0, ARGS.jitter_ms)) / 1000.0
            time.sleep(delay)

            if ARGS.mode == "llm":
                payload = {
                    "dtype": "string",
                    "shape": [1, 1],
                    "data": list(MOCK_RESPONSE_TEXT.encode("utf-8")),
                }
            else:
                # Return a float32 tensor [1, 10] as a classification output
                values = [random.random() for _ in range(10)]
                raw = struct.pack(f"@10f", *values)
                payload = {
                    "dtype": "float32",
                    "shape": [1, 10],
                    "data": list(raw),
                }

            self.send_json(200, payload)
        else:
            self.send_json(404, {"error": "not found"})


def main() -> None:
    server = HTTPServer((ARGS.host, ARGS.port), MockHandler)
    print(f"Mock kapsl-runtime listening on {ARGS.host}:{ARGS.port}  "
          f"mode={ARGS.mode}  latency={ARGS.latency_ms}ms ± {ARGS.jitter_ms}ms")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
