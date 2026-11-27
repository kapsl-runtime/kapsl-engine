#!/usr/bin/env python3
"""Host-only streaming/cancellation regressions for the GPU evidence probe."""

import contextlib
import http.server
import importlib.util
import pathlib
import tempfile
import threading
import unittest

SCRIPT = pathlib.Path(__file__).with_name("probe-live-kv-owner.py")
SPEC = importlib.util.spec_from_file_location("probe_live_kv_owner", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
OWNER = "gguf:1:0:kv-cache"


@contextlib.contextmanager
def endpoint(*, wrong_owner=False, leak=False, done=False):
    active = threading.Event()
    closed = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.assert_authorized()
            self.send_response(200)
            self.end_headers()
            owner = "gguf:2:0:kv-cache" if wrong_owner else OWNER
            body = ""
            if active.is_set():
                body = f'kapsl_gpu_device_pool_owner_usage_bytes{{device="0",owner="{owner}"}} 64\n'
            self.wfile.write(body.encode())

        def assert_authorized(self):
            assert self.headers["Authorization"] == "Bearer test-token"

        def do_POST(self):
            self.assert_authorized()
            assert self.path == "/api/models/1/infer/stream"
            self.rfile.read(int(self.headers["Content-Length"]))
            active.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b"data: [DONE]\n\n" if done else b'data: {"token":"one"}\n\n'
            )
            self.wfile.flush()
            self.connection.settimeout(5)
            try:
                self.connection.recv(1)
            finally:
                if not leak:
                    active.clear()
                closed.set()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", closed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class LiveKvOwnerTests(unittest.TestCase):
    def run_probe(self, base, root):
        PROBE.probe(
            base,
            "test-token",
            1,
            OWNER,
            pathlib.Path(root) / "evidence",
            poll_seconds=0.001,
        )

    def test_observes_live_owner_then_disconnect_reclaims_it(self):
        with tempfile.TemporaryDirectory() as root, endpoint() as (base, closed):
            self.run_probe(base, root)
            self.assertTrue(closed.wait(1))
            metrics = (pathlib.Path(root) / "evidence/metrics-active.txt").read_text()
            self.assertEqual(PROBE.owner_usage(metrics, OWNER), 64)

    def test_other_model_allocations_do_not_qualify(self):
        with (
            tempfile.TemporaryDirectory() as root,
            endpoint(wrong_owner=True) as (base, closed),
        ):
            with self.assertRaisesRegex(AssertionError, "no live allocation"):
                self.run_probe(base, root)
            self.assertTrue(closed.wait(1))

    def test_retained_allocations_after_cancellation_fail(self):
        with (
            tempfile.TemporaryDirectory() as root,
            endpoint(leak=True) as (base, _closed),
        ):
            with self.assertRaisesRegex(AssertionError, "retained request KV"):
                self.run_probe(base, root)

    def test_completed_stream_is_not_live_ownership_proof(self):
        with (
            tempfile.TemporaryDirectory() as root,
            endpoint(done=True) as (base, closed),
        ):
            with self.assertRaisesRegex(AssertionError, "generation ended"):
                self.run_probe(base, root)
            self.assertTrue(closed.wait(1))


if __name__ == "__main__":
    unittest.main()
