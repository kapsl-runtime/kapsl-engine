#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("github_jit_runner.py")
SPEC = importlib.util.spec_from_file_location("github_jit_runner", MODULE_PATH)
assert SPEC and SPEC.loader
jit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = jit
SPEC.loader.exec_module(jit)


class JitRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(os.environ, {jit.TOKEN_ENV: "app-token"})
        self.environment.start()
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()
        self.environment.stop()

    @mock.patch.object(jit, "_request")
    def test_create_writes_secret_without_printing_it(self, request: mock.Mock) -> None:
        request.return_value = (
            201,
            {
                "runner": {
                    "id": 42,
                    "name": "kapsl-vllm-123-1",
                    "labels": [
                        {"name": "self-hosted"},
                        {"name": "linux"},
                        {"name": "x64"},
                        {"name": "kapsl-vllm-123-1"},
                    ],
                },
                "encoded_jit_config": "encoded-secret",
            },
        )
        path = Path(self.temporary.name) / "jit-config"
        result = jit.create_runner(
            "kapsl-runtime",
            "kapsl-engine",
            runner_group_id="1",
            name="kapsl-vllm-123-1",
            labels=["self-hosted", "linux", "x64", "kapsl-vllm-123-1"],
            config_file=path,
        )
        self.assertEqual(result, {"runner_id": 42, "runner_name": "kapsl-vllm-123-1"})
        self.assertEqual(path.read_text(encoding="utf-8"), "encoded-secret")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        payload = request.call_args.kwargs["payload"]
        self.assertEqual(payload["runner_group_id"], 1)
        self.assertEqual(payload["work_folder"], "_work")
        self.assertNotIn("encoded-secret", json.dumps(result))

    def test_create_rejects_duplicate_or_injected_labels_before_api(self) -> None:
        path = Path(self.temporary.name) / "jit-config"
        for labels in (["gpu", "GPU"], ["gpu", "bad label"], ["gpu", "x;rm"]):
            with self.subTest(labels=labels), self.assertRaises(jit.JitRunnerError):
                jit.create_runner(
                    "kapsl-runtime",
                    "kapsl-engine",
                    runner_group_id=1,
                    name="kapsl-vllm-123-1",
                    labels=labels,
                    config_file=path,
                )

    @mock.patch.object(jit.time, "sleep")
    @mock.patch.object(jit, "get_runner")
    def test_wait_observes_same_runner_until_online(
        self,
        get_runner: mock.Mock,
        sleep: mock.Mock,
    ) -> None:
        get_runner.side_effect = [
            {"id": 42, "name": "kapsl-vllm-123-1", "status": "offline"},
            {
                "id": 42,
                "name": "kapsl-vllm-123-1",
                "status": "online",
                "busy": False,
            },
        ]
        result = jit.wait_for_runner(
            "kapsl-runtime",
            "kapsl-engine",
            42,
            expected_name="kapsl-vllm-123-1",
            timeout_seconds=30,
            interval_seconds=1,
        )
        self.assertEqual(result["status"], "online")
        sleep.assert_called_once_with(1)

    @mock.patch.object(jit, "get_runner")
    def test_wait_rejects_runner_claimed_by_another_job(
        self, get_runner: mock.Mock
    ) -> None:
        get_runner.return_value = {
            "id": 42,
            "name": "kapsl-vllm-123-1",
            "status": "online",
            "busy": True,
        }
        with self.assertRaises(jit.JitRunnerError):
            jit.wait_for_runner(
                "kapsl-runtime",
                "kapsl-engine",
                42,
                expected_name="kapsl-vllm-123-1",
                timeout_seconds=30,
                interval_seconds=1,
            )

    @mock.patch.object(jit, "_request", return_value=(404, None))
    def test_delete_is_idempotent(self, request: mock.Mock) -> None:
        self.assertFalse(jit.delete_runner("kapsl-runtime", "kapsl-engine", 42))
        request.assert_called_once_with(
            "DELETE",
            "/repos/kapsl-runtime/kapsl-engine/actions/runners/42",
            allow_not_found=True,
        )

    def test_missing_token_fails_closed(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(jit.JitRunnerError):
            jit._token()


if __name__ == "__main__":
    unittest.main()
