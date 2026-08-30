#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("managed_vllm_bridge_benchmark.py")
SPEC = importlib.util.spec_from_file_location("managed_vllm_bridge_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def report(target: str, throughput: float, median_ttft: float, p95_ttft: float):
    return {
        "schema_version": 1,
        "target": target,
        "model": "model",
        "request_body_sha256": "sha256:body",
        "trials": 3,
        "engine_settings": {
            "vllm_version": "pinned",
            "vllm_build_id": "sha256:vllm",
            "model_revision": "commit",
            "kv_cache_memory_bytes_per_rank": 1024,
            "tensor_parallel_size": 1,
            "max_model_len": 1024,
            "attention_backend": "FLASH_ATTN",
            "enforce_eager": True,
        },
        "points": [
            {
                "concurrency": concurrency,
                "trials": [
                    {
                        "output_tokens_per_second": throughput,
                        "median_ttft_seconds": median_ttft,
                        "p95_ttft_seconds": p95_ttft,
                    }
                    for _ in range(3)
                ],
            }
            for concurrency in (1, 4, 8, 16)
        ],
    }


class ManagedVllmBridgeBenchmarkTests(unittest.TestCase):
    def test_acceptance_run_defaults_to_fifteen_independent_trials(self):
        parser = benchmark._parser()
        args = parser.parse_args(
            [
                "run",
                "--base-url",
                "http://127.0.0.1:8000",
                "--model",
                "model",
                "--target",
                "kapsl",
                "--output",
                "/tmp/report.json",
                "--vllm-version",
                "pinned",
                "--vllm-build-id",
                "sha256:vllm",
                "--model-revision",
                "commit",
                "--kv-cache-memory-bytes",
                "1024",
                "--tensor-parallel-size",
                "1",
            ]
        )
        self.assertEqual(args.trials, 15)
        args.trials = 14
        with self.assertRaisesRegex(benchmark.BenchmarkError, "at least 15"):
            benchmark.run_target(args)

    def test_percentile_interpolates(self):
        self.assertEqual(benchmark.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(benchmark.percentile([1.0, 3.0], 0.25), 1.5)

    def test_acceptance_thresholds_pass_at_or_below_limits(self):
        result = benchmark.compare_reports(
            report("direct", 100.0, 0.010, 0.020),
            report("kapsl", 98.0, 0.015, 0.030),
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["failures"], [])

    def test_throughput_regression_fails_every_concurrency(self):
        result = benchmark.compare_reports(
            report("direct", 100.0, 0.010, 0.020),
            report("kapsl", 95.0, 0.011, 0.021),
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            [failure["concurrency"] for failure in result["failures"]],
            [1, 4, 8, 16],
        )

    def test_request_body_is_identical_for_both_targets(self):
        first = benchmark._request_body("model", 128)
        second = benchmark._request_body("model", 128)
        self.assertEqual(first, second)
        self.assertTrue(first["ignore_eos"])
        self.assertEqual(first["max_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
