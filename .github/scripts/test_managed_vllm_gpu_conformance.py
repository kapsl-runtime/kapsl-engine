#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("managed_vllm_gpu_conformance.py")
SPEC = importlib.util.spec_from_file_location("managed_vllm_gpu_conformance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class ManagedVllmGpuConformanceTests(unittest.TestCase):
    def snapshot(self, **overrides):
        values = {
            "row_keys": ["replica=0,device=0"],
            "requested_bytes": [800],
            "granted_bytes": [600],
            "minimum_bytes": [200],
            "backing_bytes": [600],
            "total_blocks": [8],
            "allocated_blocks": [6],
            "active_blocks": [2],
            "idle_blocks": [4],
            "quarantine_bytes": [0],
            "restart_generation": 1,
        }
        values.update(overrides)
        return probe.MemorySnapshot(**values)

    def test_prometheus_parser_preserves_labels_and_values(self):
        parsed = probe.parse_prometheus(
            '# HELP ignored x\nmetric_name{mode="wire",device="0"} 12\n'
        )
        self.assertEqual(parsed["metric_name"][0].labels, {"mode": "wire", "device": "0"})
        self.assertEqual(parsed["metric_name"][0].value, 12)

    def test_vllm_prefix_cache_counters_use_prometheus_counter_names(self):
        exposition = "\n".join(
            [
                '# TYPE vllm:prefix_cache_queries_total counter',
                'vllm:prefix_cache_queries_total{model_name="model",engine="0"} 48.0',
                '# TYPE vllm:prefix_cache_hits_total counter',
                'vllm:prefix_cache_hits_total{model_name="model",engine="0"} 32.0',
            ]
        )
        with mock.patch.object(probe, "_get_text", return_value=exposition):
            self.assertEqual(
                probe._vllm_counter(
                    "http://127.0.0.1:8000", probe.VLLM_PREFIX_CACHE_QUERY_METRIC
                ),
                48,
            )
            self.assertEqual(
                probe._vllm_counter(
                    "http://127.0.0.1:8000", probe.VLLM_PREFIX_CACHE_HIT_METRIC
                ),
                32,
            )

    def test_exact_initial_and_resized_accounting_are_distinct(self):
        snapshot = self.snapshot()
        self.assertEqual(
            probe.validate_memory_snapshot(
                snapshot, elastic=True, exact_initial_grant=True
            ),
            100,
        )
        grown = self.snapshot(backing_bytes=[700], allocated_blocks=[7], idle_blocks=[5])
        self.assertEqual(
            probe.validate_memory_snapshot(grown, elastic=True), 100
        )
        with self.assertRaisesRegex(probe.ConformanceError, "exact grant"):
            probe.validate_memory_snapshot(
                grown, elastic=True, exact_initial_grant=True
            )

    def test_memory_collection_keeps_every_metric_bound_to_its_device_row(self):
        lines = []
        values = {
            "requested_bytes": {("0", "0"): 800, ("0", "1"): 900},
            "granted_bytes": {("0", "0"): 600, ("0", "1"): 700},
            "minimum_bytes": {("0", "0"): 200, ("0", "1"): 300},
            "backing_bytes": {("0", "0"): 600, ("0", "1"): 700},
            "total_blocks": {("0", "0"): 8, ("0", "1"): 9},
            "allocated_blocks": {("0", "0"): 6, ("0", "1"): 7},
            "active_blocks": {("0", "0"): 2, ("0", "1"): 3},
            "idle_blocks": {("0", "0"): 4, ("0", "1"): 4},
            "quarantine_bytes": {("0", "0"): 0, ("0", "1"): 0},
        }
        for field, metric in probe.MEMORY_METRICS.items():
            for replica, device in reversed(tuple(values[field])):
                lines.append(
                    f'{metric}{{model="target",replica="{replica}",device="{device}"}} '
                    f'{values[field][(replica, device)]}'
                )
            lines.append(
                f'{metric}{{model="other",replica="0",device="0"}} 999999'
            )
        lines.extend(
            [
                f'{probe.GENERATION_METRIC}{{model="target",replica="0"}} 3',
                f'{probe.GENERATION_METRIC}{{model="other",replica="0"}} 99',
            ]
        )
        with mock.patch.object(probe, "_get_text", return_value="\n".join(lines)):
            snapshot = probe.collect_memory_snapshot("http://metrics", "target")

        self.assertEqual(
            snapshot.row_keys,
            ["replica=0,device=0", "replica=0,device=1"],
        )
        self.assertEqual(snapshot.granted_bytes, [600, 700])
        self.assertEqual(snapshot.minimum_bytes, [200, 300])
        self.assertEqual(snapshot.backing_bytes, [600, 700])
        self.assertEqual(snapshot.restart_generation, 3)

    def test_quarantine_and_missing_virtual_headroom_fail_closed(self):
        with self.assertRaisesRegex(probe.ConformanceError, "quarantined"):
            probe.validate_memory_snapshot(
                self.snapshot(quarantine_bytes=[1]), elastic=True
            )
        with self.assertRaisesRegex(probe.ConformanceError, "virtual"):
            probe.validate_memory_snapshot(
                self.snapshot(total_blocks=[6]), elastic=True
            )

    def test_child_log_requires_one_address_to_span_grow_and_shrink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_dir = root / "state" / "managed-vllm" / "model-0"
            child_dir.mkdir(parents=True)
            runtime_log = root / "runtime.log"
            child_log = child_dir / "vllm.log"
            runtime_log.write_text(
                f"Managed vLLM process started: pid=42 endpoint=x log={child_log}\n",
                encoding="utf-8",
            )
            child_log.write_text(
                "\n".join(
                    [
                        "KAPSL_VMM_CONFORMANCE stable_address=0x1000 mapped_bytes=200 virtual_bytes=800 phase=initial zeroed=true",
                        "KAPSL_VMM_CONFORMANCE allocator_delta_bytes=0 virtual_bytes=800",
                        "capped vLLM startup warmup to 2 mapped blocks out of 8 virtual blocks",
                        "KAPSL_VMM_CONFORMANCE stable_address=0x1000 mapped_bytes=600 virtual_bytes=800 phase=grow zeroed=true",
                        "KAPSL_VMM_CONFORMANCE stable_address=0x1000 mapped_bytes=200 virtual_bytes=800 phase=shrink",
                        # A new generation may have only initial evidence by the
                        # time the report is finalized.
                        "KAPSL_VMM_CONFORMANCE stable_address=0x2000 mapped_bytes=200 virtual_bytes=800 phase=initial zeroed=true",
                        "KAPSL_VMM_CONFORMANCE allocator_delta_bytes=0 virtual_bytes=800",
                        "capped vLLM startup warmup to 2 mapped blocks out of 8 virtual blocks",
                    ]
                ),
                encoding="utf-8",
            )
            evidence = probe._validate_vmm_logs(runtime_log, root / "state")
            self.assertEqual(evidence["resize_stable_addresses"], ["0x1000"])
            self.assertEqual(evidence["child_logs"], [str(child_log.resolve())])
            self.assertEqual(evidence["startup_warmup_caps"], [(2, 8), (2, 8)])
            with self.assertRaisesRegex(probe.ConformanceError, "churn"):
                probe._validate_vmm_logs(
                    runtime_log, root / "state", minimum_resize_cycles=2
                )

            child_log.write_text(
                "\n".join(
                    line
                    for line in child_log.read_text(encoding="utf-8").splitlines()
                    if "startup warmup" not in line
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(probe.ConformanceError, "startup warmup"):
                probe._validate_vmm_logs(runtime_log, root / "state")

    def test_runtime_cannot_redirect_log_evidence_outside_state_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            outside = root / "outside.log"
            outside.write_text("", encoding="utf-8")
            runtime_log = root / "runtime.log"
            runtime_log.write_text(
                f"Managed vLLM process started: pid=42 endpoint=x log={outside}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(probe.ConformanceError, "escaped"):
                probe._managed_vllm_log_paths(runtime_log, state)

    def test_private_endpoint_and_cross_token_stop_candidate_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_log = root / "runtime.log"
            runtime_log.write_text(
                "Managed vLLM process started: pid=42 "
                "endpoint=http://127.0.0.1:8123 log=/tmp/vllm.log\n",
                encoding="utf-8",
            )
            self.assertEqual(
                probe._managed_vllm_endpoint(runtime_log),
                "http://127.0.0.1:8123",
            )
            runtime_log.write_text(
                "Managed vLLM process started: pid=42 "
                "endpoint=http://192.0.2.1:8123 log=/tmp/vllm.log\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(probe.ConformanceError, "loopback"):
                probe._managed_vllm_endpoint(runtime_log)

            state = root / "state" / "managed-vllm" / "model-0"
            state.mkdir(parents=True)
            child_log = state / "vllm.log"
            runtime_log.write_text("normal runtime output\n", encoding="utf-8")
            child_log.write_text(
                "(APIServer pid=42) INFO [entry.py:139] "
                "Starting vLLM server on http://127.0.0.1:9123\n",
                encoding="utf-8",
            )
            self.assertEqual(
                probe._managed_vllm_endpoint(runtime_log, root / "state"),
                "http://127.0.0.1:9123",
            )
            child_log.write_text(
                "Starting vLLM server on http://192.0.2.1:9123\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(probe.ConformanceError, "loopback"):
                probe._managed_vllm_endpoint(runtime_log, root / "state")

        class FakeTokenizer:
            def __call__(self, *_args, **_kwargs):
                return {"offset_mapping": [(0, 3), (3, 6)]}

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
            )
        )
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            candidate, start, boundary = probe._cross_token_stop_string(
                "abcDEF", Path("/model")
            )
        self.assertEqual((candidate, start, boundary), ("cD", 2, 3))

    def test_full_context_request_accounts_exact_prompt_and_generation_tokens(self):
        template_calls = []

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                template_calls.append(kwargs)
                if kwargs.get("return_dict") is not False:
                    return {
                        "input_ids": [1, 2, 3],
                        "attention_mask": [1, 1, 1],
                    }
                repetitions = messages[0]["content"].count("capacity ")
                return list(range(10 + repetitions))

        def completion(_url, _model, _max_tokens=32, *, body=None):
            assert body is not None
            prompt_tokens = 10 + body["messages"][0]["content"].count("capacity ")
            completion_tokens = body["max_tokens"]
            return {
                "id": "chatcmpl-full",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

        with (
            mock.patch.object(probe, "_load_tokenizer", return_value=FakeTokenizer()),
            mock.patch.object(probe, "_require_completion", side_effect=completion),
        ):
            evidence = probe._require_full_context_request(
                "http://localhost/v1/chat/completions",
                "model",
                Path("/model"),
                128,
            )
        self.assertEqual(evidence["prompt_tokens"], 96)
        self.assertEqual(evidence["completion_tokens"], 32)
        self.assertEqual(evidence["total_tokens"], 128)
        self.assertTrue(template_calls)
        self.assertTrue(all(call["return_dict"] is False for call in template_calls))


if __name__ == "__main__":
    unittest.main()
