#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "certify-mixed-backend-concurrency.py"
SPEC = importlib.util.spec_from_file_location("certify_mixed_backend_concurrency", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class MixedBackendMemoryTests(unittest.TestCase):
    def fixture(self, available: int = 1024, quarantine: int = 0) -> str:
        return "\n".join(
            [
                'kapsl_managed_vllm_kv_backing_bytes{model="vllm",replica="0",device="0"} 600',
                f'kapsl_managed_vllm_kv_quarantine_bytes{{model="vllm",replica="0",device="0"}} {quarantine}',
                'kapsl_gpu_device_pool_owner_admitted{device="0",owner="gguf:7:0:persistent-weights"} 1',
                'kapsl_gpu_device_pool_owner_admitted{device="0",owner="gguf:7:0:kv-cache"} 1',
                'kapsl_gpu_device_pool_owner_usage_bytes{device="0",owner="gguf:7:0:kv-cache"} 256',
                f'kapsl_device_memory_available_bytes{{device="0"}} {available}',
                'kapsl_gpu_device_pool_allocated_bytes{device="0"} 512',
                'kapsl_gpu_device_pool_free_bytes{device="0"} 512',
                'kapsl_gpu_device_pool_free_ranges{device="0"} 1',
                'kapsl_gpu_device_pool_largest_free_range_bytes{device="0"} 512',
                'kapsl_gpu_device_pool_fragmentation_ratio{device="0"} 0',
            ]
        )

    def test_mixed_memory_requires_both_owners_and_ungranted_headroom(self):
        result = probe.validate_mixed_memory(
            self.fixture(), vllm_model="vllm", llama_model_id=7
        )
        self.assertEqual(
            result["llama_admission_owner"], "gguf:7:0:persistent-weights"
        )
        self.assertEqual(result["llama_kv_owner"], "gguf:7:0:kv-cache")
        self.assertEqual(result["authority_available_bytes"], {"0": 1024})

    def test_unrelated_or_legacy_gguf_owner_cannot_satisfy_gate(self):
        metrics = self.fixture().replace(
            'owner="gguf:7:0:persistent-weights"', 'owner="gguf:8:0:persistent-weights"'
        ).replace('owner="gguf:7:0:kv-cache"', 'owner="gguf_kv:7"')
        with self.assertRaisesRegex(ValueError, "has no matching samples"):
            probe.validate_mixed_memory(
                metrics, vllm_model="vllm", llama_model_id=7
            )

    def test_llama_and_vllm_must_share_a_device(self):
        metrics = self.fixture().replace(
            'kapsl_managed_vllm_kv_backing_bytes{model="vllm",replica="0",device="0"}',
            'kapsl_managed_vllm_kv_backing_bytes{model="vllm",replica="0",device="1"}',
        ).replace(
            'kapsl_managed_vllm_kv_quarantine_bytes{model="vllm",replica="0",device="0"}',
            'kapsl_managed_vllm_kv_quarantine_bytes{model="vllm",replica="0",device="1"}',
        ).replace(
            'kapsl_device_memory_available_bytes{device="0"}',
            'kapsl_device_memory_available_bytes{device="1"}',
        )
        with self.assertRaisesRegex(ValueError, "same devices"):
            probe.validate_mixed_memory(
                metrics, vllm_model="vllm", llama_model_id=7
            )

    def test_pool_metrics_are_joined_by_device_instead_of_exposition_order(self):
        metrics = self.fixture() + "\n" + "\n".join(
            [
                'kapsl_managed_vllm_kv_backing_bytes{model="vllm",replica="0",device="1"} 700',
                'kapsl_managed_vllm_kv_quarantine_bytes{model="vllm",replica="0",device="1"} 0',
                'kapsl_device_memory_available_bytes{device="1"} 2048',
                'kapsl_gpu_device_pool_allocated_bytes{device="1"} 0',
                'kapsl_gpu_device_pool_free_bytes{device="1"} 1',
                'kapsl_gpu_device_pool_free_ranges{device="1"} 1',
                'kapsl_gpu_device_pool_largest_free_range_bytes{device="1"} 1',
                'kapsl_gpu_device_pool_fragmentation_ratio{device="1"} 0',
            ]
        )
        result = probe.validate_mixed_memory(
            metrics, vllm_model="vllm", llama_model_id=7
        )
        self.assertEqual(result["vllm_backing_bytes"], {"0": 600, "1": 700})
        self.assertEqual(result["general_pool_allocated_bytes"]["0"], 512)

    def test_zero_available_or_quarantine_fails(self):
        with self.assertRaisesRegex(ValueError, "ungranted"):
            probe.validate_mixed_memory(
                self.fixture(available=0), vllm_model="vllm", llama_model_id=7
            )
        with self.assertRaisesRegex(ValueError, "quarantined"):
            probe.validate_mixed_memory(
                self.fixture(quarantine=1), vllm_model="vllm", llama_model_id=7
            )


if __name__ == "__main__":
    unittest.main()
