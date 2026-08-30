#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "managed_vllm_scaling_conformance.py"
SPEC = importlib.util.spec_from_file_location("managed_vllm_scaling_conformance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def metric(name: str, value: int, replica: int = 0, device: int = 0) -> str:
    return f'{name}{{model="model",replica="{replica}",device="{device}"}} {value}'


class ManagedVllmScalingConformanceTests(unittest.TestCase):
    def metrics(self, *, replicas: int = 1, quarantine: int = 0) -> str:
        lines = []
        values = {
            "requested_bytes": 800,
            "granted_bytes": 600,
            "minimum_bytes": 200,
            "backing_bytes": 600,
            "total_blocks": 8,
            "allocated_blocks": 6,
            "active_blocks": 2,
            "idle_blocks": 4,
            "quarantine_bytes": quarantine,
        }
        for replica in range(replicas):
            for field, metric_name in probe.MEMORY_METRICS.items():
                lines.append(metric(metric_name, values[field], replica=replica))
        return "\n".join(lines) + "\n"

    def test_active_rows_preserve_replica_device_identity(self):
        rows = probe.active_kv_rows(self.metrics(replicas=2), "model")
        probe.validate_exact_rows(rows, 2)
        self.assertEqual([(row.replica, row.device) for row in rows], [(0, 0), (1, 0)])

    def test_released_zero_backing_row_is_ignored(self):
        text = self.metrics()
        released = self.metrics().replace('replica="0"', 'replica="1"')
        released = released.replace(
            f"{probe.MEMORY_METRICS['backing_bytes']}{{model=\"model\",replica=\"1\",device=\"0\"}} 600",
            f"{probe.MEMORY_METRICS['backing_bytes']}{{model=\"model\",replica=\"1\",device=\"0\"}} 0",
        )
        rows = probe.active_kv_rows(text + released, "model")
        self.assertEqual([row.replica for row in rows], [0])

    def test_quarantine_row_is_never_hidden(self):
        rows = probe.active_kv_rows(
            self.metrics(quarantine=128).replace(
                f"{probe.MEMORY_METRICS['backing_bytes']}{{model=\"model\",replica=\"0\",device=\"0\"}} 600",
                f"{probe.MEMORY_METRICS['backing_bytes']}{{model=\"model\",replica=\"0\",device=\"0\"}} 0",
            ),
            "model",
        )
        with self.assertRaisesRegex(probe.ScalingConformanceError, "quarantined"):
            probe.validate_exact_rows(rows, 1)

    def test_metric_identity_labels_must_be_canonical_integers(self):
        with self.assertRaisesRegex(probe.ScalingConformanceError, "canonical"):
            probe.active_kv_rows(
                self.metrics().replace('replica="0"', 'replica="00"'), "model"
            )

    def test_registry_filter_uses_base_and_replica_ids(self):
        models = [
            {"id": 7, "name": "model", "base_model_id": 7, "replica_id": 0, "status": "active"},
            {"id": 8, "name": "model", "base_model_id": 7, "replica_id": 1, "status": "active"},
            {"id": 9, "name": "other", "base_model_id": 9, "replica_id": 0, "status": "active"},
        ]
        self.assertEqual(probe.active_replicas(models, "model"), (7, [0, 1]))


if __name__ == "__main__":
    unittest.main()
