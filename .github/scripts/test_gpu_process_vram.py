#!/usr/bin/env python3
"""Host-only regressions for container GPU memory attribution."""

import importlib.util
import pathlib
import unittest

SPEC = importlib.util.spec_from_file_location(
    "gpu_process_vram", pathlib.Path(__file__).with_name("gpu-process-vram.py")
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class GpuProcessVramTests(unittest.TestCase):
    def test_native_runtime_pid(self):
        identity, memory = PROBE.measure("", "GPU-a, 42, 123\n", 42)
        self.assertEqual(identity["method"], "runtime-pid")
        self.assertEqual(memory, 123 * 1024 * 1024)

    def test_host_pid_requires_explicit_exclusive_mode(self):
        with self.assertRaisesRegex(AssertionError, "absent"):
            PROBE.measure("", "GPU-a, 99, 123\n", 42)
        identity, _ = PROBE.measure("", "GPU-a, 99, 123\n", 42, exclusive=True)
        self.assertEqual(identity["runtime_pid"], 42)
        self.assertEqual(identity["nvml_pid"], 99)

    def test_exclusive_gpu_must_start_idle(self):
        with self.assertRaisesRegex(AssertionError, "already in use"):
            PROBE.measure("GPU-a, 99, 1\n", "GPU-a, 99, 123\n", 42, exclusive=True)

    def test_multiple_or_missing_processes_are_rejected(self):
        for snapshot in ("", "GPU-a, 99, 123\nGPU-a, 100, 1\n"):
            with self.subTest(snapshot=snapshot):
                with self.assertRaisesRegex(AssertionError, "ambiguous"):
                    PROBE.measure("", snapshot, 42, exclusive=True)

    def test_gpu_or_process_identity_cannot_change(self):
        identity, _ = PROBE.measure("", "GPU-a, 99, 123\n", 42, exclusive=True)
        for snapshot in ("GPU-a, 100, 123\n", "GPU-b, 99, 123\n"):
            with self.subTest(snapshot=snapshot):
                with self.assertRaisesRegex(AssertionError, "identity changed"):
                    PROBE.measure("", snapshot, 42, identity, exclusive=True)
        _, memory = PROBE.measure("", "GPU-a, 99, 100\n", 42, identity, exclusive=True)
        self.assertEqual(memory, 100 * 1024 * 1024)

    def test_unavailable_memory_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "unavailable"):
            PROBE.measure("", "GPU-a, 99, [N/A]\n", 42, exclusive=True)


if __name__ == "__main__":
    unittest.main()
