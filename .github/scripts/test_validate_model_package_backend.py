#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import validate_model_package_backend as validator


class PackageContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self, backend: str) -> dict[str, object]:
        is_vllm = backend == "vllm"
        return {
            "project_name": "fixture",
            "framework": "safetensors" if is_vllm else "llm",
            "version": "1.0.0",
            "model_file": "model.safetensors" if is_vllm else "model.gguf",
            "format": "safetensors" if is_vllm else "gguf",
            "model_type": "causal-lm",
            "task": "generate",
            "metadata": {"serving": {"backend": backend}},
        }

    def package(
        self, manifest: dict[str, object], *, duplicate: bool = False
    ) -> Path:
        path = self.root / "fixture.aimod"
        payload = json.dumps(manifest).encode()
        with tarfile.open(path, mode="w:gz") as archive:
            for _ in range(2 if duplicate else 1):
                info = tarfile.TarInfo("metadata.json")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return path

    def test_accepts_certified_vllm_and_llama_contracts(self) -> None:
        for backend in validator.SUPPORTED_BACKENDS:
            with self.subTest(backend=backend):
                evidence = validator.validate_backend_contract(
                    self.package(self.manifest(backend)), backend
                )
                self.assertEqual(evidence["declared_backend"], backend)
                self.assertEqual(evidence["project_name"], "fixture")

    def test_rejects_backend_and_vllm_axis_mismatches(self) -> None:
        package = self.package(self.manifest("vllm"))
        with self.assertRaisesRegex(validator.PackageContractError, "backend mismatch"):
            validator.validate_backend_contract(package, "llama_cpp")

        manifest = self.manifest("vllm")
        manifest["task"] = "embed"
        with self.assertRaisesRegex(validator.PackageContractError, "vLLM requires"):
            validator.validate_backend_contract(self.package(manifest), "vllm")

    def test_rejects_duplicate_or_oversized_metadata(self) -> None:
        duplicate = self.package(self.manifest("vllm"), duplicate=True)
        with self.assertRaisesRegex(validator.PackageContractError, "duplicate"):
            validator.read_manifest(duplicate)

        path = self.root / "oversized.aimod"
        with tarfile.open(path, mode="w:gz") as archive:
            info = tarfile.TarInfo("metadata.json")
            info.size = validator.MAX_MANIFEST_BYTES + 1
            archive.addfile(info, io.BytesIO(b"x" * info.size))
        with self.assertRaisesRegex(validator.PackageContractError, "exceeds"):
            validator.read_manifest(path)

    def test_writes_atomic_evidence(self) -> None:
        package = self.package(self.manifest("vllm"))
        output = self.root / "evidence.json"
        evidence = validator.validate_backend_contract(package, "vllm")
        validator.write_evidence(output, evidence)
        self.assertEqual(json.loads(output.read_text()), evidence)


if __name__ == "__main__":
    unittest.main()
