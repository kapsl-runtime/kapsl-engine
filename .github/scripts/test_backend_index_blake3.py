#!/usr/bin/env python3
"""Host-only publication checks for the optional signed digest extension."""

import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from blake3 import blake3

SPEC = importlib.util.spec_from_file_location(
    "backend_index", Path(__file__).with_name("generate-backend-index.py")
)
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)


class Blake3IndexTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.archive = Path(self.root.name) / "pack.tar.gz"
        self.payload = {"bin/adapter": b"adapter", "licenses/LICENSE": b"license"}
        self.template = {
            "schema_version": 1,
            "runtime_abi": 1,
            "backend": "test",
            "profile": "cpu",
            "pack_version": "1.0.0",
            "compatible_kapsl": "=1.2.3",
            "platform": "linux-x86_64",
            "architecture": "x86_64",
            "accelerator_profile": "cpu",
            "execution_mode": "native",
            "entrypoint": "bin/adapter",
            "installed_bytes": 4096,
            "files": {p: hashlib.sha256(b).hexdigest() for p, b in self.payload.items()},
            "files_blake3": {p: blake3(b).hexdigest() for p, b in self.payload.items()},
            "licenses": [{"name": "Fixture", "path": "licenses/LICENSE"}],
        }
        with tarfile.open(self.archive, "w:gz") as archive:
            for path, content in self.payload.items():
                member = tarfile.TarInfo(path)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    def test_legacy_and_extended_archives(self):
        for extended in [False, True]:
            with self.subTest(extended=extended):
                template = copy.deepcopy(self.template)
                if not extended:
                    del template["files_blake3"]
                else:
                    template["files_blake3"] = {
                        p: digest.upper() for p, digest in template["files_blake3"].items()
                    }
                INDEX.validate_template(template, self.archive)
                INDEX.validate_extract_file_hashes(template, self.archive)

    def test_incomplete_extra_and_malformed_digests(self):
        for invalid in [{}, {"extra": "0" * 64}, [], "digest", {
            "bin/adapter": "g" * 64, "licenses/LICENSE": "0" * 64,
        }]:
            with self.subTest(invalid=invalid):
                template = copy.deepcopy(self.template)
                template["files_blake3"] = invalid
                with self.assertRaises(SystemExit):
                    INDEX.validate_template(template, self.archive)

    def test_mismatching_either_digest_is_rejected_before_signing(self):
        for algorithm in ["files", "files_blake3"]:
            for path in self.payload:
                with self.subTest(algorithm=algorithm, path=path):
                    template = copy.deepcopy(self.template)
                    template[algorithm][path] = "0" * 64
                    with self.assertRaises(SystemExit):
                        INDEX.validate_extract_file_hashes(template, self.archive)


if __name__ == "__main__":
    unittest.main()
