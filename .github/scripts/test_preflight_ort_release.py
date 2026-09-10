#!/usr/bin/env python3
"""Host-only tests for early ORT release readiness and runtime trust checks."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from unittest import TestCase, main, mock

from test_import_signed_backend_release import ReleaseFixture


SPEC = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).with_name("preflight-ort-release.py")
)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)
ROOT = Path(__file__).resolve().parents[2]


class PreflightTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.fixture = ReleaseFixture(root)
        self.addCleanup(self.fixture.close)
        self.fixture.build()
        self.manifest = root / "Cargo.toml"
        self.manifest.write_text('[package]\nname = "kapsl"\nversion = "1.2.3"\n')
        self.key = root / "public-key.txt"
        self.key.write_text(self.fixture.public_key)

    def check(self, require_runtime_trust=False):
        preflight.preflight(self.manifest, self.fixture.lock_path, self.key, require_runtime_trust)

    def test_mismatch_fails_before_any_network_access(self):
        self.manifest.write_text('[package]\nversion = "1.2.4"\n')
        with mock.patch.object(preflight.release_import, "import_release") as importer:
            with self.assertRaisesRegex(preflight.release_import.ReleaseImportError, "expected '=1.2.4'"):
                self.check()
            importer.assert_not_called()

    def test_matching_lock_uses_metadata_only_and_pinned_public_key(self):
        with mock.patch.object(preflight.release_import, "import_release") as importer:
            self.check()
            args = importer.call_args.args[0]
            self.assertTrue(args.metadata_only)
            self.assertEqual(args.version, "1.2.3")
            self.assertEqual(args.expected_public_key, [self.fixture.public_key])
            self.assertIsNone(args.release_base_url)
            self.assertFalse(args.allow_http_test_url)

    def test_release_requires_the_pack_signer_in_runtime_trust(self):
        wrong_key = base64.b64encode(bytes(32)).decode("ascii")
        with mock.patch.dict(os.environ, {"KAPSL_BACKEND_PUBLIC_KEYS": wrong_key}):
            with mock.patch.object(preflight.release_import, "import_release") as importer:
                with self.assertRaisesRegex(preflight.release_import.ReleaseImportError, "missing from engine runtime trust"):
                    self.check(require_runtime_trust=True)
                importer.assert_not_called()
        with mock.patch.dict(os.environ, {"KAPSL_BACKEND_PUBLIC_KEYS": wrong_key + " " + self.fixture.public_key}):
            with mock.patch.object(preflight.release_import, "import_release") as importer:
                self.check(require_runtime_trust=True)
                importer.assert_called_once()

    def test_all_three_profiles_are_mandatory(self):
        lock = json.loads(self.fixture.lock_path.read_text())
        lock["profiles"] = ["cpu", "cuda12"]
        self.fixture.lock_path.write_text(json.dumps(lock))
        with self.assertRaisesRegex(preflight.release_import.ReleaseImportError, "TensorRT10"):
            self.check()

    def test_preflight_is_host_only_and_blocks_all_stable_builds(self):
        release = (ROOT / ".github/workflows/release-runtime-installers.yml").read_text()
        prepare = release.split("  prepare-version:\n", 1)[1].split("  publish-docker:\n", 1)[0]
        self.assertIn("if: steps.version.outputs.is_stable_release == 'true'", prepare)
        self.assertIn("preflight-ort-release.py --require-runtime-trust", prepare)
        for job in ("build-runtime-installers", "build-cuda-runtime", "stable-release-cpu-conformance"):
            block = release.split(f"  {job}:\n", 1)[1].split("    steps:", 1)[0]
            self.assertIn("needs: [prepare-version, stable-release-infrastructure-preflight]", block)
        workflow = (ROOT / ".github/workflows/release-preflight.yml").read_text()
        self.assertIn("  pull_request:", workflow)
        self.assertIn("preflight-ort-release.py", workflow)
        pr_job = workflow.split("  ort-release-preflight:\n", 1)[1].split("  live-infrastructure-preflight:\n", 1)[0]
        self.assertNotIn("secrets.", pr_job)
        self.assertNotIn("self-hosted", workflow)
        self.assertNotIn("gpu-device-pool-integration.yml", workflow)

    def test_integration_signer_is_added_without_replacing_engine_signers(self):
        for name in ("release-runtime-installers.yml", "gpu-device-pool-integration.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text()
            self.assertIn(
                "KAPSL_BACKEND_PUBLIC_KEYS: ${{ secrets.KAPSL_BACKEND_PUBLIC_KEYS }} ${{ vars.KAPSL_ORT_BACKEND_PUBLIC_KEY }}",
                workflow,
            )


if __name__ == "__main__":
    main()
