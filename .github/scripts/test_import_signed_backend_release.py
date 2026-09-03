#!/usr/bin/env python3
"""Host-only tests for immutable signed backend-release import."""

from __future__ import annotations

import base64
import functools
import hashlib
import http.server
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import threading
import types
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("import-signed-backend-release.py")
SPEC = importlib.util.spec_from_file_location("import_signed_backend_release", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release_import = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_import)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


class ReleaseFixture:
    VERSION = "1.2.3"
    PACK_VERSION = "4.5.6"
    PLATFORM = "linux-x86_64"
    PROFILES = ("cpu", "cuda12", "tensorrt10")

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.release_dir = root / "release"
        self.output_dir = root / "output"
        self.release_dir.mkdir()
        self.output_dir.mkdir()
        self.private_key = root / "signing.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(self.private_key)],
            check=True,
            capture_output=True,
        )
        public_der = subprocess.run(
            [
                "openssl",
                "pkey",
                "-in",
                str(self.private_key),
                "-pubout",
                "-outform",
                "DER",
            ],
            check=True,
            capture_output=True,
        ).stdout
        self.public_key = base64.b64encode(public_der[-32:]).decode("ascii")
        handler = functools.partial(QuietHandler, directory=str(self.release_dir))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.lock_path = root / "release.lock.json"
        self.part_paths: list[pathlib.Path] = []

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def signature(self, digest: str) -> str:
        message = self.root / "message"
        message.write_bytes(
            release_import.ARTIFACT_DOMAIN + f"sha256:{digest}".encode("ascii")
        )
        completed = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(self.private_key),
                "-in",
                str(message),
            ],
            check=True,
            capture_output=True,
        )
        return "ed25519:" + base64.b64encode(completed.stdout).decode("ascii")

    @staticmethod
    def json_bytes(value: object) -> bytes:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def write(self, name: str, payload: bytes) -> tuple[pathlib.Path, dict[str, object]]:
        path = self.release_dir / name
        path.write_bytes(payload)
        return path, {
            "name": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }

    def build(self, escaped_part_url: bool = False) -> None:
        release_tag = f"kapsl-ort-packs-v{self.PACK_VERSION}-kapsl-v{self.VERSION}"
        identity = {
            "release_tag": release_tag,
            "source_repository": "https://github.com/example/integrations",
            "source_commit": "a" * 40,
            "backend": "onnx",
            "pack_version": self.PACK_VERSION,
            "compatible_kapsl": f"={self.VERSION}",
            "platform": self.PLATFORM,
        }
        profiles: dict[str, object] = {}
        for profile in self.PROFILES:
            archive_name = (
                f"kapsl-backend-onnx-{profile}-{self.VERSION}-{self.PLATFORM}.tar.gz"
            )
            archive_bytes = (f"signed archive for {profile}\n" * 7).encode("ascii")
            archive_digest = hashlib.sha256(archive_bytes).hexdigest()
            archive_signature = self.signature(archive_digest)
            split = max(1, len(archive_bytes) // 2)
            parts = []
            for number, payload in enumerate((archive_bytes[:split], archive_bytes[split:])):
                part_name = f"{archive_name}.part-{number:03d}"
                part_path, part = self.write(part_name, payload)
                self.part_paths.append(part_path)
                part["url"] = f"{self.base_url}/{part_name}"
                parts.append(part)
            if escaped_part_url and profile == "cpu":
                parts[0]["url"] = f"{self.base_url}/../outside/{parts[0]['name']}"

            manifest = {
                "schema_version": 1,
                "backend": "onnx",
                "profile": profile,
                "pack_version": self.PACK_VERSION,
                "runtime_abi": 1,
                "adapter_abi": "kapsl-backend-v1",
                "compatible_kapsl": f"={self.VERSION}",
                "platform": self.PLATFORM,
                "architecture": "x86_64",
                "accelerator_profile": "cpu" if profile == "cpu" else "cuda",
                "execution_mode": "native",
                "entrypoint": "libbackend.so",
                "installed_bytes": 1,
                "files": {"libbackend.so": "0" * 64},
                "licenses": [{"name": "fixture", "path": "libbackend.so"}],
            }
            _, manifest_metadata = self.write(
                f"{archive_name}.manifest.json", self.json_bytes(manifest)
            )
            _, checksum_metadata = self.write(
                f"{archive_name}.sha256",
                f"{archive_digest}  {archive_name}\n".encode("ascii"),
            )
            _, signature_metadata = self.write(
                f"{archive_name}.sig", f"{archive_signature}\n".encode("ascii")
            )
            archive = {
                "name": archive_name,
                "sha256": archive_digest,
                "size": len(archive_bytes),
                "signature": archive_signature,
                "signature_asset": signature_metadata,
                "manifest": manifest_metadata,
                "checksum": checksum_metadata,
                "parts": parts,
            }
            profile_catalog = {
                "schema_version": 1,
                **identity,
                "profile": profile,
                "archive": archive,
            }
            catalog_name = f"{archive_name}.release.json"
            catalog_path, catalog_metadata = self.write(
                catalog_name, self.json_bytes(profile_catalog)
            )
            catalog_signature = self.signature(str(catalog_metadata["sha256"]))
            self.write(
                f"{catalog_name}.sig", f"{catalog_signature}\n".encode("ascii")
            )
            profiles[profile] = {
                "catalog": {
                    "name": catalog_name,
                    "url": f"{self.base_url}/{catalog_name}",
                    "sha256": catalog_metadata["sha256"],
                    "signature": catalog_signature,
                },
                "archive": archive,
            }
            self.assert_fixture_path(catalog_path)

        top_catalog = {"schema_version": 1, **identity, "profiles": profiles}
        top_name = (
            f"kapsl-ort-packs-v{self.PACK_VERSION}-kapsl-v{self.VERSION}"
            f"-{self.PLATFORM}.release.json"
        )
        top_path, top_metadata = self.write(top_name, self.json_bytes(top_catalog))
        top_signature = self.signature(str(top_metadata["sha256"]))
        _, top_signature_metadata = self.write(
            f"{top_name}.sig", f"{top_signature}\n".encode("ascii")
        )
        lock = {
            "schema_version": 1,
            "repository": "example/integrations",
            "release_tag": release_tag,
            "source_commit": identity["source_commit"],
            "backend": "onnx",
            "pack_version": self.PACK_VERSION,
            "compatible_kapsl": f"={self.VERSION}",
            "platform": self.PLATFORM,
            "profiles": list(self.PROFILES),
            "catalog": {
                "name": top_name,
                "sha256": top_metadata["sha256"],
                "size": top_metadata["size"],
                "signature": top_signature,
                "signature_asset": top_signature_metadata,
            },
        }
        self.lock_path.write_bytes(self.json_bytes(lock))
        self.assert_fixture_path(top_path)

    def args(self, version: str | None = None) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            version=version or self.VERSION,
            lock=self.lock_path,
            artifacts_dir=self.output_dir,
            expected_public_key=[self.public_key],
            release_base_url=self.base_url,
            allow_http_test_url=True,
        )

    def assert_fixture_path(self, path: pathlib.Path) -> None:
        if not path.is_file() or not path.is_relative_to(self.release_dir):
            raise AssertionError(f"invalid fixture path {path}")


class SignedBackendReleaseImportTests(unittest.TestCase):
    def with_fixture(self) -> tuple[tempfile.TemporaryDirectory[str], ReleaseFixture]:
        temporary = tempfile.TemporaryDirectory(prefix="kapsl-release-import-test-")
        fixture = ReleaseFixture(pathlib.Path(temporary.name))
        self.addCleanup(fixture.close)
        self.addCleanup(temporary.cleanup)
        return temporary, fixture

    def test_imports_every_profile_after_full_signature_and_digest_verification(self) -> None:
        _, fixture = self.with_fixture()
        fixture.build()

        release_import.import_release(fixture.args())

        for profile in fixture.PROFILES:
            archive_name = (
                f"kapsl-backend-onnx-{profile}-{fixture.VERSION}-{fixture.PLATFORM}.tar.gz"
            )
            imported = fixture.output_dir / archive_name
            self.assertTrue(imported.is_file())
            self.assertEqual(
                imported.read_bytes(),
                (f"signed archive for {profile}\n" * 7).encode("ascii"),
            )
            self.assertTrue((fixture.output_dir / f"{archive_name}.manifest.json").is_file())
            self.assertTrue((fixture.output_dir / f"{archive_name}.sha256").is_file())
            self.assertTrue((fixture.output_dir / f"{archive_name}.sig").is_file())

    def test_tampered_transport_part_fails_without_publishing_partial_outputs(self) -> None:
        _, fixture = self.with_fixture()
        fixture.build()
        fixture.part_paths[0].write_bytes(b"tampered")

        with self.assertRaisesRegex(release_import.ReleaseImportError, "part 0 (size|SHA-256)"):
            release_import.import_release(fixture.args())

        self.assertEqual(list(fixture.output_dir.iterdir()), [])

    def test_runtime_version_must_match_the_exact_locked_compatibility(self) -> None:
        _, fixture = self.with_fixture()
        fixture.build()

        with self.assertRaisesRegex(release_import.ReleaseImportError, "targets Kapsl"):
            release_import.import_release(fixture.args(version="1.2.4"))

        self.assertEqual(list(fixture.output_dir.iterdir()), [])

    def test_signed_transport_urls_cannot_escape_the_locked_release(self) -> None:
        _, fixture = self.with_fixture()
        fixture.build(escaped_part_url=True)

        with self.assertRaisesRegex(release_import.ReleaseImportError, "URL is not bound"):
            release_import.import_release(fixture.args())

        self.assertEqual(list(fixture.output_dir.iterdir()), [])

    def test_tampered_top_catalog_is_rejected_before_profile_downloads(self) -> None:
        _, fixture = self.with_fixture()
        fixture.build()
        lock = json.loads(fixture.lock_path.read_text(encoding="utf-8"))
        top_path = fixture.release_dir / lock["catalog"]["name"]
        top_path.write_bytes(top_path.read_bytes() + b" ")

        with self.assertRaisesRegex(
            release_import.ReleaseImportError, "release catalog (size|SHA-256)"
        ):
            release_import.import_release(fixture.args())

        self.assertEqual(list(fixture.output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
