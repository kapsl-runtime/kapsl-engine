#!/usr/bin/env python3
"""Secret-free, host-only readiness check for an engine's immutable ORT release."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import tempfile

from validate_stable_gpu_release import runtime_version


SPEC = importlib.util.spec_from_file_location(
    "release_import", Path(__file__).with_name("import-signed-backend-release.py")
)
assert SPEC and SPEC.loader
release_import = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_import)


def preflight(manifest: Path, lock_path: Path, key_file: Path, require_runtime_trust: bool) -> None:
    version = runtime_version(manifest)
    release_import.require_string(version, "engine version", release_import.STABLE_VERSION)
    lock = release_import.read_json(lock_path, "ORT release lock")
    identity, profiles = release_import.validate_lock(lock, version)
    if identity["backend"] != "onnx" or identity["platform"] != "linux-x86_64":
        raise release_import.ReleaseImportError("ORT release must target onnx/linux-x86_64")
    if set(profiles) != {"cpu", "cuda12", "tensorrt10"}:
        raise release_import.ReleaseImportError("ORT release must contain CPU, CUDA12 and TensorRT10")
    public_key = key_file.read_text(encoding="utf-8").strip()
    pinned_keys = release_import.trusted_public_keys([public_key])
    if require_runtime_trust:
        runtime_keys = release_import.trusted_public_keys([
            os.environ.get("KAPSL_BACKEND_PUBLIC_KEYS", "")
        ])
        if not set(pinned_keys).issubset(runtime_keys):
            raise release_import.ReleaseImportError(
                "ORT signing key is missing from engine runtime trust; "
                "check KAPSL_ORT_BACKEND_PUBLIC_KEY before releasing"
            )
    with tempfile.TemporaryDirectory(prefix="kapsl-ort-preflight-") as temporary:
        release_import.import_release(argparse.Namespace(
            version=version, lock=lock_path, expected_public_key=[public_key],
            artifacts_dir=Path(temporary), metadata_only=True,
            release_base_url=None, allow_http_test_url=False,
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("kapsl-runtime/crates/kapsl-cli/Cargo.toml"))
    parser.add_argument("--lock", type=Path, default=Path(".github/ort-release.lock.json"))
    parser.add_argument("--public-key-file", type=Path, default=Path(".github/ort-release-public-key.txt"))
    parser.add_argument("--require-runtime-trust", action="store_true")
    args = parser.parse_args()
    try:
        preflight(args.manifest, args.lock, args.public_key_file, args.require_runtime_trust)
    except (OSError, ValueError, release_import.ReleaseImportError) as error:
        raise SystemExit(f"ORT release preflight failed: {error}") from error


if __name__ == "__main__":
    main()
