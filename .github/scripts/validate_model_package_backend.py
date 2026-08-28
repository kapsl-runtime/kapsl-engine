#!/usr/bin/env python3
"""Validate the serving-backend contract embedded in a Kapsl model package."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Sequence


MAX_MANIFEST_BYTES = 1024 * 1024
SUPPORTED_BACKENDS = ("vllm", "llama_cpp")


class PackageContractError(RuntimeError):
    """The package does not contain the expected serving contract."""


def _required_string(manifest: dict[str, Any], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"metadata.json.{name} must be a non-empty string")
    return value


def read_manifest(package: Path) -> dict[str, Any]:
    if not package.is_file():
        raise PackageContractError(f"package does not exist: {package}")

    payload: bytes | None = None
    try:
        with tarfile.open(package, mode="r:gz") as archive:
            for member in archive:
                if member.name != "metadata.json":
                    continue
                if payload is not None:
                    raise PackageContractError("package contains duplicate metadata.json entries")
                if not member.isfile():
                    raise PackageContractError("package metadata.json is not a regular file")
                if member.size > MAX_MANIFEST_BYTES:
                    raise PackageContractError(
                        f"metadata.json exceeds {MAX_MANIFEST_BYTES} bytes"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise PackageContractError("package metadata.json could not be opened")
                payload = stream.read(MAX_MANIFEST_BYTES + 1)
    except (OSError, tarfile.TarError) as error:
        raise PackageContractError(f"invalid package archive: {error}") from error

    if payload is None:
        raise PackageContractError("package does not contain metadata.json")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise PackageContractError(f"metadata.json exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageContractError(f"metadata.json is invalid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise PackageContractError("metadata.json must contain a JSON object")
    return manifest


def validate_backend_contract(
    package: Path, expected_backend: str
) -> dict[str, Any]:
    if expected_backend not in SUPPORTED_BACKENDS:
        raise PackageContractError(f"unsupported expected backend: {expected_backend}")

    manifest = read_manifest(package)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise PackageContractError("metadata.json.metadata must be an object")
    serving = metadata.get("serving")
    if not isinstance(serving, dict):
        raise PackageContractError("metadata.json.metadata.serving must be an object")
    declared_backend = serving.get("backend")
    if declared_backend != expected_backend:
        raise PackageContractError(
            "package serving backend mismatch: "
            f"expected {expected_backend!r}, found {declared_backend!r}"
        )

    model_format = _required_string(manifest, "format")
    model_type = _required_string(manifest, "model_type")
    task = _required_string(manifest, "task")
    if expected_backend == "vllm":
        expected_axes = ("safetensors", "causal-lm", "generate")
        actual_axes = (model_format, model_type, task)
        if actual_axes != expected_axes:
            raise PackageContractError(
                "vLLM requires format/model_type/task "
                f"{expected_axes!r}, found {actual_axes!r}"
            )
    elif model_format != "gguf":
        raise PackageContractError(
            f"llama_cpp requires format 'gguf', found {model_format!r}"
        )

    return {
        "schema_version": 1,
        "package": package.name,
        "project_name": _required_string(manifest, "project_name"),
        "declared_backend": declared_backend,
        "format": model_format,
        "model_type": model_type,
        "task": task,
    }


def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(evidence, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--expected-backend", choices=SUPPORTED_BACKENDS, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = validate_backend_contract(args.package, args.expected_backend)
        write_evidence(args.output, evidence)
    except PackageContractError as error:
        print(f"package contract validation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
