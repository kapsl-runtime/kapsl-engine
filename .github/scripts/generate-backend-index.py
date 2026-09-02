#!/usr/bin/env python3
"""Generate and Ed25519-sign the immutable Kapsl backend index."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import subprocess
import tarfile
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit


INDEX_DOMAIN = b"kapsl-backend-index-v1\0"
ARTIFACT_DOMAIN = b"kapsl-backend-artifact-v1\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--artifacts-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--signing-key", type=pathlib.Path, required=True)
    parser.add_argument("--base-url", default="https://downloads.kapsl.net")
    parser.add_argument("--allow-insecure-test-url", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--channel", choices=("stable", "beta"), default="stable")
    parser.add_argument(
        "--expected-public-key",
        action="append",
        default=[],
        help=(
            "Trusted Ed25519 public key(s), as raw 32-byte base64 values. "
            "Comma, semicolon, and whitespace-separated key lists are accepted."
        ),
    )
    return parser.parse_args()


def sign(key: pathlib.Path, message: bytes) -> str:
    with tempfile.NamedTemporaryFile() as source:
        source.write(message)
        source.flush()
        completed = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(key),
                "-in",
                source.name,
            ],
            check=True,
            capture_output=True,
        )
    if len(completed.stdout) != 64:
        raise SystemExit(
            f"OpenSSL emitted a {len(completed.stdout)} byte signature; expected Ed25519 (64 bytes)"
        )
    return "ed25519:" + base64.b64encode(completed.stdout).decode("ascii")


def signing_public_key(key: pathlib.Path) -> str:
    completed = subprocess.run(
        ["openssl", "pkey", "-in", str(key), "-pubout", "-outform", "DER"],
        check=True,
        capture_output=True,
    )
    # RFC 8410 Ed25519 SubjectPublicKeyInfo is a fixed 12-byte prefix followed
    # by the raw 32-byte public key accepted by ed25519-dalek.
    if len(completed.stdout) != 44:
        raise SystemExit(
            "The backend signing key is not an Ed25519 private key "
            f"(public DER length was {len(completed.stdout)}, expected 44)"
        )
    return base64.b64encode(completed.stdout[-32:]).decode("ascii")


def expected_public_keys(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        normalized = value.replace(",", " ").replace(";", " ")
        for candidate in normalized.split():
            candidate = candidate.removeprefix("ed25519:")
            try:
                raw = base64.b64decode(candidate, validate=True)
            except ValueError as error:
                raise SystemExit(f"Invalid expected backend public key: {error}") from error
            if len(raw) != 32:
                raise SystemExit(
                    "Expected backend Ed25519 public keys must contain 32 raw bytes"
                )
            parsed.add(base64.b64encode(raw).decode("ascii"))
    return parsed


def canonical_platform_path(channel: str, version: str) -> str:
    if channel == "beta":
        return f"runtime/beta/v{version}"
    return f"runtime/v{version}"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def payload_manifest(archive_path: pathlib.Path) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if pathlib.PurePosixPath(member.name).name != "backend-pack.json":
                continue
            if not member.isfile():
                raise SystemExit(f"{archive_path}: backend-pack.json is not a file")
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit(f"{archive_path}: cannot read backend-pack.json")
            matches.append(json.load(stream))
    if len(matches) != 1:
        raise SystemExit(
            f"{archive_path}: expected exactly one backend-pack.json, found {len(matches)}"
        )
    return matches[0]


def validate_payload(template: dict[str, Any], payload: dict[str, Any], archive: pathlib.Path) -> None:
    fields = (
        "schema_version",
        "backend",
        "profile",
        "pack_version",
        "runtime_abi",
        "adapter_abi",
        "platform",
        "execution_mode",
        "entrypoint",
    )
    mismatches = [field for field in fields if template.get(field) != payload.get(field)]
    if mismatches:
        raise SystemExit(
            f"{archive}: payload/template mismatch for {', '.join(mismatches)}"
        )


def validate_extract_file_hashes(template: dict[str, Any], archive_path: pathlib.Path) -> None:
    """Match every signed installed-file hash against an extract archive.

    Bootstrap packs describe their post-bootstrap tree, so their installed
    files are validated by the bootstrap contract. Extract packs have no such
    transformation and can be checked byte-for-byte before publication.
    """
    if template.get("installer", {"kind": "extract"}).get("kind") != "extract":
        return

    expected = {str(path): str(digest).lower() for path, digest in template["files"].items()}
    actual: dict[str, str] = {}
    verified_installed_bytes = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = pathlib.PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(f"{archive_path}: unsafe archive path {member.name}")
            relative = member_path.as_posix().removeprefix("./")
            if relative not in expected:
                continue
            if not member.isfile():
                raise SystemExit(
                    f"{archive_path}: signed installed file is not regular: {relative}"
                )
            if relative in actual:
                raise SystemExit(
                    f"{archive_path}: duplicate signed installed file: {relative}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit(f"{archive_path}: cannot read signed installed file {relative}")
            digest = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
                verified_installed_bytes += len(block)
            actual[relative] = digest.hexdigest()

    missing = sorted(set(expected) - set(actual))
    if missing:
        raise SystemExit(
            f"{archive_path}: signed installed files are missing: {', '.join(missing)}"
        )
    mismatched = sorted(path for path in expected if expected[path] != actual[path])
    if mismatched:
        raise SystemExit(
            f"{archive_path}: installed file digest mismatch: {', '.join(mismatched)}"
        )
    if verified_installed_bytes > template["installed_bytes"]:
        raise SystemExit(
            f"{archive_path}: signed installed files require {verified_installed_bytes} bytes, "
            f"exceeding installed_bytes={template['installed_bytes']}"
        )


def validate_relative_path(value: Any, label: str, source: pathlib.Path) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{source}: {label} must be a non-empty relative path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SystemExit(f"{source}: {label} is not a normalized relative path: {value}")
    return value


def validate_template(template: dict[str, Any], source: pathlib.Path) -> None:
    if template.get("schema_version") != 1 or template.get("runtime_abi") != 1:
        raise SystemExit(f"{source}: only backend schema/ABI version 1 is publishable")
    for field in (
        "backend",
        "profile",
        "pack_version",
        "compatible_kapsl",
        "platform",
        "architecture",
        "accelerator_profile",
    ):
        if not isinstance(template.get(field), str) or not template[field]:
            raise SystemExit(f"{source}: {field} must be a non-empty string")
    if template["accelerator_profile"] not in ("cpu", "cuda", "tensorrt"):
        raise SystemExit(f"{source}: unsupported accelerator_profile")
    if template.get("execution_mode") not in ("native", "external"):
        raise SystemExit(f"{source}: unsupported execution_mode")
    adapter_abi = template.get("adapter_abi")
    if adapter_abi is not None:
        if template["execution_mode"] != "native":
            raise SystemExit(f"{source}: only native packs may declare adapter_abi")
        if adapter_abi != "kapsl-backend-v1":
            raise SystemExit(f"{source}: unsupported adapter_abi")
    kv_mode = template.get("kv_mode")
    if template.get("backend") == "llama-cpp" and kv_mode not in (
        "native",
        "shared_pool",
    ):
        raise SystemExit(f"{source}: unsupported kv_mode for {template.get('backend')}")
    if template.get("backend") != "llama-cpp" and kv_mode is not None:
        raise SystemExit(f"{source}: unsupported kv_mode for {template.get('backend')}")
    validate_relative_path(template.get("entrypoint"), "entrypoint", source)
    installed_bytes = template.get("installed_bytes")
    if not isinstance(installed_bytes, int) or isinstance(installed_bytes, bool) or installed_bytes <= 0:
        raise SystemExit(f"{source}: installed_bytes must be a positive integer")
    memory = template.get("memory", {})
    if not isinstance(memory, dict):
        raise SystemExit(f"{source}: memory must be an object")
    for field in (
        "host_bytes",
        "accelerator_bytes",
        "workspace_weight_ppm",
        "minimum_workspace_bytes",
    ):
        value = memory.get(field, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SystemExit(f"{source}: memory.{field} must be a non-negative integer")
    installer = template.get("installer", {"kind": "extract"})
    if not isinstance(installer, dict) or installer.get("kind") not in ("extract", "bootstrap"):
        raise SystemExit(f"{source}: installer.kind must be extract or bootstrap")
    if installer.get("kind") == "bootstrap":
        validate_relative_path(installer.get("path"), "installer.path", source)
    files = template.get("files", {})
    if not isinstance(files, dict):
        raise SystemExit(f"{source}: files must be an object")
    for path, digest in files.items():
        validate_relative_path(path, "files key", source)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise SystemExit(f"{source}: invalid SHA-256 for installed file {path}")
    if template["entrypoint"] not in files:
        raise SystemExit(
            f"{source}: entrypoint requires a signed installed-file checksum"
        )
    for field in ("minimum_cuda", "minimum_driver"):
        value = template.get(field)
        if value is not None and (
            not isinstance(value, str) or re.search(r"[0-9]", value) is None
        ):
            raise SystemExit(f"{source}: {field} must contain a numeric version")
    licenses = template.get("licenses", [])
    if not isinstance(licenses, list) or not licenses:
        raise SystemExit(f"{source}: licenses must be a non-empty array")
    packaged_license = False
    for notice in licenses:
        if (
            not isinstance(notice, dict)
            or not isinstance(notice.get("name"), str)
            or not notice["name"].strip()
        ):
            raise SystemExit(f"{source}: each license notice requires a non-empty name")
        if "path" in notice:
            path = validate_relative_path(notice["path"], "license path", source)
            if path not in files:
                raise SystemExit(
                    f"{source}: packaged license {path} requires a signed file checksum"
                )
            packaged_license = True
    if not packaged_license:
        raise SystemExit(f"{source}: at least one packaged license file is required")
    priority = template.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise SystemExit(f"{source}: priority must be an integer")


def main() -> None:
    args = parse_args()
    parsed_base_url = urlsplit(args.base_url)
    if not parsed_base_url.netloc or (
        parsed_base_url.scheme != "https"
        and not (args.allow_insecure_test_url and parsed_base_url.scheme == "http")
    ):
        raise SystemExit("Backend artifact base URL must be an absolute HTTPS URL")
    trusted_keys = expected_public_keys(args.expected_public_key)
    if trusted_keys:
        actual_public_key = signing_public_key(args.signing_key)
        if actual_public_key not in trusted_keys:
            raise SystemExit(
                "The backend signing key does not match any public key embedded "
                "in this Kapsl runtime release"
            )
    templates = sorted(args.artifacts_dir.glob("*.tar.gz.manifest.json"))
    if not templates:
        raise SystemExit(f"No backend manifest templates found in {args.artifacts_dir}")
    packs: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, str, str, str]] = set()
    release_path = canonical_platform_path(args.channel, args.version)
    for template_path in templates:
        archive_name = template_path.name.removesuffix(".manifest.json")
        archive_path = args.artifacts_dir / archive_name
        if not archive_path.is_file():
            raise SystemExit(f"Missing artifact for {template_path}: {archive_path}")
        template = json.loads(template_path.read_text(encoding="utf-8"))
        validate_template(template, template_path)
        if template.get("compatible_kapsl") != f"={args.version}":
            raise SystemExit(
                f"{template_path}: compatible_kapsl must resolve from the one release version"
            )
        validate_payload(template, payload_manifest(archive_path), archive_path)
        validate_extract_file_hashes(template, archive_path)
        identity = (
            str(template.get("backend")),
            str(template.get("profile")),
            str(template.get("pack_version")),
            str(template.get("platform")),
            str(template.get("architecture")),
            str(template.get("accelerator_profile")),
        )
        if identity in identities:
            raise SystemExit(
                "Duplicate backend identity: "
                f"{identity[0]}/{identity[1]} version {identity[2]} "
                f"for {identity[3]}-{identity[4]} ({identity[5]})"
            )
        identities.add(identity)
        digest = sha256(archive_path)
        artifact_message = ARTIFACT_DOMAIN + f"sha256:{digest}".encode("ascii")
        pack = dict(template)
        pack.update(
            {
                "artifact": f"{args.base_url.rstrip('/')}/{release_path}/{archive_name}",
                "download_bytes": archive_path.stat().st_size,
                "sha256": digest,
                "signature": sign(args.signing_key, artifact_message),
            }
        )
        packs.append(pack)

    packs.sort(
        key=lambda item: (
            item["backend"],
            item["profile"],
            item["platform"],
            item["architecture"],
            item["accelerator_profile"],
            item["pack_version"],
        )
    )
    index = {
        "schema_version": 1,
        "runtime_version": args.version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "packs": packs,
    }
    index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(index_bytes)
    args.output.with_name(args.output.name + ".sig").write_text(
        sign(args.signing_key, INDEX_DOMAIN + index_bytes) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
