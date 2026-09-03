#!/usr/bin/env python3
"""Import immutable backend archives from a signed integration release catalog."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping


ARTIFACT_DOMAIN = b"kapsl-backend-artifact-v1\0"
PUBLIC_KEY_DER_PREFIX = bytes.fromhex("302a300506032b6570032100")
STABLE_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
SAFE_ASSET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
READ_BLOCK_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024


class ReleaseImportError(RuntimeError):
    """A signed release failed identity, authenticity, or integrity checks."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--lock", type=pathlib.Path, required=True)
    parser.add_argument("--artifacts-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--expected-public-key",
        action="append",
        default=[],
        help=(
            "Trusted Ed25519 public key(s), encoded as raw 32-byte Base64. "
            "Comma, semicolon, and whitespace-separated lists are accepted."
        ),
    )
    parser.add_argument("--release-base-url", help=argparse.SUPPRESS)
    parser.add_argument("--allow-http-test-url", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseImportError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseImportError(f"{label} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ReleaseImportError(f"{label} has an invalid value: {value!r}")
    return value


def require_positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReleaseImportError(f"{label} must be a positive integer")
    return value


def require_asset_name(value: Any, label: str) -> str:
    return require_string(value, label, SAFE_ASSET)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: pathlib.Path, label: str) -> Mapping[str, Any]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            raise ReleaseImportError(f"{label} exceeds {MAX_METADATA_BYTES} bytes")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseImportError(f"cannot read {label} {path}: {error}") from error
    return require_mapping(value, label)


def parse_signature(value: str, label: str) -> tuple[str, bytes]:
    if not value.startswith("ed25519:"):
        raise ReleaseImportError(f"{label} must use ed25519 encoding")
    try:
        raw = base64.b64decode(value.removeprefix("ed25519:"), validate=True)
    except ValueError as error:
        raise ReleaseImportError(f"{label} is invalid Base64") from error
    if len(raw) != 64:
        raise ReleaseImportError(f"{label} must contain a 64-byte Ed25519 signature")
    return value, raw


def trusted_public_keys(values: Iterable[str]) -> list[bytes]:
    parsed: list[bytes] = []
    for value in values:
        normalized = value.replace(",", " ").replace(";", " ")
        for candidate in normalized.split():
            candidate = candidate.removeprefix("ed25519:")
            try:
                raw = base64.b64decode(candidate, validate=True)
            except ValueError as error:
                raise ReleaseImportError("expected public key is invalid Base64") from error
            if len(raw) != 32:
                raise ReleaseImportError("expected public key must contain 32 raw bytes")
            if raw not in parsed:
                parsed.append(raw)
    if not parsed:
        raise ReleaseImportError("at least one trusted backend signing public key is required")
    return parsed


def verify_signature(keys: Iterable[bytes], digest: str, signature: str, label: str) -> None:
    require_string(digest, f"{label} digest", HEX_SHA256)
    _, raw_signature = parse_signature(signature, f"{label} signature")
    message = ARTIFACT_DOMAIN + f"sha256:{digest}".encode("ascii")
    with tempfile.TemporaryDirectory(prefix="kapsl-release-signature-") as temporary:
        root = pathlib.Path(temporary)
        message_path = root / "message"
        signature_path = root / "signature"
        message_path.write_bytes(message)
        signature_path.write_bytes(raw_signature)
        for number, key in enumerate(keys):
            key_path = root / f"public-{number}.der"
            key_path.write_bytes(PUBLIC_KEY_DER_PREFIX + key)
            completed = subprocess.run(
                [
                    "openssl",
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-keyform",
                    "DER",
                    "-inkey",
                    str(key_path),
                    "-rawin",
                    "-in",
                    str(message_path),
                    "-sigfile",
                    str(signature_path),
                ],
                check=False,
                capture_output=True,
            )
            if completed.returncode == 0:
                return
    raise ReleaseImportError(f"{label} signature does not match a trusted release key")


def release_base_url(
    repository: str,
    release_tag: str,
    override: str | None,
    allow_http_test_url: bool,
) -> str:
    expected = f"https://github.com/{repository}/releases/download/{release_tag}"
    value = (override or expected).rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if override is None:
        if value != expected:
            raise ReleaseImportError("release URL does not match the locked repository and tag")
    elif not allow_http_test_url:
        raise ReleaseImportError("a release URL override is permitted only by host-only tests")
    elif parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ReleaseImportError("test release URLs must use HTTP loopback")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ReleaseImportError("release URL must not contain credentials, a query, or a fragment")
    return value


def asset_url(base_url: str, name: str) -> str:
    require_asset_name(name, "release asset name")
    return f"{base_url}/{name}"


def download(url: str, destination: pathlib.Path) -> None:
    if destination.exists():
        raise ReleaseImportError(f"refusing to overwrite download destination {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        temporary = destination.with_name(f".{destination.name}.download-{attempt}")
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "kapsl-release-import/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
                "xb"
            ) as output:
                while block := response.read(READ_BLOCK_BYTES):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
            return
        except (OSError, http.client.HTTPException, urllib.error.URLError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt)
    raise ReleaseImportError(f"download failed for {url}: {last_error}")


def verify_bound_file(
    path: pathlib.Path,
    expected_size: int | None,
    expected_digest: str,
    label: str,
) -> None:
    expected_digest = require_string(expected_digest, f"{label} sha256", HEX_SHA256)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ReleaseImportError(
            f"{label} size is {path.stat().st_size}; expected {expected_size}"
        )
    actual = sha256_file(path)
    if actual != expected_digest:
        raise ReleaseImportError(f"{label} SHA-256 is {actual}; expected {expected_digest}")


def bound_asset(
    metadata: Any,
    label: str,
    base_url: str,
    destination_dir: pathlib.Path,
) -> pathlib.Path:
    item = require_mapping(metadata, label)
    name = require_asset_name(item.get("name"), f"{label}.name")
    size = require_positive_integer(item.get("size"), f"{label}.size")
    digest = require_string(item.get("sha256"), f"{label}.sha256", HEX_SHA256)
    path = destination_dir / name
    download(asset_url(base_url, name), path)
    verify_bound_file(path, size, digest, label)
    return path


def validate_identity(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if payload.get("schema_version") != 1:
        raise ReleaseImportError(f"{label} has an unsupported schema version")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ReleaseImportError(
                f"{label} {field} is {payload.get(field)!r}; expected {value!r}"
            )


def validate_lock(lock: Mapping[str, Any], version: str) -> tuple[dict[str, Any], list[str]]:
    if lock.get("schema_version") != 1:
        raise ReleaseImportError("release lock has an unsupported schema version")
    repository = require_string(lock.get("repository"), "lock.repository", REPOSITORY)
    release_tag = require_string(lock.get("release_tag"), "lock.release_tag", SAFE_ASSET)
    source_commit = require_string(lock.get("source_commit"), "lock.source_commit", HEX_COMMIT)
    backend = require_string(lock.get("backend"), "lock.backend", SAFE_ASSET)
    pack_version = require_string(lock.get("pack_version"), "lock.pack_version", STABLE_VERSION)
    compatible_kapsl = require_string(lock.get("compatible_kapsl"), "lock.compatible_kapsl")
    platform = require_string(lock.get("platform"), "lock.platform", SAFE_ASSET)
    if compatible_kapsl != f"={version}":
        raise ReleaseImportError(
            f"release lock targets Kapsl {compatible_kapsl!r}; expected '={version}'"
        )
    profiles_value = lock.get("profiles")
    if not isinstance(profiles_value, list) or not profiles_value:
        raise ReleaseImportError("lock.profiles must be a non-empty array")
    profiles = [require_string(value, "lock profile", SAFE_ASSET) for value in profiles_value]
    if len(profiles) != len(set(profiles)):
        raise ReleaseImportError("lock.profiles contains duplicates")
    identity = {
        "release_tag": release_tag,
        "source_repository": f"https://github.com/{repository}",
        "source_commit": source_commit,
        "backend": backend,
        "pack_version": pack_version,
        "compatible_kapsl": compatible_kapsl,
        "platform": platform,
    }
    return identity, profiles


def validate_archive_metadata(
    archive_value: Any,
    backend: str,
    profile: str,
    version: str,
    platform: str,
    base_url: str,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    archive = require_mapping(archive_value, f"{profile} archive")
    expected_name = f"kapsl-backend-{backend}-{profile}-{version}-{platform}.tar.gz"
    if archive.get("name") != expected_name:
        raise ReleaseImportError(
            f"{profile} archive name is {archive.get('name')!r}; expected {expected_name!r}"
        )
    archive_size = require_positive_integer(archive.get("size"), f"{profile} archive.size")
    require_string(archive.get("sha256"), f"{profile} archive.sha256", HEX_SHA256)
    parse_signature(
        require_string(archive.get("signature"), f"{profile} archive.signature"),
        f"{profile} archive signature",
    )
    parts_value = archive.get("parts")
    if not isinstance(parts_value, list) or not parts_value:
        raise ReleaseImportError(f"{profile} archive.parts must be a non-empty array")
    parts: list[Mapping[str, Any]] = []
    total_size = 0
    for number, part_value in enumerate(parts_value):
        part = require_mapping(part_value, f"{profile} part {number}")
        expected_part_name = f"{expected_name}.part-{number:03d}"
        if part.get("name") != expected_part_name:
            raise ReleaseImportError(
                f"{profile} part {number} name is not the expected ordered transport name"
            )
        size = require_positive_integer(part.get("size"), f"{profile} part {number}.size")
        require_string(part.get("sha256"), f"{profile} part {number}.sha256", HEX_SHA256)
        expected_url = asset_url(base_url, expected_part_name)
        if part.get("url") != expected_url:
            raise ReleaseImportError(
                f"{profile} part {number} URL is not bound to the locked release"
            )
        total_size += size
        parts.append(part)
    if total_size != archive_size:
        raise ReleaseImportError(
            f"{profile} transport parts total {total_size} bytes; expected {archive_size}"
        )
    return archive, parts


def import_profile(
    profile: str,
    profile_value: Any,
    identity: Mapping[str, Any],
    version: str,
    base_url: str,
    keys: list[bytes],
    stage: pathlib.Path,
) -> None:
    profile_entry = require_mapping(profile_value, f"profile {profile}")
    catalog_metadata = require_mapping(profile_entry.get("catalog"), f"{profile} catalog")
    catalog_name = require_asset_name(catalog_metadata.get("name"), f"{profile} catalog.name")
    expected_catalog_name = (
        f"kapsl-backend-{identity['backend']}-{profile}-{version}-{identity['platform']}"
        ".tar.gz.release.json"
    )
    if catalog_name != expected_catalog_name:
        raise ReleaseImportError(f"{profile} catalog has an unexpected asset name")
    if catalog_metadata.get("url") != asset_url(base_url, catalog_name):
        raise ReleaseImportError(f"{profile} catalog URL is not bound to the locked release")
    catalog_digest = require_string(
        catalog_metadata.get("sha256"), f"{profile} catalog.sha256", HEX_SHA256
    )
    catalog_signature = require_string(
        catalog_metadata.get("signature"), f"{profile} catalog.signature"
    )
    catalog_path = stage / catalog_name
    download(asset_url(base_url, catalog_name), catalog_path)
    verify_bound_file(catalog_path, None, catalog_digest, f"{profile} catalog")
    verify_signature(keys, catalog_digest, catalog_signature, f"{profile} catalog")
    catalog_signature_path = stage / f"{catalog_name}.sig"
    download(asset_url(base_url, catalog_signature_path.name), catalog_signature_path)
    observed_catalog_signature = catalog_signature_path.read_text(encoding="ascii").strip()
    if observed_catalog_signature != catalog_signature:
        raise ReleaseImportError(f"{profile} catalog signature asset differs from the signed index")

    profile_catalog = read_json(catalog_path, f"{profile} catalog")
    expected_identity = dict(identity)
    expected_identity["profile"] = profile
    validate_identity(profile_catalog, expected_identity, f"{profile} catalog")
    if profile_catalog.get("archive") != profile_entry.get("archive"):
        raise ReleaseImportError(f"{profile} catalog archive differs from the signed release index")

    archive, parts = validate_archive_metadata(
        profile_entry.get("archive"),
        str(identity["backend"]),
        profile,
        version,
        str(identity["platform"]),
        base_url,
    )
    archive_name = str(archive["name"])
    assembled = stage / f".{archive_name}.assembling"
    archive_digest = hashlib.sha256()
    archive_size = 0
    parts_dir = stage / ".parts"
    parts_dir.mkdir(exist_ok=True)
    with assembled.open("xb") as output:
        for number, part in enumerate(parts):
            part_name = str(part["name"])
            part_path = parts_dir / part_name
            download(str(part["url"]), part_path)
            verify_bound_file(
                part_path,
                int(part["size"]),
                str(part["sha256"]),
                f"{profile} part {number}",
            )
            with part_path.open("rb") as source:
                while block := source.read(READ_BLOCK_BYTES):
                    output.write(block)
                    archive_digest.update(block)
                    archive_size += len(block)
            part_path.unlink()
        output.flush()
        os.fsync(output.fileno())
    if archive_size != archive["size"] or archive_digest.hexdigest() != archive["sha256"]:
        raise ReleaseImportError(f"{profile} reconstructed archive failed integrity verification")
    verify_signature(keys, str(archive["sha256"]), str(archive["signature"]), f"{profile} archive")
    assembled.replace(stage / archive_name)

    manifest_path = bound_asset(
        archive.get("manifest"), f"{profile} manifest", base_url, stage
    )
    manifest = read_json(manifest_path, f"{profile} manifest")
    manifest_identity = {
        "backend": identity["backend"],
        "profile": profile,
        "pack_version": identity["pack_version"],
        "compatible_kapsl": identity["compatible_kapsl"],
        "platform": identity["platform"],
    }
    validate_identity(manifest, manifest_identity, f"{profile} manifest")

    checksum_path = bound_asset(
        archive.get("checksum"), f"{profile} checksum", base_url, stage
    )
    expected_checksum = f"{archive['sha256']}  {archive_name}\n"
    if checksum_path.read_text(encoding="ascii") != expected_checksum:
        raise ReleaseImportError(f"{profile} checksum does not bind the reconstructed archive")

    signature_path = bound_asset(
        archive.get("signature_asset"), f"{profile} signature asset", base_url, stage
    )
    if signature_path.read_text(encoding="ascii").strip() != archive["signature"]:
        raise ReleaseImportError(f"{profile} signature asset differs from the signed catalog")


def import_release(args: argparse.Namespace) -> None:
    version = require_string(args.version, "runtime version", STABLE_VERSION)
    lock = read_json(args.lock.resolve(), "backend release lock")
    identity, profiles = validate_lock(lock, version)
    keys = trusted_public_keys(args.expected_public_key)
    repository = str(identity["source_repository"]).removeprefix("https://github.com/")
    base_url = release_base_url(
        repository,
        str(identity["release_tag"]),
        args.release_base_url,
        args.allow_http_test_url,
    )

    catalog_lock = require_mapping(lock.get("catalog"), "lock.catalog")
    catalog_name = require_asset_name(catalog_lock.get("name"), "lock.catalog.name")
    if not catalog_name.endswith(".release.json"):
        raise ReleaseImportError("locked release catalog must be a .release.json asset")
    catalog_digest = require_string(
        catalog_lock.get("sha256"), "lock.catalog.sha256", HEX_SHA256
    )
    catalog_size = require_positive_integer(catalog_lock.get("size"), "lock.catalog.size")
    locked_signature = require_string(catalog_lock.get("signature"), "lock.catalog.signature")
    signature_asset = require_mapping(
        catalog_lock.get("signature_asset"), "lock.catalog.signature_asset"
    )

    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".backend-release-", dir=artifacts_dir) as temporary:
        stage = pathlib.Path(temporary)
        catalog_path = stage / catalog_name
        download(asset_url(base_url, catalog_name), catalog_path)
        verify_bound_file(catalog_path, catalog_size, catalog_digest, "release catalog")
        signature_path = bound_asset(
            signature_asset, "release catalog signature asset", base_url, stage
        )
        observed_signature = signature_path.read_text(encoding="ascii").strip()
        if observed_signature != locked_signature:
            raise ReleaseImportError("release catalog signature differs from the pinned lock")
        verify_signature(keys, catalog_digest, locked_signature, "release catalog")

        catalog = read_json(catalog_path, "release catalog")
        validate_identity(catalog, identity, "release catalog")
        catalog_profiles = require_mapping(catalog.get("profiles"), "release catalog profiles")
        if set(catalog_profiles) != set(profiles):
            raise ReleaseImportError(
                "release catalog profiles do not exactly match the pinned required profiles"
            )
        for profile in profiles:
            import_profile(
                profile,
                catalog_profiles[profile],
                identity,
                version,
                base_url,
                keys,
                stage,
            )

        staged_files = sorted(path for path in stage.iterdir() if path.is_file())
        if not staged_files:
            raise ReleaseImportError("signed release import produced no files")
        collisions = [
            artifacts_dir / path.name
            for path in staged_files
            if (artifacts_dir / path.name).exists()
        ]
        if collisions:
            raise ReleaseImportError(
                "refusing to overwrite existing release outputs: "
                + ", ".join(str(path) for path in collisions)
            )
        for path in staged_files:
            path.replace(artifacts_dir / path.name)

    print(
        f"Imported signed {identity['backend']} release {identity['release_tag']} "
        f"for Kapsl {version}: {', '.join(profiles)}"
    )


def main() -> None:
    args = parse_args()
    try:
        import_release(args)
    except (OSError, UnicodeDecodeError, ReleaseImportError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
