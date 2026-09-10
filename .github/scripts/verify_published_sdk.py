#!/usr/bin/env python3
"""Check SDK pins without builds; recheck actual Cargo resolution before conformance.

Requires Python 3.11+ for the standard-library Cargo.lock TOML reader.
"""

import argparse
import json
from pathlib import Path
import re
import tomllib


CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
# SDK crates have independent release versions. Keep one contract for the cheap
# host-only lockfile check and the GPU job's resolved-metadata check.
EXPECTED_VERSIONS = {
    "kapsl-backend-abi": "0.2.0",
    "kapsl-engine-api": "0.3.0",
    "kapsl-hal": "0.3.0",
    "kapsl-ipc": "0.4.0",
    "kapsl-kv-abi": "0.6.0",
    "kapsl-llm": "0.3.4",
    "kapsl-monitor": "0.3.0",
    "kapsl-scheduler": "0.3.0",
    "kapsl-shm": "0.4.0",
    "kapsl-transport": "0.4.0",
}


def validate_packages(packages, *, require_checksums=False):
    if not isinstance(packages, list) or any(not isinstance(p, dict) for p in packages):
        raise ValueError("SDK verification requires a Cargo package list.")
    failures = []
    for name, version in EXPECTED_VERSIONS.items():
        candidates = [p for p in packages if p.get("name") == name]
        if len(candidates) != 1:
            failures.append(
                f"{name}: expected one crates.io {version}, found {len(candidates)}"
            )
            continue
        package = candidates[0]
        if package.get("version") != version:
            failures.append(
                f"{name}: expected {version}, resolved {package.get('version')}"
            )
        if package.get("source") != CRATES_IO:
            # Do not print arbitrary source URLs, which can contain credentials.
            failures.append(
                f"{name}: must resolve from crates.io, not a path/git/other source"
            )
        if require_checksums and not re.fullmatch(
            r"[0-9a-f]{64}", str(package.get("checksum", ""))
        ):
            failures.append(
                f"{name}: missing or invalid registry checksum in Cargo.lock"
            )
    if failures:
        raise ValueError("Published SDK contract failed:\n- " + "\n- ".join(failures))


def verify_lockfile(path):
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    validate_packages(lock.get("package"), require_checksums=True)


def verify_metadata(path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError("SDK verification requires cargo metadata --format-version 1.")
    validate_packages(metadata.get("packages"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--lockfile", type=Path, help="Cheap host-only check; no Cargo or network"
    )
    source.add_argument(
        "--metadata", type=Path, help="Check cargo metadata --locked output"
    )
    args = parser.parse_args()
    try:
        if args.lockfile is not None:
            verify_lockfile(args.lockfile)
        else:
            verify_metadata(args.metadata)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    scope = "lockfile pins" if args.lockfile is not None else "resolved Cargo metadata"
    print(
        f"Published SDK {scope} verified: {json.dumps(EXPECTED_VERSIONS, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
