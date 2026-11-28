#!/usr/bin/env python3
"""Plan host-only release builds and validate their same-run artifact handoff."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?")
# The exact kapsl-integrations commit the managed-vLLM connector is built from.
VLLM_INTEGRATION_LOCK = Path(".github/vllm-integration.lock")


def flag(environ, name):
    value = environ.get(name, "false")
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def integration_ref(lock):
    """Return the one exact commit an integration lock file pins."""
    try:
        text = lock.read_text()
    except OSError:
        raise ValueError(f"{lock.as_posix()} is missing or unreadable") from None
    if not re.fullmatch(r"[0-9a-f]{40}\n?", text):
        raise ValueError(f"{lock.as_posix()} must contain one exact lowercase 40-hex commit")
    return text.strip()


def components(stable, cache_only=False):
    result = [
        {"component": "engine", "target": "target", "profile": ""},
        {"component": "llama-cpu", "target": "target-llama/cpu", "profile": "cpu"},
        {"component": "llama-cuda12", "target": "target-llama/cuda12", "profile": "cuda12"},
    ]
    if not cache_only:
        result.append({"component": "vllm", "target": "", "profile": ""})
        if not stable:
            # Managed SGLang ships in betas until a stable release has
            # qualified it on a GPU through the release gate.
            result.append({"component": "sglang", "target": "", "profile": ""})
            result.append({"component": "ort-cpu", "target": "", "profile": ""})
    return result


def plan(environ, manifest, vllm_lock):
    stable = flag(environ, "STABLE_RELEASE")
    cache_only = flag(environ, "CACHE_ONLY")
    if environ.get("RELEASE_CHANNEL") not in ("stable", "beta"):
        raise ValueError("RELEASE_CHANNEL must be stable or beta")
    version = environ.get("RELEASE_VERSION", "")
    cargo_match = re.search(r'^version\s*=\s*"([^"]+)"', manifest.read_text(), re.MULTILINE)
    if not cargo_match:
        raise ValueError("Runtime Cargo version is missing")
    if cache_only:
        if environ.get("GITHUB_REF") != "refs/heads/main" or stable:
            raise ValueError("Cache-only builds are restricted to main and cannot qualify a release")
        version = cargo_match[1]
    else:
        # Every build that is not cache-only packages managed vLLM, so a
        # malformed pin fails here rather than after the component fan-out.
        integration_ref(vllm_lock)
    if not VERSION.fullmatch(version):
        raise ValueError("Release version is invalid")
    if stable and (
        environ.get("GITHUB_EVENT_NAME") != "push"
        or environ.get("GITHUB_REF_TYPE") != "tag"
        or environ.get("GITHUB_REF_NAME") != f"v{version}"
        or version != cargo_match[1]
        or "-" in version
        or environ["RELEASE_CHANNEL"] != "stable"
    ):
        raise ValueError("Stable build requires the exact official tag and matching Cargo version")
    return version, {"include": components(stable, cache_only)}


def assemble(root, output, version, commit, stable):
    if not VERSION.fullmatch(version) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid release version or source commit")
    expected = {f"build-cuda-{item['component']}-{commit}" for item in components(stable)}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("Downloaded component set does not match the release plan")
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise ValueError("Assembly destination must be empty")
    files = {}
    for directory in sorted(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Component must be a real directory")
        entries = list(directory.iterdir())
        if not entries:
            raise ValueError("A release component is empty")
        for source in entries:
            if source.is_symlink() or not source.is_file():
                raise ValueError("Component contains a non-regular file")
            if source.name in files:
                raise ValueError(f"Duplicate component output: {source.name}")
            if f"-{version}-" not in source.name or not source.name.endswith(
                (".tar.gz", ".tar.gz.sha256", ".tar.gz.manifest.json")
            ):
                raise ValueError(f"Unexpected or wrong-version component output: {source.name}")
            files[source.name] = source
        for archive in (path for path in entries if path.name.endswith(".tar.gz")):
            if not directory.name.startswith("build-cuda-engine-"):
                manifest = archive.with_name(archive.name + ".manifest.json")
                if not manifest.is_file() or manifest.is_symlink():
                    raise ValueError(f"Missing backend manifest: {archive.name}")
            checksum = archive.with_name(archive.name + ".sha256")
            if not checksum.is_file() or checksum.is_symlink():
                raise ValueError(f"Missing archive checksum: {archive.name}")
            match = re.fullmatch(r"([0-9a-f]{64})\s+\*?(?:dist/)?([^/\r\n]+)\s*", checksum.read_text())
            if not match or match[2] != archive.name:
                raise ValueError(f"Invalid archive checksum record: {archive.name}")
            digest = hashlib.sha256()
            with archive.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != match[1]:
                raise ValueError(f"Archive checksum mismatch: {archive.name}")
        if not any(path.name.endswith(".tar.gz") for path in entries):
            raise ValueError("Component contains no archives")
    # Validate everything before moving anything. Moving on the same filesystem
    # avoids doubling the multi-GB payload on disk during assembly.
    output.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        source.replace(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assemble", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    try:
        if args.assemble:
            assemble(args.assemble, args.output, os.environ["RELEASE_VERSION"],
                     os.environ["GITHUB_SHA"], flag(os.environ, "STABLE_RELEASE"))
        else:
            version, matrix = plan(os.environ, Path("kapsl-runtime/crates/kapsl-cli/Cargo.toml"),
                                   VLLM_INTEGRATION_LOCK)
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
                output.write(f"version={version}\nmatrix={json.dumps(matrix)}\n")
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
