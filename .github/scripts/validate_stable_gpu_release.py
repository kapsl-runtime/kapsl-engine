#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


STABLE_TAG_RE = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
SDK_REF_RE = re.compile(r"^[0-9a-f]{40}$")


class AuthorizationError(ValueError):
    """The workflow context is not an official stable release."""


def runtime_version(manifest: Path) -> str:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r'version\s*=\s*"([^"]+)"', line.strip())
        if match is not None:
            return match.group(1)
    raise AuthorizationError(f"runtime version is missing from {manifest}")


def authorize(
    *,
    event_name: str,
    ref_type: str,
    ref_name: str,
    release_tag: str,
    manifest: Path,
    sdk_ref: str | None = None,
) -> None:
    if event_name != "push":
        raise AuthorizationError("real GPU certification requires a tag push")
    if ref_type != "tag":
        raise AuthorizationError("real GPU certification requires a tag ref")
    if ref_name != release_tag:
        raise AuthorizationError("release tag does not match the triggering ref")
    if STABLE_TAG_RE.fullmatch(release_tag) is None:
        raise AuthorizationError(
            "real GPU certification requires a stable vMAJOR.MINOR.PATCH tag"
        )

    expected_version = release_tag.removeprefix("v")
    actual_version = runtime_version(manifest)
    if actual_version != expected_version:
        raise AuthorizationError(
            f"release tag version {expected_version} does not match runtime {actual_version}"
        )

    if sdk_ref is not None and SDK_REF_RE.fullmatch(sdk_ref) is None:
        raise AuthorizationError(
            "stable release SDK ref must be an exact lowercase 40-hex commit"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authorize a real GPU job only for an official stable release tag."
    )
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref-type", required=True)
    parser.add_argument("--ref-name", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sdk-ref")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        authorize(
            event_name=args.event_name,
            ref_type=args.ref_type,
            ref_name=args.ref_name,
            release_tag=args.release_tag,
            manifest=args.manifest,
            sdk_ref=args.sdk_ref,
        )
    except AuthorizationError as error:
        raise SystemExit(f"stable GPU release authorization failed: {error}")
    print(f"authorized stable GPU release: {args.release_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
