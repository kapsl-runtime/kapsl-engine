#!/usr/bin/env python3
"""Resolve a tag-scoped waiver of the automated stable GPU conformance job.

The stable release workflow certifies the CUDA candidate on an ephemeral GCP
L4 runner. When GCP cannot supply that machine, a maintainer may qualify the
candidate on other hardware and record it in `.github/stable-gpu-gate-waiver.json`.

A waiver names exactly one stable tag, so it expires by itself: every other
release tag still requires the automated job. A malformed waiver fails the
release instead of being ignored.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path


STABLE_TAG_RE = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
EVIDENCE_PREFIX = "https://github.com/kapsl-runtime/kapsl-engine/"
DEFAULT_WAIVER = Path(".github/stable-gpu-gate-waiver.json")
WAIVER_KEYS = {"schema_version", "release_tag", "reason", "manual_qualification"}
QUALIFICATION_KEYS = {"performed_on", "hardware", "candidate", "evidence"}


class WaiverError(ValueError):
    """The waiver file exists but cannot authorize anything."""


def _text(container: dict, key: str, where: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WaiverError(f"{where}.{key} must be a non-empty string")
    return value


def load_waiver(path: Path) -> dict | None:
    """Return the validated waiver, or None when no waiver file exists."""
    if not path.exists():
        return None
    try:
        waiver = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WaiverError(f"{path} is unreadable or not JSON: {error}") from error
    if not isinstance(waiver, dict) or set(waiver) != WAIVER_KEYS:
        raise WaiverError(f"{path} must contain exactly {sorted(WAIVER_KEYS)}")
    if waiver["schema_version"] != 1:
        raise WaiverError("waiver.schema_version must be 1")
    tag = _text(waiver, "release_tag", "waiver")
    if STABLE_TAG_RE.fullmatch(tag) is None:
        raise WaiverError("waiver.release_tag must be a stable vMAJOR.MINOR.PATCH tag")
    _text(waiver, "reason", "waiver")

    qualification = waiver["manual_qualification"]
    if not isinstance(qualification, dict) or set(qualification) != QUALIFICATION_KEYS:
        raise WaiverError(
            f"waiver.manual_qualification must contain exactly {sorted(QUALIFICATION_KEYS)}"
        )
    performed_on = _text(qualification, "performed_on", "manual_qualification")
    try:
        datetime.date.fromisoformat(performed_on)
    except ValueError as error:
        raise WaiverError(
            "manual_qualification.performed_on must be a YYYY-MM-DD date"
        ) from error
    _text(qualification, "hardware", "manual_qualification")
    _text(qualification, "candidate", "manual_qualification")
    evidence = _text(qualification, "evidence", "manual_qualification")
    if not evidence.startswith(EVIDENCE_PREFIX):
        raise WaiverError(
            f"manual_qualification.evidence must be a link under {EVIDENCE_PREFIX}"
        )
    return waiver


def is_waived(release_tag: str, path: Path) -> bool:
    """True only when a valid waiver names exactly this stable tag."""
    waiver = load_waiver(path)
    if waiver is None:
        return False
    if STABLE_TAG_RE.fullmatch(release_tag) is None:
        return False
    return waiver["release_tag"] == release_tag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--waiver", type=Path, default=DEFAULT_WAIVER)
    arguments = parser.parse_args()
    try:
        waived = is_waived(arguments.release_tag, arguments.waiver)
    except WaiverError as error:
        print(f"stable GPU gate waiver error: {error}", file=sys.stderr)
        return 1
    print("true" if waived else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
