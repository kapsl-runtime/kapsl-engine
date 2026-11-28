#!/usr/bin/env python3
"""Validate a run-scoped waiver of the automated stable GPU conformance job.

The stable release workflow certifies its CUDA candidate on an ephemeral GCP
L4 runner. When that job cannot pass for a reason outside the candidate (for
example no L4 capacity), a maintainer may qualify the candidate on other
hardware and record it in `.github/stable-gpu-gate-waiver.json` on the default
branch, then re-run the failed jobs of the same release run.

The waiver names one stable tag, one workflow run and the digest of that run's
CUDA artifact, so it can only publish the exact build that was qualified, and
it cannot apply to any other release. Anything malformed or mismatched fails.
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
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
EVIDENCE_PREFIX = "https://github.com/kapsl-runtime/kapsl-engine/"
DEFAULT_WAIVER = Path(".github/stable-gpu-gate-waiver.json")
WAIVER_KEYS = {
    "schema_version",
    "release_tag",
    "run_id",
    "candidate",
    "reason",
    "manual_qualification",
}
CANDIDATE_KEYS = {"artifact", "digest"}
QUALIFICATION_KEYS = {"performed_on", "hardware", "evidence"}


class WaiverError(ValueError):
    """The waiver is malformed or does not authorize this publication."""


def _text(container: dict, key: str, where: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WaiverError(f"{where}.{key} must be a non-empty string")
    return value


def _object(container: dict, key: str, keys: set[str], where: str) -> dict:
    value = container.get(key)
    if not isinstance(value, dict) or set(value) != keys:
        raise WaiverError(f"{where}.{key} must contain exactly {sorted(keys)}")
    return value


def load_waiver(path: Path) -> dict | None:
    """Return the validated waiver, or None when no waiver file exists."""
    if not path.exists():
        return None
    try:
        waiver = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WaiverError(f"{path} is unreadable or not JSON: {error}") from error
    if not isinstance(waiver, dict) or set(waiver) != WAIVER_KEYS:
        raise WaiverError(f"{path} must contain exactly {sorted(WAIVER_KEYS)}")
    if waiver["schema_version"] != 1:
        raise WaiverError("waiver.schema_version must be 1")
    if STABLE_TAG_RE.fullmatch(_text(waiver, "release_tag", "waiver")) is None:
        raise WaiverError("waiver.release_tag must be a stable vMAJOR.MINOR.PATCH tag")
    run_id = waiver["run_id"]
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise WaiverError("waiver.run_id must be a positive integer")
    _text(waiver, "reason", "waiver")

    candidate = _object(waiver, "candidate", CANDIDATE_KEYS, "waiver")
    _text(candidate, "artifact", "candidate")
    if DIGEST_RE.fullmatch(_text(candidate, "digest", "candidate")) is None:
        raise WaiverError("candidate.digest must be sha256:<64 lowercase hex>")

    qualification = _object(
        waiver, "manual_qualification", QUALIFICATION_KEYS, "waiver"
    )
    try:
        datetime.date.fromisoformat(
            _text(qualification, "performed_on", "manual_qualification")
        )
    except ValueError as error:
        raise WaiverError(
            "manual_qualification.performed_on must be a YYYY-MM-DD date"
        ) from error
    _text(qualification, "hardware", "manual_qualification")
    if not _text(qualification, "evidence", "manual_qualification").startswith(
        EVIDENCE_PREFIX
    ):
        raise WaiverError(
            f"manual_qualification.evidence must be a link under {EVIDENCE_PREFIX}"
        )
    return waiver


def authorize(
    waiver_path: Path,
    *,
    release_tag: str,
    run_id: int,
    artifact: str,
    artifact_digests: list[str],
) -> dict:
    """Return the waiver if it authorizes publishing this run's candidate."""
    waiver = load_waiver(waiver_path)
    if waiver is None:
        raise WaiverError(f"no waiver at {waiver_path}")
    if waiver["release_tag"] != release_tag:
        raise WaiverError(
            f"the waiver is for {waiver['release_tag']}, not {release_tag}"
        )
    if waiver["run_id"] != run_id:
        raise WaiverError(f"the waiver is for run {waiver['run_id']}, not {run_id}")
    candidate = waiver["candidate"]
    if candidate["artifact"] != artifact:
        raise WaiverError(
            f"the waiver qualifies artifact {candidate['artifact']}, not {artifact}"
        )
    if len(artifact_digests) != 1:
        raise WaiverError(
            f"run {run_id} must have exactly one unexpired {artifact} artifact, "
            f"found {len(artifact_digests)}"
        )
    if artifact_digests[0] != candidate["digest"]:
        raise WaiverError(
            f"the qualified digest {candidate['digest']} differs from this run's "
            f"{artifact} artifact {artifact_digests[0]}"
        )
    return waiver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waiver", type=Path, default=DEFAULT_WAIVER)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--artifact-digests",
        required=True,
        help="space-separated digests of the run's unexpired artifacts with that name",
    )
    arguments = parser.parse_args()
    try:
        waiver = authorize(
            arguments.waiver,
            release_tag=arguments.release_tag,
            run_id=arguments.run_id,
            artifact=arguments.artifact,
            artifact_digests=arguments.artifact_digests.split(),
        )
    except WaiverError as error:
        print(f"stable GPU gate waiver refused: {error}", file=sys.stderr)
        return 1
    qualification = waiver["manual_qualification"]
    print(
        f"stable GPU gate waived for {waiver['release_tag']} run {waiver['run_id']}: "
        f"{waiver['reason']} Manual qualification on {qualification['performed_on']} "
        f"({qualification['hardware']}): {qualification['evidence']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
