#!/usr/bin/env python3
"""Create, observe, and remove a repository-scoped GitHub JIT runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


API_ROOT = "https://api.github.com"
API_VERSION = "2026-03-10"
TOKEN_ENV = "GITHUB_RUNNER_APP_TOKEN"
MAX_RESPONSE_BYTES = 1024 * 1024
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_RUNNER_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?$")


class JitRunnerError(RuntimeError):
    """A fail-closed JIT runner operation failure."""


def _validate(pattern: re.Pattern[str], name: str, value: str) -> str:
    if not pattern.fullmatch(value):
        raise JitRunnerError(f"invalid {name}: {value!r}")
    return value


def _positive_integer(name: str, value: str | int) -> int:
    text = str(value)
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        raise JitRunnerError(f"{name} must be a positive decimal integer")
    return int(text)


def _token() -> str:
    token = os.environ.get(TOKEN_ENV, "")
    if not token or "\n" in token or "\r" in token:
        raise JitRunnerError(f"{TOKEN_ENV} must contain the GitHub App token")
    return token


def _repository_path(owner: str, repository: str) -> str:
    _validate(_OWNER_RE, "owner", owner)
    _validate(_REPO_RE, "repository", repository)
    return f"/repos/{owner}/{repository}/actions/runners"


def _request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    allow_not_found: bool = False,
) -> tuple[int, dict[str, Any] | None]:
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        API_ROOT + path,
        method=method,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {_token()}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "kapsl-ephemeral-gpu-runner",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise JitRunnerError("GitHub API response exceeded the size limit")
            status = response.status
    except urllib.error.HTTPError as error:
        if allow_not_found and error.code == 404:
            return 404, None
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise JitRunnerError(
            f"GitHub API {method} {path} failed ({error.code}): {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise JitRunnerError(f"GitHub API {method} {path} failed: {error.reason}") from error
    if not body:
        return status, None
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise JitRunnerError("GitHub API returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise JitRunnerError("GitHub API response was not an object")
    return status, decoded


def _write_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise JitRunnerError("JIT configuration permissions are not mode 0600")


def create_runner(
    owner: str,
    repository: str,
    *,
    runner_group_id: str | int,
    name: str,
    labels: Sequence[str],
    config_file: Path,
) -> dict[str, Any]:
    base = _repository_path(owner, repository)
    group = _positive_integer("runner_group_id", runner_group_id)
    _validate(_RUNNER_NAME_RE, "runner name", name)
    if not 1 <= len(labels) <= 100:
        raise JitRunnerError("labels must contain between 1 and 100 values")
    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        _validate(_LABEL_RE, "runner label", label)
        key = label.lower()
        if key in seen:
            raise JitRunnerError(f"duplicate runner label: {label}")
        seen.add(key)
        normalized.append(label)
    status, response = _request(
        "POST",
        base + "/generate-jitconfig",
        payload={
            "name": name,
            "runner_group_id": group,
            "labels": normalized,
            "work_folder": "_work",
        },
    )
    if status != 201 or response is None:
        raise JitRunnerError(f"unexpected JIT creation status: {status}")
    runner = response.get("runner")
    encoded = response.get("encoded_jit_config")
    if not isinstance(runner, dict) or not isinstance(encoded, str) or not encoded:
        raise JitRunnerError("GitHub API omitted the JIT runner contract")
    runner_id = _positive_integer("runner id", runner.get("id", ""))
    if runner.get("name") != name:
        raise JitRunnerError("GitHub API returned a different runner name")
    returned_labels = runner.get("labels")
    if not isinstance(returned_labels, list):
        raise JitRunnerError("GitHub API omitted the runner labels")
    returned_names = {
        value.get("name", "").lower()
        for value in returned_labels
        if isinstance(value, dict) and isinstance(value.get("name"), str)
    }
    if not seen.issubset(returned_names):
        raise JitRunnerError("GitHub API omitted a requested runner label")
    if len(encoded.encode("utf-8")) > 128 * 1024:
        raise JitRunnerError("JIT configuration exceeded the metadata limit")
    _write_secret(config_file, encoded)
    return {"runner_id": runner_id, "runner_name": name}


def get_runner(owner: str, repository: str, runner_id: str | int) -> dict[str, Any] | None:
    base = _repository_path(owner, repository)
    identifier = _positive_integer("runner_id", runner_id)
    status, response = _request("GET", f"{base}/{identifier}", allow_not_found=True)
    if status == 404:
        return None
    if status != 200 or response is None:
        raise JitRunnerError(f"unexpected runner status response: {status}")
    return response


def wait_for_runner(
    owner: str,
    repository: str,
    runner_id: str | int,
    *,
    expected_name: str,
    timeout_seconds: int,
    interval_seconds: int,
) -> dict[str, Any]:
    _validate(_RUNNER_NAME_RE, "runner name", expected_name)
    if not 30 <= timeout_seconds <= 1800:
        raise JitRunnerError("timeout_seconds must be between 30 and 1800")
    if not 1 <= interval_seconds <= 60:
        raise JitRunnerError("interval_seconds must be between 1 and 60")
    deadline = time.monotonic() + timeout_seconds
    while True:
        runner = get_runner(owner, repository, runner_id)
        if runner is None:
            raise JitRunnerError("JIT runner disappeared before becoming ready")
        if runner.get("name") != expected_name:
            raise JitRunnerError("GitHub API returned a different runner identity")
        if runner.get("status") == "online":
            if runner.get("busy") is not False:
                raise JitRunnerError("JIT runner was claimed before conformance dispatch")
            return runner
        if time.monotonic() >= deadline:
            raise JitRunnerError("timed out waiting for the JIT runner to become online")
        time.sleep(interval_seconds)


def delete_runner(owner: str, repository: str, runner_id: str | int) -> bool:
    base = _repository_path(owner, repository)
    identifier = _positive_integer("runner_id", runner_id)
    status, _ = _request("DELETE", f"{base}/{identifier}", allow_not_found=True)
    if status == 404:
        return False
    if status != 204:
        raise JitRunnerError(f"unexpected runner deletion status: {status}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--owner", required=True)
    create.add_argument("--repository", required=True)
    create.add_argument("--runner-group-id", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--label", action="append", required=True)
    create.add_argument("--config-file", type=Path, required=True)

    wait = subparsers.add_parser("wait")
    wait.add_argument("--owner", required=True)
    wait.add_argument("--repository", required=True)
    wait.add_argument("--runner-id", required=True)
    wait.add_argument("--expected-name", required=True)
    wait.add_argument("--timeout-seconds", type=int, default=900)
    wait.add_argument("--interval-seconds", type=int, default=10)

    delete = subparsers.add_parser("delete")
    delete.add_argument("--owner", required=True)
    delete.add_argument("--repository", required=True)
    delete.add_argument("--runner-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            result = create_runner(
                args.owner,
                args.repository,
                runner_group_id=args.runner_group_id,
                name=args.name,
                labels=args.label,
                config_file=args.config_file,
            )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "wait":
            runner = wait_for_runner(
                args.owner,
                args.repository,
                args.runner_id,
                expected_name=args.expected_name,
                timeout_seconds=args.timeout_seconds,
                interval_seconds=args.interval_seconds,
            )
            print(json.dumps({"runner_id": runner.get("id"), "status": "online"}))
        elif args.command == "delete":
            deleted = delete_runner(args.owner, args.repository, args.runner_id)
            print("deleted" if deleted else "absent")
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (JitRunnerError, OSError) as error:
        print(f"GitHub JIT runner error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
