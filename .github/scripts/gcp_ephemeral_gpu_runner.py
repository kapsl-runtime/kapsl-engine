#!/usr/bin/env python3
"""Provision and retire tightly-scoped ephemeral GCE GPU runners.

The helper deliberately owns all resource-name validation and destructive GCE
operations used by the GPU conformance workflow.  JIT runner credentials are
handled by github_jit_runner.py and are only passed here as a mode-0600 file.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


RUNNER_CONTRACT = "v1"
MANAGED_BY = "kapsl-gha"
PURPOSE = "vllm-conformance"
DEFAULT_MACHINE_TYPE = "g2-standard-12"
DEFAULT_BOOT_DISK_GB = 200
DEFAULT_MAX_RUN_SECONDS = 18_000
MIN_MAX_RUN_SECONDS = 900
MAX_MAX_RUN_SECONDS = 21_600
MAX_METADATA_FILE_BYTES = 128 * 1024

RUNNER_VERSION = "2.337.0"
RUNNER_SHA256 = "70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613"
CONFORMANCE_IMAGE = (
    "nvidia/cuda:13.0.2-devel-ubuntu24.04@"
    "sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
)

_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_ZONE_RE = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?-[a-z]$")
_INSTANCE_RE = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MACHINE_RE = re.compile(r"^[a-z][a-z0-9-]{1,61}[a-z0-9]$")
_IMAGE_RE = re.compile(
    r"^(?:https://www\.googleapis\.com/compute/v1/)?"
    r"projects/[a-z][a-z0-9-]{4,28}[a-z0-9]/global/images/"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_SUBNET_RE = re.compile(
    r"^(?:https://www\.googleapis\.com/compute/v1/)?"
    r"projects/[a-z][a-z0-9-]{4,28}[a-z0-9]/regions/"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?/subnetworks/"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ConfigurationError(ValueError):
    """A fail-closed configuration validation failure."""


def _require_match(name: str, value: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value):
        raise ConfigurationError(f"invalid {name}: {value!r}")
    return value


def _positive_decimal(name: str, value: str | int) -> int:
    text = str(value)
    if not text.isascii() or not text.isdecimal():
        raise ConfigurationError(f"{name} must be a positive decimal integer")
    number = int(text)
    if number <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return number


def parse_bool(name: str, value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ConfigurationError(f"{name} must be exactly true or false")


def repository_label(repository: str) -> str:
    _require_match("repository", repository, _REPOSITORY_RE)
    label = re.sub(r"[^a-z0-9_-]+", "-", repository.lower()).strip("-_")
    if not label or len(label) > 63:
        raise ConfigurationError("repository does not fit a GCE label value")
    return label


@dataclass(frozen=True)
class RunnerIdentity:
    instance_name: str
    runner_name: str
    runner_label: str
    expires_at: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "instance_name": self.instance_name,
            "runner_name": self.runner_name,
            "runner_label": self.runner_label,
            "expires_at": self.expires_at,
        }


def make_identity(
    run_id: str | int,
    run_attempt: str | int,
    *,
    now_epoch: int | None = None,
    max_run_seconds: int = DEFAULT_MAX_RUN_SECONDS,
) -> RunnerIdentity:
    run = _positive_decimal("run_id", run_id)
    attempt = _positive_decimal("run_attempt", run_attempt)
    if not MIN_MAX_RUN_SECONDS <= max_run_seconds <= MAX_MAX_RUN_SECONDS:
        raise ConfigurationError(
            f"max_run_seconds must be between {MIN_MAX_RUN_SECONDS} and "
            f"{MAX_MAX_RUN_SECONDS}"
        )
    now = int(time.time()) if now_epoch is None else int(now_epoch)
    if now <= 0:
        raise ConfigurationError("now_epoch must be positive")
    suffix = f"{run}-{attempt}"
    instance = f"kapsl-vllm-{suffix}"
    label = f"kapsl-vllm-{suffix}"
    _require_match("instance_name", instance, _INSTANCE_RE)
    if len(label) > 63:
        raise ConfigurationError("runner label exceeds 63 characters")
    return RunnerIdentity(instance, instance, label, now + max_run_seconds)


def validate_metadata_file(name: str, path: Path, *, sensitive: bool) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ConfigurationError(f"{name} is not a regular file")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_METADATA_FILE_BYTES:
        raise ConfigurationError(
            f"{name} must contain 1..{MAX_METADATA_FILE_BYTES} bytes"
        )
    if sensitive:
        mode = stat.S_IMODE(resolved.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ConfigurationError(f"{name} must not be accessible by group or other")
    return resolved


@dataclass(frozen=True)
class ProvisionConfig:
    project: str
    zone: str
    instance_name: str
    image: str
    repository: str
    run_id: int
    run_attempt: int
    expires_at: int
    provisioning_model: str
    startup_script: Path
    jit_config: Path
    subnet: str | None = None
    external_ip: bool = True
    machine_type: str = DEFAULT_MACHINE_TYPE
    boot_disk_gb: int = DEFAULT_BOOT_DISK_GB
    max_run_seconds: int = DEFAULT_MAX_RUN_SECONDS

    def validated(self) -> "ProvisionConfig":
        _require_match("project", self.project, _PROJECT_RE)
        _require_match("zone", self.zone, _ZONE_RE)
        _require_match("instance_name", self.instance_name, _INSTANCE_RE)
        _require_match("machine_type", self.machine_type, _MACHINE_RE)
        if self.machine_type != DEFAULT_MACHINE_TYPE:
            raise ConfigurationError(
                f"machine_type must be the certified {DEFAULT_MACHINE_TYPE} shape"
            )
        _require_match("image", self.image, _IMAGE_RE)
        repository_label(self.repository)
        run_id = _positive_decimal("run_id", self.run_id)
        run_attempt = _positive_decimal("run_attempt", self.run_attempt)
        expected_instance = f"kapsl-vllm-{run_id}-{run_attempt}"
        if self.instance_name != expected_instance:
            raise ConfigurationError("instance_name does not match the GitHub run identity")
        _positive_decimal("expires_at", self.expires_at)
        now = int(time.time())
        if self.expires_at <= now:
            raise ConfigurationError("expires_at must be in the future")
        model = self.provisioning_model.upper()
        if model not in {"SPOT", "STANDARD"}:
            raise ConfigurationError("provisioning_model must be SPOT or STANDARD")
        if self.subnet:
            _require_match("subnet", self.subnet, _SUBNET_RE)
        if not 100 <= self.boot_disk_gb <= 500:
            raise ConfigurationError("boot_disk_gb must be between 100 and 500")
        if not MIN_MAX_RUN_SECONDS <= self.max_run_seconds <= MAX_MAX_RUN_SECONDS:
            raise ConfigurationError(
                f"max_run_seconds must be between {MIN_MAX_RUN_SECONDS} and "
                f"{MAX_MAX_RUN_SECONDS}"
            )
        if self.expires_at > now + self.max_run_seconds:
            raise ConfigurationError("expires_at exceeds the VM maximum runtime")
        validate_metadata_file("startup_script", self.startup_script, sensitive=False)
        validate_metadata_file("jit_config", self.jit_config, sensitive=True)
        return self


def build_create_command(config: ProvisionConfig) -> list[str]:
    config.validated()
    labels = {
        "managed-by": MANAGED_BY,
        "purpose": PURPOSE,
        "repository": repository_label(config.repository),
        "github-run-id": str(config.run_id),
        "github-run-attempt": str(config.run_attempt),
        "expires-at": str(config.expires_at),
    }
    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        config.instance_name,
        "--quiet",
        f"--project={config.project}",
        f"--zone={config.zone}",
        f"--machine-type={config.machine_type}",
        f"--image={config.image}",
        f"--boot-disk-size={config.boot_disk_gb}GB",
        "--boot-disk-type=pd-balanced",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--provisioning-model={config.provisioning_model.upper()}",
        "--instance-termination-action=DELETE",
        f"--max-run-duration={config.max_run_seconds}s",
        "--no-service-account",
        "--shielded-vtpm",
        "--shielded-integrity-monitoring",
        "--network-tier=PREMIUM",
        "--labels=" + ",".join(f"{key}={value}" for key, value in labels.items()),
        "--metadata="
        f"block-project-ssh-keys=true,kapsl-runner-contract={RUNNER_CONTRACT}",
        "--metadata-from-file="
        f"startup-script={config.startup_script.resolve()},"
        f"runner-jit-config={config.jit_config.resolve()}",
    ]
    if config.subnet:
        command.append(f"--subnet={config.subnet}")
    else:
        command.append("--network=default")
    if not config.external_ip:
        command.append("--no-address")
    return command


def _resource_basename(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"GCE instance has an invalid {name}")
    return value.rstrip("/").rsplit("/", 1)[-1]


def _normalize_google_resource(value: str) -> str:
    prefix = "https://www.googleapis.com/compute/v1/"
    return value.removeprefix(prefix).rstrip("/")


def _duration_seconds(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("seconds")
    elif isinstance(value, str) and value.endswith("s"):
        value = value[:-1]
    return _positive_decimal("GCE maxRunDuration", value)


def verify_instance_contract(document: Any, config: ProvisionConfig) -> None:
    config.validated()
    if not isinstance(document, dict):
        raise ConfigurationError("GCE instance contract must be a JSON object")
    if document.get("name") != config.instance_name:
        raise ConfigurationError("GCE instance name does not match the request")
    if _resource_basename(document.get("zone"), "zone") != config.zone:
        raise ConfigurationError("GCE instance zone does not match the request")
    if (
        _resource_basename(document.get("machineType"), "machine type")
        != config.machine_type
    ):
        raise ConfigurationError("GCE machine type does not match the certified shape")

    scheduling = document.get("scheduling")
    if not isinstance(scheduling, dict):
        raise ConfigurationError("GCE instance omitted scheduling")
    expected_scheduling = {
        "automaticRestart": False,
        "instanceTerminationAction": "DELETE",
        "onHostMaintenance": "TERMINATE",
        "provisioningModel": config.provisioning_model.upper(),
    }
    for key, expected in expected_scheduling.items():
        if scheduling.get(key) != expected:
            raise ConfigurationError(f"GCE scheduling.{key} did not match {expected!r}")
    if _duration_seconds(scheduling.get("maxRunDuration")) != config.max_run_seconds:
        raise ConfigurationError("GCE maximum runtime does not match the request")

    disks = document.get("disks")
    if not isinstance(disks, list):
        raise ConfigurationError("GCE instance omitted disks")
    boot_disks = [disk for disk in disks if isinstance(disk, dict) and disk.get("boot")]
    if len(boot_disks) != 1 or boot_disks[0].get("autoDelete") is not True:
        raise ConfigurationError("GCE boot disk is not uniquely auto-deleted")
    service_accounts = document.get("serviceAccounts", [])
    if service_accounts not in (None, []):
        raise ConfigurationError("GCE runner must not have a service account")
    if document.get("deletionProtection", False) is not False:
        raise ConfigurationError("GCE runner unexpectedly has deletion protection")

    shielded = document.get("shieldedInstanceConfig")
    if not isinstance(shielded, dict):
        raise ConfigurationError("GCE instance omitted Shielded VM configuration")
    if shielded.get("enableVtpm") is not True:
        raise ConfigurationError("GCE vTPM is not enabled")
    if shielded.get("enableIntegrityMonitoring") is not True:
        raise ConfigurationError("GCE integrity monitoring is not enabled")

    expected_labels = {
        "managed-by": MANAGED_BY,
        "purpose": PURPOSE,
        "repository": repository_label(config.repository),
        "github-run-id": str(config.run_id),
        "github-run-attempt": str(config.run_attempt),
        "expires-at": str(config.expires_at),
    }
    labels = document.get("labels")
    if not isinstance(labels, dict) or any(
        labels.get(key) != value for key, value in expected_labels.items()
    ):
        raise ConfigurationError("GCE ownership or expiry labels do not match")

    interfaces = document.get("networkInterfaces")
    if not isinstance(interfaces, list) or len(interfaces) != 1:
        raise ConfigurationError("GCE runner must have exactly one network interface")
    interface = interfaces[0]
    if not isinstance(interface, dict):
        raise ConfigurationError("GCE runner network interface is invalid")
    access_configs = interface.get("accessConfigs", [])
    if not isinstance(access_configs, list):
        raise ConfigurationError("GCE access configuration is invalid")
    if config.external_ip and len(access_configs) != 1:
        raise ConfigurationError("GCE runner did not receive one external address")
    if not config.external_ip and access_configs:
        raise ConfigurationError("GCE private runner unexpectedly has an external address")
    if config.subnet:
        actual_subnet = interface.get("subnetwork")
        if not isinstance(actual_subnet, str) or _normalize_google_resource(
            actual_subnet
        ) != _normalize_google_resource(config.subnet):
            raise ConfigurationError("GCE runner subnet does not match the request")


def run_command(
    command: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def delete_instance(project: str, zone: str, instance_name: str) -> bool:
    _require_match("project", project, _PROJECT_RE)
    _require_match("zone", zone, _ZONE_RE)
    _require_match("instance_name", instance_name, _INSTANCE_RE)
    listed = run_command(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--quiet",
            f"--project={project}",
            f"--filter=name={instance_name}",
            "--format=json(name,zone)",
        ],
        capture_output=True,
    )
    try:
        instances = json.loads(listed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("gcloud returned invalid deletion preflight JSON") from error
    if not isinstance(instances, list):
        raise RuntimeError("gcloud deletion preflight was not a JSON array")
    if not instances:
        return False
    if len(instances) != 1 or not isinstance(instances[0], dict):
        raise RuntimeError("GCE deletion preflight returned ambiguous instances")
    found = instances[0]
    if found.get("name") != instance_name or _zone_basename(found.get("zone")) != zone:
        raise RuntimeError("GCE deletion preflight returned an unexpected identity")
    run_command(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            instance_name,
            "--quiet",
            f"--project={project}",
            f"--zone={zone}",
        ]
    )
    return True


def _zone_basename(value: Any) -> str:
    return _resource_basename(value, "zone")


def sweep_expired(
    project: str,
    repository: str,
    *,
    now_epoch: int | None = None,
) -> list[tuple[str, str]]:
    _require_match("project", project, _PROJECT_RE)
    expected_repository = repository_label(repository)
    now = int(time.time()) if now_epoch is None else int(now_epoch)
    if now <= 0:
        raise ConfigurationError("now_epoch must be positive")
    result = run_command(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--quiet",
            f"--project={project}",
            f"--filter=labels.managed-by={MANAGED_BY} AND labels.purpose={PURPOSE}",
            "--format=json(name,zone,labels)",
        ],
        capture_output=True,
    )
    try:
        instances = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("gcloud returned invalid instance JSON") from error
    if not isinstance(instances, list):
        raise RuntimeError("gcloud instance list was not a JSON array")

    deleted: list[tuple[str, str]] = []
    for item in instances:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        labels = item.get("labels")
        if not isinstance(name, str) or not isinstance(labels, dict):
            continue
        if (
            labels.get("managed-by") != MANAGED_BY
            or labels.get("purpose") != PURPOSE
        ):
            continue
        if labels.get("repository") != expected_repository:
            continue
        expires_text = labels.get("expires-at")
        if (
            not isinstance(expires_text, str)
            or not expires_text.isascii()
            or not expires_text.isdecimal()
        ):
            continue
        if int(expires_text) > now:
            continue
        try:
            _require_match("instance_name", name, _INSTANCE_RE)
            zone = _zone_basename(item.get("zone"))
            _require_match("zone", zone, _ZONE_RE)
        except ConfigurationError:
            continue
        if delete_instance(project, zone, name):
            deleted.append((name, zone))
    return deleted


def startup_script() -> str:
    # JIT configuration is fetched from instance metadata at runtime and is
    # never interpolated into this generated script or emitted to its log.
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
set +x

readonly METADATA_URL='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
readonly RUNNER_VERSION='{RUNNER_VERSION}'
readonly RUNNER_SHA256='{RUNNER_SHA256}'
readonly RUNNER_ARCHIVE="actions-runner-linux-x64-${{RUNNER_VERSION}}.tar.gz"
readonly RUNNER_RELEASES='https://github.com/actions/runner/releases/download'
readonly RUNNER_URL="$RUNNER_RELEASES/v${{RUNNER_VERSION}}/${{RUNNER_ARCHIVE}}"
readonly CONFORMANCE_IMAGE='{CONFORMANCE_IMAGE}'

finish() {{
  status=$?
  trap - EXIT
  unset jit_config || true
  sync || true
  shutdown -h now || systemctl poweroff || true
  exit "$status"
}}
trap finish EXIT

metadata() {{
  curl --fail --silent --show-error \
    --header 'Metadata-Flavor: Google' \
    "$METADATA_URL/$1"
}}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git gnupg jq
if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends docker.io
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y --no-install-recommends nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker

for _ in $(seq 1 60); do
  if driver_version=$(nvidia-smi --query-gpu=driver_version \
      --format=csv,noheader 2>/dev/null | head -n1); then
    break
  fi
  sleep 5
done
test -n "${{driver_version:-}}"
driver_major=${{driver_version%%.*}}
[[ "$driver_major" =~ ^[0-9]+$ ]]
if (( driver_major < 580 )); then
  echo "NVIDIA driver 580 or newer is required; found $driver_version" >&2
  exit 1
fi

if ! id github-runner >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash github-runner
fi
usermod -aG docker github-runner
install -d -m 0755 -o github-runner -g github-runner /opt/actions-runner
runner_archive="/tmp/$RUNNER_ARCHIVE"
curl --fail --silent --show-error --location --retry 3 \
  --output "$runner_archive" "$RUNNER_URL"
printf '%s  %s\n' "$RUNNER_SHA256" "$runner_archive" | sha256sum --check -
tar --extract --gzip --file "$runner_archive" --directory /opt/actions-runner
rm -f "$runner_archive"
chown -R github-runner:github-runner /opt/actions-runner
/opt/actions-runner/bin/installdependencies.sh

docker pull "$CONFORMANCE_IMAGE"
docker run --rm --gpus all "$CONFORMANCE_IMAGE" nvidia-smi

jit_config=$(metadata runner-jit-config)
if [[ -z "$jit_config" || ${{#jit_config}} -gt 131072 ]]; then
  echo 'Missing or oversized JIT runner configuration.' >&2
  exit 1
fi

cd /opt/actions-runner
set +e
runuser -u github-runner -- ./run.sh --jitconfig "$jit_config"
runner_status=$?
set -e
unset jit_config
exit "$runner_status"
"""


def _provision_config_from_args(args: argparse.Namespace) -> ProvisionConfig:
    return ProvisionConfig(
        project=args.project,
        zone=args.zone,
        instance_name=args.instance_name,
        image=args.image,
        repository=args.repository,
        run_id=_positive_decimal("run_id", args.run_id),
        run_attempt=_positive_decimal("run_attempt", args.run_attempt),
        expires_at=_positive_decimal("expires_at", args.expires_at),
        provisioning_model=args.provisioning_model,
        startup_script=Path(args.startup_script),
        jit_config=Path(args.jit_config),
        subnet=args.subnet or None,
        external_ip=parse_bool("external_ip", args.external_ip),
        machine_type=args.machine_type,
        boot_disk_gb=int(args.boot_disk_gb),
        max_run_seconds=int(args.max_run_seconds),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity", help="derive bounded per-run names")
    identity.add_argument("--run-id", required=True)
    identity.add_argument("--run-attempt", required=True)
    identity.add_argument("--now-epoch", type=int)
    identity.add_argument("--max-run-seconds", type=int, default=DEFAULT_MAX_RUN_SECONDS)

    startup = subparsers.add_parser("startup-script", help="render the VM startup script")
    startup.add_argument("--output", type=Path)

    provision = subparsers.add_parser("provision", help="create one exact GCE instance")
    provision.add_argument("--project", required=True)
    provision.add_argument("--zone", required=True)
    provision.add_argument("--instance-name", required=True)
    provision.add_argument("--image", required=True)
    provision.add_argument("--repository", required=True)
    provision.add_argument("--run-id", required=True)
    provision.add_argument("--run-attempt", required=True)
    provision.add_argument("--expires-at", required=True)
    provision.add_argument("--provisioning-model", required=True)
    provision.add_argument("--startup-script", required=True)
    provision.add_argument("--jit-config", required=True)
    provision.add_argument("--subnet", default="")
    provision.add_argument("--external-ip", default="true")
    provision.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE)
    provision.add_argument("--boot-disk-gb", type=int, default=DEFAULT_BOOT_DISK_GB)
    provision.add_argument("--max-run-seconds", type=int, default=DEFAULT_MAX_RUN_SECONDS)
    provision.add_argument("--dry-run", action="store_true")

    verify = subparsers.add_parser("verify", help="verify the created VM contract")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--project", required=True)
    verify.add_argument("--zone", required=True)
    verify.add_argument("--instance-name", required=True)
    verify.add_argument("--image", required=True)
    verify.add_argument("--repository", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--run-attempt", required=True)
    verify.add_argument("--expires-at", required=True)
    verify.add_argument("--provisioning-model", required=True)
    verify.add_argument("--startup-script", required=True)
    verify.add_argument("--jit-config", required=True)
    verify.add_argument("--subnet", default="")
    verify.add_argument("--external-ip", default="true")
    verify.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE)
    verify.add_argument("--boot-disk-gb", type=int, default=DEFAULT_BOOT_DISK_GB)
    verify.add_argument("--max-run-seconds", type=int, default=DEFAULT_MAX_RUN_SECONDS)

    delete = subparsers.add_parser("delete", help="idempotently delete one exact instance")
    delete.add_argument("--project", required=True)
    delete.add_argument("--zone", required=True)
    delete.add_argument("--instance-name", required=True)

    sweep = subparsers.add_parser("sweep", help="delete expired managed instances")
    sweep.add_argument("--project", required=True)
    sweep.add_argument("--repository", required=True)
    sweep.add_argument("--now-epoch", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "identity":
            value = make_identity(
                args.run_id,
                args.run_attempt,
                now_epoch=args.now_epoch,
                max_run_seconds=args.max_run_seconds,
            )
            print(json.dumps(value.as_dict(), sort_keys=True))
        elif args.command == "startup-script":
            rendered = startup_script()
            if args.output:
                args.output.write_text(rendered, encoding="utf-8")
                args.output.chmod(0o700)
            else:
                sys.stdout.write(rendered)
        elif args.command == "provision":
            command = build_create_command(_provision_config_from_args(args))
            if args.dry_run:
                print(json.dumps(command))
            else:
                run_command(command)
        elif args.command == "verify":
            document = json.loads(args.input.read_text(encoding="utf-8"))
            verify_instance_contract(document, _provision_config_from_args(args))
            print("verified")
        elif args.command == "delete":
            deleted = delete_instance(args.project, args.zone, args.instance_name)
            print("deleted" if deleted else "absent")
        elif args.command == "sweep":
            deleted = sweep_expired(
                args.project,
                args.repository,
                now_epoch=args.now_epoch,
            )
            print(json.dumps([{"name": name, "zone": zone} for name, zone in deleted]))
        else:  # pragma: no cover - argparse makes this unreachable.
            raise AssertionError(args.command)
    except (
        ConfigurationError,
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        RuntimeError,
    ) as error:
        print(f"gcp GPU runner error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
