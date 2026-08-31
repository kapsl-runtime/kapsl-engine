#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import copy
import json
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("gcp_ephemeral_gpu_runner.py")
SPEC = importlib.util.spec_from_file_location("gcp_ephemeral_gpu_runner", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

POLICY_PATH = Path(__file__).with_name("validate_stable_gpu_release.py")
POLICY_SPEC = importlib.util.spec_from_file_location(
    "validate_stable_gpu_release", POLICY_PATH
)
assert POLICY_SPEC and POLICY_SPEC.loader
policy = importlib.util.module_from_spec(POLICY_SPEC)
sys.modules[POLICY_SPEC.name] = policy
POLICY_SPEC.loader.exec_module(policy)


class IdentityTests(unittest.TestCase):
    def test_identity_is_unique_bounded_and_expiring(self) -> None:
        identity = runner.make_identity("33077455449", "2", now_epoch=1_800_000_000)
        self.assertEqual(identity.instance_name, "kapsl-vllm-33077455449-2")
        self.assertEqual(identity.runner_label, identity.instance_name)
        self.assertEqual(identity.expires_at, 1_800_018_000)

    def test_identity_rejects_non_decimal_values(self) -> None:
        for value in ("", "0", "-1", "1.0", "1;rm"):
            with self.subTest(value=value), self.assertRaises(runner.ConfigurationError):
                runner.make_identity(value, "1", now_epoch=1_800_000_000)

    def test_repository_label_is_canonical(self) -> None:
        self.assertEqual(
            runner.repository_label("kapsl-runtime/kapsl-engine"),
            "kapsl-runtime-kapsl-engine",
        )

    def test_dispatcher_isolates_jit_runner_with_unique_label(self) -> None:
        dispatcher = (
            MODULE_PATH.parent.parent
            / "workflows"
            / "gpu-device-pool-integration.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('--label "${{ steps.identity.outputs.runner_label }}"', dispatcher)
        self.assertNotIn("--label gpu", dispatcher)


class StableGpuReleasePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.manifest = Path(self.temporary.name) / "Cargo.toml"
        self.manifest.write_text(
            '[package]\nname = "kapsl"\nversion = "0.2.4"\n',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def authorize(self, **overrides: object) -> None:
        values: dict[str, object] = {
            "event_name": "push",
            "ref_type": "tag",
            "ref_name": "v0.2.4",
            "release_tag": "v0.2.4",
            "manifest": self.manifest,
            "sdk_ref": "a" * 40,
        }
        values.update(overrides)
        policy.authorize(**values)

    def test_official_stable_release_is_authorized(self) -> None:
        self.authorize()

    def test_manual_branch_beta_and_prerelease_contexts_are_rejected(self) -> None:
        invalid = (
            {"event_name": "workflow_dispatch"},
            {"ref_type": "branch", "ref_name": "main", "release_tag": "main"},
            {
                "ref_name": "v0.2.4-beta.20260831.abcdef12",
                "release_tag": "v0.2.4-beta.20260831.abcdef12",
            },
            {"ref_name": "v0.2.4-rc.1", "release_tag": "v0.2.4-rc.1"},
            {"ref_name": "v0.2.5", "release_tag": "v0.2.5"},
            {"release_tag": "v0.2.5"},
            {"sdk_ref": "main"},
        )
        for overrides in invalid:
            with self.subTest(overrides), self.assertRaises(policy.AuthorizationError):
                self.authorize(**overrides)

    def test_every_real_gpu_entry_point_is_stable_release_only(self) -> None:
        workflows = MODULE_PATH.parent.parent / "workflows"
        dispatcher = (workflows / "gpu-device-pool-integration.yml").read_text(
            encoding="utf-8"
        )
        conformance = (workflows / "vllm-shared-pool-conformance.yml").read_text(
            encoding="utf-8"
        )
        lazy = (
            workflows / "lazy-llama-cpp-backend-pack-gpu-certification.yml"
        ).read_text(encoding="utf-8")
        stable_release = (workflows / "release-runtime-installers.yml").read_text(
            encoding="utf-8"
        )
        beta_release = (workflows / "beta-runtime-installers.yml").read_text(
            encoding="utf-8"
        )

        for workflow in (dispatcher, conformance, lazy):
            self.assertIn("workflow_call:", workflow)
            self.assertNotIn("workflow_dispatch:", workflow)
            self.assertIn("validate_stable_gpu_release.py", workflow)

        self.assertIn('cron: "17 */6 * * *"', dispatcher)
        self.assertIn("if: github.event_name == 'schedule'", dispatcher)
        self.assertIn("PROVISIONING_MODEL: STANDARD", dispatcher)
        self.assertNotIn("runs-on: [self-hosted, linux, x64, gpu]", dispatcher)

        self.assertIn("stable-release-gpu-conformance:", stable_release)
        self.assertIn("./.github/workflows/gpu-device-pool-integration.yml", stable_release)
        self.assertIn("is_stable_release:", stable_release)
        self.assertIn(
            "if: needs.prepare-version.outputs.is_stable_release == 'true'",
            stable_release,
        )
        self.assertIn("release_tag: ${{ github.ref_name }}", stable_release)
        self.assertIn("sdk_ref: ${{ vars.KAPSL_VLLM_SDK_REF }}", stable_release)
        self.assertNotIn("gpu-device-pool-integration.yml", beta_release)


class ProvisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.startup = root / "startup.sh"
        self.startup.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        self.jit = root / "jit"
        self.jit.write_text("super-secret-jit", encoding="utf-8")
        self.jit.chmod(0o600)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, **overrides: object):
        values = {
            "project": "kapsl-gpu-ci",
            "zone": "us-central1-a",
            "instance_name": "kapsl-vllm-123-1",
            "image": (
                "projects/deeplearning-platform-release/global/images/"
                "common-cu129-ubuntu-2404-nvidia-580-v20260801"
            ),
            "repository": "kapsl-runtime/kapsl-engine",
            "run_id": 123,
            "run_attempt": 1,
            "expires_at": int(time.time()) + 18_000,
            "provisioning_model": "SPOT",
            "startup_script": self.startup,
            "jit_config": self.jit,
        }
        values.update(overrides)
        return runner.ProvisionConfig(**values)

    def test_command_has_cost_security_and_lifecycle_bounds(self) -> None:
        command = runner.build_create_command(self.config())
        joined = "\n".join(command)
        self.assertIn("--machine-type=g2-standard-12", command)
        self.assertIn("--provisioning-model=SPOT", command)
        self.assertIn("--instance-termination-action=DELETE", command)
        self.assertIn("--max-run-duration=18000s", command)
        self.assertIn("--no-service-account", command)
        self.assertIn("--no-scopes", command)
        self.assertIn("--maintenance-policy=TERMINATE", command)
        self.assertIn("--network=default", command)
        self.assertIn("managed-by=kapsl-gha", joined)
        self.assertIn("expires-at=", joined)
        self.assertIn(str(self.jit.resolve()), joined)
        self.assertNotIn("super-secret-jit", joined)

    def test_private_subnet_disables_external_address(self) -> None:
        command = runner.build_create_command(
            self.config(
                subnet=(
                    "projects/kapsl-gpu-ci/regions/us-central1/subnetworks/"
                    "github-runners"
                ),
                external_ip=False,
                provisioning_model="STANDARD",
            )
        )
        self.assertIn("--no-address", command)
        self.assertIn("--provisioning-model=STANDARD", command)
        self.assertFalse(any(argument == "--network=default" for argument in command))

    def test_sensitive_config_permissions_fail_closed(self) -> None:
        self.jit.chmod(0o644)
        with self.assertRaises(runner.ConfigurationError):
            runner.build_create_command(self.config())

    def test_unpinned_image_and_injected_names_are_rejected(self) -> None:
        invalid = (
            {"image": "common-cu129-ubuntu-2404-nvidia-580"},
            {"instance_name": "runner;gcloud compute instances delete victim"},
            {"instance_name": "kapsl-vllm-999-1"},
            {"zone": "us-central1-a --quiet"},
            {"project": "UPPERCASE"},
            {"machine_type": "g2-standard-96"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(runner.ConfigurationError):
                runner.build_create_command(self.config(**overrides))

    def test_expiry_cannot_outlive_the_hard_vm_runtime(self) -> None:
        with self.assertRaises(runner.ConfigurationError):
            runner.build_create_command(
                self.config(expires_at=int(time.time()) + 19_000)
            )

    def test_startup_is_shell_valid_and_contains_pinned_contract(self) -> None:
        script = runner.startup_script()
        self.assertIn(runner.RUNNER_VERSION, script)
        self.assertIn(runner.RUNNER_SHA256, script)
        self.assertIn(runner.CONFORMANCE_IMAGE, script)
        self.assertIn("metadata runner-jit-config", script)
        self.assertIn("driver_major < 580", script)
        self.assertIn("./run.sh --jitconfig", script)
        self.assertIn("systemctl mask --now", script)
        self.assertIn("apt-daily-upgrade.timer", script)
        self.assertIn("apt-daily-upgrade.service", script)
        self.assertIn("completed with result:", script)
        self.assertIn("result: Succeeded", script)
        self.assertIn("if (( status == 0 )); then", script)
        self.assertIn('shutdown -h "+$FAILURE_SHUTDOWN_MINUTES"', script)
        self.assertIn("retaining the VM for diagnostics", script)
        self.assertNotIn("super-secret-jit", script)
        conformance_workflow = (
            MODULE_PATH.parent.parent
            / "workflows"
            / "vllm-shared-pool-conformance.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(runner.CONFORMANCE_IMAGE, conformance_workflow)
        self.assertIn("options: --runtime=nvidia --gpus all", conformance_workflow)
        self.assertIn("Verify GPU passthrough before build", conformance_workflow)
        self.assertIn("test -c /dev/nvidiactl", conformance_workflow)
        self.assertIn("nvidia-smi --list-gpus", conformance_workflow)
        self.assertIn("validate_model_package_backend.py", conformance_workflow)
        self.assertNotIn('backend-plan "$package"', conformance_workflow)
        self.assertNotIn('backend-plan "$fixed_package"', conformance_workflow)
        self.assertNotIn('backend-plan "$llama_package"', conformance_workflow)
        path = Path(self.temporary.name) / "rendered.sh"
        path.write_text(script, encoding="utf-8")
        subprocess.run(["bash", "-n", str(path)], check=True)

    def instance_document(self, config) -> dict[str, object]:
        return {
            "name": config.instance_name,
            "zone": (
                "https://www.googleapis.com/compute/v1/projects/kapsl-gpu-ci/"
                f"zones/{config.zone}"
            ),
            "machineType": (
                "https://www.googleapis.com/compute/v1/projects/kapsl-gpu-ci/"
                f"zones/{config.zone}/machineTypes/{config.machine_type}"
            ),
            "scheduling": {
                "automaticRestart": False,
                "instanceTerminationAction": "DELETE",
                "maxRunDuration": {"seconds": str(config.max_run_seconds)},
                "onHostMaintenance": "TERMINATE",
                "provisioningModel": config.provisioning_model,
            },
            "disks": [{"boot": True, "autoDelete": True}],
            "deletionProtection": False,
            "shieldedInstanceConfig": {
                "enableVtpm": True,
                "enableIntegrityMonitoring": True,
            },
            "labels": {
                "managed-by": runner.MANAGED_BY,
                "purpose": runner.PURPOSE,
                "repository": "kapsl-runtime-kapsl-engine",
                "github-run-id": str(config.run_id),
                "github-run-attempt": str(config.run_attempt),
                "expires-at": str(config.expires_at),
            },
            "networkInterfaces": [{"accessConfigs": [{"natIP": "203.0.113.2"}]}],
        }

    def test_created_instance_contract_is_verified(self) -> None:
        config = self.config()
        runner.verify_instance_contract(self.instance_document(config), config)

    def test_created_instance_contract_rejects_lifecycle_drift(self) -> None:
        config = self.config()
        mutations = {
            "maximum runtime": lambda value: value["scheduling"].update(
                {"maxRunDuration": "36000s"}
            ),
            "boot disk retention": lambda value: value["disks"][0].update(
                {"autoDelete": False}
            ),
            "service account": lambda value: value.update(
                {"serviceAccounts": [{"email": "default@example.invalid"}]}
            ),
            "external address": lambda value: value["networkInterfaces"][0].update(
                {"accessConfigs": []}
            ),
            "expiry label": lambda value: value["labels"].update(
                {"expires-at": "1"}
            ),
        }
        for name, mutate in mutations.items():
            document = copy.deepcopy(self.instance_document(config))
            mutate(document)
            with self.subTest(name=name), self.assertRaises(runner.ConfigurationError):
                runner.verify_instance_contract(document, config)

    def test_private_subnet_contract_normalizes_full_resource_url(self) -> None:
        subnet = (
            "projects/kapsl-gpu-ci/regions/us-central1/subnetworks/github-runners"
        )
        config = self.config(subnet=subnet, external_ip=False)
        document = self.instance_document(config)
        document["networkInterfaces"] = [
            {
                "accessConfigs": [],
                "subnetwork": "https://www.googleapis.com/compute/v1/" + subnet,
            }
        ]
        runner.verify_instance_contract(document, config)


class CleanupTests(unittest.TestCase):
    @mock.patch.object(runner, "run_command")
    def test_exact_delete_is_idempotent(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 0, "[]", "")
        self.assertFalse(
            runner.delete_instance("kapsl-gpu-ci", "us-central1-a", "kapsl-vllm-1-1")
        )
        self.assertEqual(run.call_count, 1)

    @mock.patch.object(runner, "run_command")
    def test_delete_rejects_wrong_zone_before_mutation(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            [],
            0,
            json.dumps([{"name": "kapsl-vllm-1-1", "zone": "us-east1-b"}]),
            "",
        )
        with self.assertRaises(RuntimeError):
            runner.delete_instance(
                "kapsl-gpu-ci", "us-central1-a", "kapsl-vllm-1-1"
            )
        self.assertEqual(run.call_count, 1)

    @mock.patch.object(runner, "delete_instance", return_value=True)
    @mock.patch.object(runner, "run_command")
    def test_sweeper_deletes_only_expired_matching_resources(
        self,
        run_command: mock.Mock,
        delete_instance: mock.Mock,
    ) -> None:
        instances = [
            {
                "name": "kapsl-vllm-1-1",
                "zone": "https://compute.googleapis.com/compute/v1/projects/p/zones/us-central1-a",
                "labels": {
                    "managed-by": runner.MANAGED_BY,
                    "purpose": runner.PURPOSE,
                    "repository": "kapsl-runtime-kapsl-engine",
                    "expires-at": "99",
                },
            },
            {
                "name": "kapsl-vllm-2-1",
                "zone": "us-central1-a",
                "labels": {
                    "managed-by": runner.MANAGED_BY,
                    "purpose": runner.PURPOSE,
                    "repository": "kapsl-runtime-kapsl-engine",
                    "expires-at": "101",
                },
            },
            {
                "name": "kapsl-vllm-3-1",
                "zone": "us-central1-a",
                "labels": {
                    "managed-by": runner.MANAGED_BY,
                    "purpose": runner.PURPOSE,
                    "repository": "another-owner-another-repo",
                    "expires-at": "1",
                },
            },
            {
                "name": "victim;bad",
                "zone": "us-central1-a",
                "labels": {
                    "managed-by": runner.MANAGED_BY,
                    "purpose": runner.PURPOSE,
                    "repository": "kapsl-runtime-kapsl-engine",
                    "expires-at": "1",
                },
            },
        ]
        run_command.return_value = subprocess.CompletedProcess(
            [], 0, json.dumps(instances), ""
        )
        deleted = runner.sweep_expired(
            "kapsl-gpu-ci",
            "kapsl-runtime/kapsl-engine",
            now_epoch=100,
        )
        self.assertEqual(deleted, [("kapsl-vllm-1-1", "us-central1-a")])
        delete_instance.assert_called_once_with(
            "kapsl-gpu-ci", "us-central1-a", "kapsl-vllm-1-1"
        )


if __name__ == "__main__":
    unittest.main()
