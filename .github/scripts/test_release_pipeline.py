#!/usr/bin/env python3
"""Host-only regression tests for fast preflight and parallel release builds."""

import base64
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from unittest import TestCase, main, mock


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


environment = load("preflight_release_environment")
build = load("release_build_inputs")


def workflow(name):
    return (WORKFLOWS / (name + ".yml")).read_text()


def job(source, name):
    match = re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:|\Z)", source)
    if not match:
        raise AssertionError(f"Job {name} is missing")
    return match[1]


class EnvironmentTests(TestCase):
    def test_sdk_ref_must_resolve_before_build_fanout(self):
        values = {"KAPSL_VLLM_SDK_REF": "a" * 40, "GH_TOKEN": "fixture-token"}
        with mock.patch.object(environment.subprocess, "run") as request:
            request.return_value.stdout = "a" * 40 + "\n"
            environment.validate_sdk(values)
            self.assertEqual(request.call_args.kwargs["timeout"], 30)
            request.return_value.stdout = "b" * 40
            with self.assertRaisesRegex(environment.PreflightError, "exact requested revision"):
                environment.validate_sdk(values)
            request.reset_mock()
            with self.assertRaises(environment.PreflightError):
                environment.validate_sdk(values | {"KAPSL_VLLM_SDK_REF": "main"})
            request.assert_not_called()

    def gpu_env(self):
        return {
            "GCP_GPU_PROJECT_ID": "fixture-project",
            "GCP_GPU_ZONE": "us-central1-a",
            "GCP_GPU_RUNNER_IMAGE": "projects/fixture-project/global/images/immutable-image",
            "GCP_WORKLOAD_IDENTITY_PROVIDER": "fixture-provider",
            "GCP_GPU_PROVISIONER_SERVICE_ACCOUNT": "fixture-service-account",
            "GPU_RUNNER_GITHUB_APP_ID": "123",
            "GPU_RUNNER_GITHUB_APP_PRIVATE_KEY": "fixture-private-key",
            "KAPSL_VLLM_SDK_REF": "a" * 40,
        }

    def test_valid_gpu_configuration_and_empty_optional_defaults(self):
        values = self.gpu_env()
        values.update(GCP_GPU_RUNNER_GROUP_ID="", GCP_GPU_EXTERNAL_IP="")
        environment.validate_gpu(values)

    def test_missing_secret_explains_inheritance_without_revealing_values(self):
        values = self.gpu_env()
        values["GPU_RUNNER_GITHUB_APP_PRIVATE_KEY"] = " "
        with self.assertRaisesRegex(environment.PreflightError, "secrets: inherit") as error:
            environment.validate_gpu(values)
        self.assertNotIn("fixture-private-key", str(error.exception))

    def test_bad_configuration_fails_before_any_external_actions(self):
        bad = {
            "GCP_GPU_PROJECT_ID": "",
            "GCP_GPU_RUNNER_IMAGE": "projects/fixture-project/global/images/family/latest",
            "GPU_RUNNER_GITHUB_APP_ID": "invalid-sensitive-value",
            "GCP_GPU_RUNNER_GROUP_ID": "0",
            "GCP_GPU_EXTERNAL_IP": "yes",
            "KAPSL_VLLM_SDK_REF": "develop",
        }
        for name, value in bad.items():
            with self.subTest(name=name):
                values = self.gpu_env() | {name: value}
                with self.assertRaises(environment.PreflightError) as error:
                    environment.validate_gpu(values)
                self.assertNotIn("invalid-sensitive-value", str(error.exception))

    def test_signing_requires_matching_trust_and_publication_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "fixture.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)],
                           check=True, capture_output=True)
            public = subprocess.check_output(["openssl", "pkey", "-in", str(key),
                                               "-pubout", "-outform", "DER"])[-32:]
            values = {
                "KAPSL_BACKEND_SIGNING_KEY_B64": base64.b64encode(key.read_bytes()).decode(),
                "KAPSL_BACKEND_PUBLIC_KEYS": base64.b64encode(public).decode(),
                "R2_ACCESS_KEY_ID": "fixture-access-id",
                "R2_SECRET_ACCESS_KEY": "fixture-access-secret",
            }
            environment.validate_signing(values)
            for name in values:
                with self.subTest(missing=name), self.assertRaises(environment.PreflightError):
                    environment.validate_signing(values | {name: ""})
            with self.assertRaisesRegex(environment.PreflightError, "does not match"):
                environment.validate_signing(values | {
                    "KAPSL_BACKEND_PUBLIC_KEYS": base64.b64encode(bytes(32)).decode()
                })
            secret = "malformed-sensitive-material"
            result = subprocess.run(
                [sys.executable, str(Path(environment.__file__)), "--signing"],
                env=os.environ | values | {"KAPSL_BACKEND_SIGNING_KEY_B64": secret},
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(secret, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)


class BuildPlanTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.manifest = Path(temporary.name) / "Cargo.toml"
        self.manifest.write_text('[package]\nversion = "0.2.7"\n')
        self.values = {
            "RELEASE_VERSION": "0.2.7", "RELEASE_CHANNEL": "stable",
            "STABLE_RELEASE": "true", "CACHE_ONLY": "false",
            "KAPSL_VLLM_SDK_REF": "a" * 40,
            "GITHUB_REF": "refs/tags/v0.2.7", "GITHUB_REF_TYPE": "tag",
            "GITHUB_REF_NAME": "v0.2.7", "GITHUB_EVENT_NAME": "push",
        }

    def test_stable_parallelizes_engine_and_backend_builds_without_rebuilding_ort(self):
        version, matrix = build.plan(self.values, self.manifest)
        self.assertEqual(version, "0.2.7")
        self.assertEqual([item["component"] for item in matrix["include"]],
                         ["engine", "llama-cpu", "llama-cuda12", "vllm"])
        targets = [item["target"] for item in matrix["include"] if item["target"]]
        self.assertEqual(len(targets), len(set(targets)))

    def test_beta_includes_certified_cpu_but_never_legacy_accelerator_components(self):
        version, matrix = build.plan(self.values | {
            "STABLE_RELEASE": "false", "RELEASE_CHANNEL": "beta",
            "RELEASE_VERSION": "0.2.7-beta.20260907.abcdef12",
            "GITHUB_REF": "refs/heads/develop", "GITHUB_REF_TYPE": "branch",
        }, self.manifest)
        self.assertIn("beta", version)
        self.assertEqual(matrix["include"][-1]["component"], "ort-cpu")
        self.assertFalse(any("onnx" in item["component"] for item in matrix["include"]))

    def test_cache_warming_is_main_only_and_produces_no_vllm_or_ort_release_parts(self):
        values = self.values | {"CACHE_ONLY": "true", "STABLE_RELEASE": "false",
                                "GITHUB_REF": "refs/heads/main", "KAPSL_VLLM_SDK_REF": ""}
        self.assertEqual(len(build.plan(values, self.manifest)[1]["include"]), 3)
        with self.assertRaisesRegex(ValueError, "restricted to main"):
            build.plan(values | {"GITHUB_REF": "refs/heads/develop"}, self.manifest)

    def test_invalid_stable_context_version_and_sdk_are_rejected(self):
        for name, value in (
            ("GITHUB_EVENT_NAME", "workflow_dispatch"), ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_REF_TYPE", "branch"), ("GITHUB_REF_NAME", "v0.2.7-beta.1"),
            ("RELEASE_VERSION", "0.2.8"), ("RELEASE_CHANNEL", "beta"),
            ("RELEASE_VERSION", "0.2.7/../../invalid"), ("KAPSL_VLLM_SDK_REF", "main"),
            ("CACHE_ONLY", "yes"), ("STABLE_RELEASE", "yes"),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                build.plan(self.values | {name: value}, self.manifest)


class AssemblyTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "parts"
        self.root.mkdir()
        self.output = self.root.parent / "dist"
        self.commit = "a" * 40
        self.version = "0.2.7"
        for item in build.components(True):
            directory = self.root / f"build-cuda-{item['component']}-{self.commit}"
            directory.mkdir()
            archive = directory / f"kapsl-{item['component']}-{self.version}-linux-x86_64.tar.gz"
            archive.write_bytes(item["component"].encode())
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            archive.with_name(archive.name + ".sha256").write_text(f"{digest}  dist/{archive.name}\n")
            if item["component"] != "engine":
                archive.with_name(archive.name + ".manifest.json").write_text("{}\n")

    def assemble(self):
        build.assemble(self.root, self.output, self.version, self.commit, True)

    def test_complete_component_set_is_moved_without_duplicate_disk_usage(self):
        self.assemble()
        self.assertEqual(len(list(self.output.iterdir())), 11)
        self.assertTrue(all(not list(directory.iterdir()) for directory in self.root.iterdir()))

    def test_missing_component_fails_without_moving_existing_outputs(self):
        next(self.root.iterdir()).rename(self.root.parent / "missing-component")
        with self.assertRaisesRegex(ValueError, "component set"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_wrong_commit_cannot_be_substituted(self):
        with self.assertRaisesRegex(ValueError, "component set"):
            build.assemble(self.root, self.output, self.version, "b" * 40, True)

    def test_checksum_tampering_is_rejected_before_any_move(self):
        next(self.root.glob("*/*.tar.gz")).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_missing_backend_manifest_is_rejected(self):
        next(self.root.glob("*/*.manifest.json")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing backend manifest"):
            self.assemble()

    def test_duplicate_output_and_symlinks_are_rejected(self):
        directories = list(self.root.iterdir())
        original = next(directories[0].glob("*.tar.gz"))
        duplicate = directories[1] / original.name
        duplicate.write_bytes(original.read_bytes())
        with self.assertRaises(ValueError):
            self.assemble()
        duplicate.unlink()
        target = self.root.parent / "outside"
        original.replace(target)
        original.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "non-regular"):
            self.assemble()


class NativeCacheTests(TestCase):
    def test_cpu_and_cuda_build_targets_survive_packaging_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            for relative in (".github/scripts", ".github/licenses", "kapsl-runtime/include"):
                (repo / relative).mkdir(parents=True, exist_ok=True)
            for relative in (".github/scripts/package-linux-llama-cpp-backend-packs.sh",
                             ".github/licenses/LLAMA-CPP-LICENSE", "LICENSE", "NOTICE",
                             "kapsl-runtime/include/kapsl_llama_cpp_backend.h", "kapsl-runtime/Cargo.toml"):
                shutil.copyfile(ROOT / relative, repo / relative)
            binary = root / "bin"
            binary.mkdir()
            for name in ("file", "ldd", "nm", "patchelf", "sha256sum", "tar"):
                stub = binary / name
                stub.write_text("#!/bin/sh\nexit 0\n")
                stub.chmod(0o755)
            cargo = binary / "cargo"
            cargo.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
target = pathlib.Path(sys.argv[sys.argv.index("--target-dir") + 1])
(target / "release").mkdir(parents=True, exist_ok=True)
(target / "release/libkapsl_backend_llama_cpp.so").write_bytes(b"fixture")
with open(os.environ["CARGO_FIXTURE_LOG"], "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "pic": os.environ.get("CMAKE_POSITION_INDEPENDENT_CODE")}) + "\\n")
''')
            cargo.chmod(0o755)
            scratch = root / "scratch"
            scratch.mkdir()
            cache = root / "cache"
            log = root / "cargo.jsonl"
            values = os.environ | {
                "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64", "RUNNER_TEMP": str(scratch),
                "KAPSL_VERSION": "0.2.7", "KAPSL_LLAMA_BUILD_ONLY": "true",
                "KAPSL_LLAMA_TARGET_ROOT": str(cache), "CARGO_FIXTURE_LOG": str(log),
                "KAPSL_LLAMA_CPU_LIBRARY": "", "KAPSL_LLAMA_CUDA_LIBRARY": "",
            }
            for profile in ("cpu", "cuda12"):
                result = subprocess.run(["bash", ".github/scripts/package-linux-llama-cpp-backend-packs.sh"],
                                        cwd=repo, env=values | {"KAPSL_LLAMA_PACK_PROFILE": profile},
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((cache / profile / "release/libkapsl_backend_llama_cpp.so").is_file())
                self.assertEqual(list(scratch.iterdir()), [])
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertIn("--locked", call["args"])
                self.assertEqual(call["pic"], "ON")
            self.assertIn("cuda12-shared-pool", calls[1]["args"])
            self.assertEqual(list((repo / "dist").iterdir()), [])


class WorkflowTests(TestCase):
    def test_all_expensive_stable_builds_wait_for_authentication(self):
        source = workflow("release-runtime-installers")
        for name in ("build-runtime-installers", "build-cuda-runtime", "stable-release-cpu-conformance"):
            block = job(source, name)
            self.assertIn("needs: [prepare-version, stable-release-infrastructure-preflight]", block)
            if name.startswith("build-"):
                self.assertIn("!cancelled()", block)
                self.assertIn("needs.prepare-version.result == 'success'", block)
                self.assertIn("needs.stable-release-infrastructure-preflight.result == 'success'", block)
        for name in ("stable-release-infrastructure-preflight", "stable-release-gpu-conformance"):
            self.assertIn("secrets: inherit", job(source, name))
        self.assertIn("INFRASTRUCTURE_RESULT", job(source, "release-conformance-gate"))
        self.assertIn("if: always()", job(source, "release-conformance-gate"))

    def test_live_preflight_is_main_only_and_cannot_provision(self):
        source = workflow("release-preflight")
        for name in ("live-infrastructure-preflight", "live-signing-preflight"):
            block = job(source, name)
            self.assertIn("github.ref == 'refs/heads/main'", block)
            self.assertIn("github.event_name == 'push' || github.event_name == 'workflow_dispatch'", block)
        self.assertNotIn("secrets.", job(source, "ort-release-preflight"))
        infrastructure = workflow("release-infrastructure-preflight")
        self.assertIn("environment: gcp-gpu-conformance", infrastructure)
        self.assertIn("permission-administration: write", infrastructure)
        self.assertIn("actions/runners?per_page=1", infrastructure)
        for forbidden in ("generate-jitconfig", " provision", "gcloud compute instances create",
                          "gcp_ephemeral_gpu_runner.py", "runs-on: [self-hosted"):
            self.assertNotIn(forbidden, infrastructure)

    def test_parallel_builds_are_host_only_and_preserve_retryable_artifacts(self):
        source = workflow("build-linux-accelerators")
        block = job(source, "components")
        self.assertIn("needs: prepare", block)
        self.assertIn("matrix: ${{ fromJSON(needs.prepare.outputs.matrix) }}", block)
        self.assertIn("fail-fast: false", block)
        self.assertNotIn("cargo clean", source)
        self.assertNotIn("package-linux-onnx-backend-packs.sh", source)
        self.assertNotIn("gpu-device-pool-integration.yml", source)
        self.assertNotIn("id-token: write", source)
        self.assertIn("build-cuda-${{ matrix.component }}-${{ github.sha }}", block)
        assemble_job = job(source, "assemble")
        self.assertIn("needs: [prepare, components]", assemble_job)
        self.assertIn("pattern: build-cuda-*-${{ github.sha }}", assemble_job)
        self.assertIn("merge-multiple: false", assemble_job)
        self.assertIn("name: runtime-cuda-linux-x86_64", assemble_job)
        for caller in ("release-runtime-installers", "beta-runtime-installers"):
            caller_source = workflow(caller)
            self.assertIn("uses: ./.github/workflows/build-linux-accelerators.yml", job(caller_source, "build-cuda-runtime"))
            self.assertNotIn("cache_only:", job(caller_source, "build-cuda-runtime"))
            self.assertIn("preflight_release_environment.py --signing --sdk", job(caller_source, "prepare-version"))
            self.assertEqual(caller_source.count("pattern: runtime-*"),
                             caller_source.count("name: Download installer artifacts"))

    def test_reusable_build_callers_grant_the_requested_permissions(self):
        # GitHub rejects the whole workflow before jobs start if a reusable
        # workflow asks for a permission omitted by its caller. actionlint's
        # syntax checks alone do not catch this cross-workflow contract.
        required = dict(re.findall(
            r"^  ([\w-]+): (read|write|none)$",
            workflow("build-linux-accelerators").split("\njobs:", 1)[0], re.MULTILINE,
        ))
        self.assertEqual(required, {"contents": "read", "actions": "read"})
        levels = {"none": 0, "read": 1, "write": 2}
        for caller, name in (
            ("beta-runtime-installers", "build-cuda-runtime"),
            ("release-runtime-installers", "build-cuda-runtime"),
            ("release-build-cache", "warm"),
        ):
            source = workflow(caller)
            block = job(source, name)
            self.assertIn("uses: ./.github/workflows/build-linux-accelerators.yml", block)
            if "    permissions:\n" in block:
                grants = dict(re.findall(r"^      ([\w-]+): (read|write|none)$", block, re.MULTILINE))
            else:
                grants = dict(re.findall(r"^  ([\w-]+): (read|write|none)$",
                                         source.split("\njobs:", 1)[0], re.MULTILINE))
            for scope, permission in required.items():
                with self.subTest(caller=caller, scope=scope):
                    self.assertGreaterEqual(levels[grants.get(scope, "none")], levels[permission])

    def test_cache_warming_does_not_publish_or_qualify(self):
        block = job(workflow("release-build-cache"), "warm")
        self.assertIn("github.ref == 'refs/heads/main'", block)
        self.assertIn("cache_only: true", block)
        self.assertNotIn("KAPSL_BACKEND_SIGNING_KEY", block)
        cache = (ROOT / ".github/actions/cache-rust-build/action.yml").read_text()
        self.assertIn("default: kapsl-runtime -> target", cache)
        self.assertIn("save-if: ${{ github.event_name != 'pull_request' || inputs.save-pr-cache == 'true' }}", cache)
        self.assertIn("cache-on-failure: ${{ github.event_name == 'pull_request' && inputs.save-pr-cache == 'true' }}", cache)
        self.assertRegex(cache, r'save-pr-cache:\n    required: false\n    default: "false"')
        self.assertNotIn("PRIVATE_KEY", cache)

    def test_ci_only_changes_do_not_automatically_build_betas(self):
        source = workflow("beta-runtime-installers")
        paths = source.split("    paths:\n", 1)[1].split("\npermissions:", 1)[0]
        patterns = re.findall(r'^      - "([^"]+)"', paths, re.MULTILINE)
        for path in ("kapsl-runtime/crates/kapsl-cli/src/main.rs", "rust-toolchain.toml",
                     "installers/install.sh", ".github/scripts/package-linux-cuda-runtime.sh"):
            self.assertTrue(any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns), path)
        for path in (".github/workflows/gpu-device-pool-integration.yml",
                     ".github/scripts/preflight_release_environment.py", "docs/architecture.md"):
            self.assertFalse(any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns), path)
        self.assertIn("workflow_dispatch:", source)
        self.assertIn("if: github.ref == 'refs/heads/develop'", source)


class CpuSmokeWorkflowTests(TestCase):
    def setUp(self):
        self.source = workflow("ort-cpu-conformance")
        trigger = self.source.split("  pull_request:\n", 1)[1].split("  workflow_call:", 1)[0]
        self.patterns = re.findall(r'^      - "([^"]+)"', trigger, re.MULTILINE)

    def matches(self, path):
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in self.patterns)

    def test_runtime_and_real_pack_inputs_still_run_correctness_smoke(self):
        for path in (
            "rust-toolchain.toml", ".cargo/config.toml", "kapsl-runtime/Cargo.lock",
            "kapsl-runtime/Cargo.toml", "kapsl-runtime/crates/kapsl-cli/Cargo.toml",
            "kapsl-runtime/crates/kapsl-cli/src/backend/native.rs",
            "kapsl-runtime/crates/kapsl-cli/src/backend/resolver.rs",
            "kapsl-runtime/crates/kapsl-cli/src/runtime/model/lifecycle.rs",
            "kapsl-runtime/crates/kapsl-backends/src/onnx.rs",
            ".github/ort-integration.lock", ".github/ort-cpu-parity.lock.json",
            ".github/licenses/ONNX-RUNTIME-LICENSE",
            ".github/scripts/certify-ort-cpu-parity.sh",
            ".github/scripts/generate-backend-index.py",
            ".github/scripts/package-linux-ort-cpu-backend.sh",
            ".github/scripts/verify-ort-integration-checkout.sh",
        ):
            with self.subTest(path=path):
                self.assertTrue(self.matches(path))

    def test_ci_authentication_and_test_only_changes_do_not_compile_ort(self):
        for path in (
            ".github/workflows/ort-cpu-conformance.yml",
            ".github/workflows/release-runtime-installers.yml",
            ".github/workflows/gpu-device-pool-integration.yml",
            ".github/workflows/release-infrastructure-preflight.yml",
            ".github/actions/cache-rust-build/action.yml",
            ".github/scripts/preflight_release_environment.py",
            ".github/scripts/validate_stable_gpu_release.py",
            ".github/scripts/test-generate-backend-index.sh",
            ".github/scripts/test-onnx-backend-release-contract.sh",
            ".github/scripts/test-package-linux-ort-cpu-backend.sh",
            ".github/scripts/test-verify-ort-integration-checkout.sh",
            ".github/scripts/test_release_pipeline.py", "docs/architecture.md",
        ):
            with self.subTest(path=path):
                self.assertFalse(self.matches(path))

    def test_removed_triggers_remain_covered_by_lightweight_checks(self):
        installer = workflow("installer-smoke")
        triggers, steps = installer.split("\njobs:\n", 1)
        for path in (
            ".github/scripts/test-generate-backend-index.sh",
            ".github/scripts/test-onnx-backend-release-contract.sh",
            ".github/scripts/test-package-linux-ort-cpu-backend.sh",
            ".github/scripts/test-verify-ort-integration-checkout.sh",
            ".github/scripts/validate_stable_gpu_release.py",
            ".github/workflows/ort-cpu-conformance.yml",
        ):
            with self.subTest(path=path):
                self.assertIn(f'      - "{path}"', triggers)
                if path.endswith(".sh"):
                    self.assertIn(f"run: {path}", steps)
        self.assertIn("python3 .github/scripts/test_release_pipeline.py", steps)
        self.assertIn("python3 .github/scripts/test_gcp_ephemeral_gpu_runner.py", steps)
        self.assertNotIn("run: .github/scripts/certify-ort-cpu-parity.sh", steps)

    def test_cache_covers_actual_engine_and_adapter_targets_but_no_packs_or_keys(self):
        block = job(self.source, "ort-cpu-release-handoff")
        cache = block.split("      - name: Cache engine", 1)[1].split("      - name:", 1)[0]
        self.assertIn("uses: ./.github/actions/cache-rust-build", cache)
        self.assertIn("ort-cpu-ubuntu22.04-${{ steps.release-inputs.outputs.integrations_ref }}", cache)
        self.assertIn('save-pr-cache: "true"', cache)
        mappings = re.findall(r'^            ([\w-]+) -> (.+)$', cache, re.MULTILINE)
        targets = {root: (ROOT / root / target).resolve() for root, target in mappings}
        self.assertEqual(targets["kapsl-runtime"], ROOT / "kapsl-runtime/target")
        build_root = re.search(r'KAPSL_ORT_PACK_BUILD_DIR: \$\{\{ github.workspace \}\}/(.+)', block)
        self.assertIsNotNone(build_root)
        self.assertEqual(targets["kapsl-integrations-ort"], ROOT / build_root[1] / "target")
        self.assertFalse(targets["kapsl-integrations-ort"].is_relative_to(ROOT / "kapsl-integrations-ort"))
        self.assertLess(block.index("Cache engine"), block.index("Build and validate release handoff"))
        self.assertLess(block.index("Install exact ORT packaging toolchain"), block.index("Cache engine"))
        for forbidden in ("dist/", ".pem", "evidence", "cache-all-crates: true"):
            self.assertNotIn(forbidden, "\n".join(line for line in cache.splitlines() if not line.lstrip().startswith("#")))
        # All other callers retain the default read-only behavior on PRs.
        for path in WORKFLOWS.glob("*.yml"):
            if path.name != "ort-cpu-conformance.yml":
                self.assertNotIn("save-pr-cache:", path.read_text(), path.name)

    def test_modes_and_unconditional_evidence_remain_intact(self):
        self.assertIn("name: ORT CPU Smoke and Release Parity", self.source)
        self.assertIn("correctness smoke (PR)", self.source)
        self.assertIn('if [[ "$GITHUB_EVENT_NAME" == "pull_request" ]]; then\n            mode=smoke', self.source)
        self.assertIn('elif [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]]; then\n            mode=performance', self.source)
        self.assertIn("python3 .github/scripts/validate_stable_gpu_release.py", self.source)
        self.assertIn("if: always()", self.source)
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", self.source)
        self.assertNotIn("self-hosted", self.source)
        self.assertNotIn("secrets.", self.source)
        self.assertNotIn("gpu-device-pool-integration.yml", self.source)


if __name__ == "__main__":
    main()
