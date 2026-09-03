#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"
: "${KAPSL_ORT_INTEGRATIONS_REF:?KAPSL_ORT_INTEGRATIONS_REF is required}"
: "${KAPSL_ORT_PARITY_HARNESS:?KAPSL_ORT_PARITY_HARNESS is required}"
: "${KAPSL_ORT_PARITY_HARNESS_PATH:?KAPSL_ORT_PARITY_HARNESS_PATH is required}"
: "${KAPSL_ORT_PARITY_HARNESS_SHA256:?KAPSL_ORT_PARITY_HARNESS_SHA256 is required}"
: "${KAPSL_ORT_PARITY_MODEL_REF:?KAPSL_ORT_PARITY_MODEL_REF is required}"
: "${KAPSL_ORT_PARITY_MODEL_PATH:?KAPSL_ORT_PARITY_MODEL_PATH is required}"
: "${KAPSL_ORT_PARITY_MODEL_SHA256:?KAPSL_ORT_PARITY_MODEL_SHA256 is required}"
: "${KAPSL_ORT_PARITY_PUBLIC_KEY:?KAPSL_ORT_PARITY_PUBLIC_KEY is required}"

conformance_mode="${KAPSL_ORT_CONFORMANCE_MODE:-performance}"
case "$conformance_mode" in
  smoke | performance) ;;
  *)
    echo "KAPSL_ORT_CONFORMANCE_MODE must be smoke or performance." >&2
    exit 1
    ;;
esac

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "ORT CPU parity is currently certified only on Linux x86_64." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
integrations_repo_dir="${KAPSL_ORT_INTEGRATIONS_REPO_DIR:-$repo_root/kapsl-integrations-ort}"
model_repo_dir="${KAPSL_ORT_PARITY_MODEL_REPO_DIR:-$repo_root/kapsl-sdk-ort-model}"
artifacts_dir="${KAPSL_ORT_PARITY_ARTIFACTS_DIR:-$repo_root/dist}"
evidence_dir="${KAPSL_ORT_PARITY_EVIDENCE_DIR:-$artifacts_dir/ort-cpu-$conformance_mode}"
index_path="$artifacts_dir/backend-index.json"
archive_name="kapsl-backend-onnx-cpu-${KAPSL_VERSION}-linux-x86_64.tar.gz"
archive_path="$artifacts_dir/$archive_name"
parity_harness="$KAPSL_ORT_PARITY_HARNESS"
source_model="$model_repo_dir/$KAPSL_ORT_PARITY_MODEL_PATH"
engine_binary="$repo_root/kapsl-runtime/target/release/kapsl"
work_root="$(mktemp -d)"
cleanup() {
  rm -rf "$work_root"
}
trap cleanup EXIT INT TERM

for command_name in cargo git python3 sha256sum; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required for ORT CPU parity certification." >&2
    exit 1
  fi
done
for required in "$index_path" "${index_path}.sig" "$archive_path" "$parity_harness" "$source_model"; do
  if [ ! -s "$required" ]; then
    echo "ORT CPU parity input is missing or empty: $required" >&2
    exit 1
  fi
done
if [ -e "$evidence_dir" ]; then
  echo "Refusing to reuse ORT CPU parity evidence directory: $evidence_dir" >&2
  exit 1
fi

verify_checkout() {
  checkout="$1"
  expected_ref="$2"
  label="$3"
  actual_ref="$(git -C "$checkout" rev-parse HEAD)"
  if [ "$actual_ref" != "$expected_ref" ]; then
    echo "$label checkout is $actual_ref, expected $expected_ref" >&2
    exit 1
  fi
  dirty="$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)"
  if [ -n "$dirty" ]; then
    echo "$label checkout is not clean:" >&2
    printf '%s\n' "$dirty" >&2
    exit 1
  fi
}

verify_checkout "$integrations_repo_dir" "$KAPSL_ORT_INTEGRATIONS_REF" "Integrations source"
verify_checkout "$model_repo_dir" "$KAPSL_ORT_PARITY_MODEL_REF" "Model source"
expected_parity_harness="$integrations_repo_dir/$KAPSL_ORT_PARITY_HARNESS_PATH"
if [ "$(realpath "$parity_harness")" != "$(realpath "$expected_parity_harness")" ]; then
  echo "ORT parity harness is not the locked integrations checkout entrypoint." >&2
  exit 1
fi
actual_harness_sha256="$(sha256sum "$parity_harness" | awk '{ print $1 }')"
if [ "$actual_harness_sha256" != "$KAPSL_ORT_PARITY_HARNESS_SHA256" ]; then
  echo "Pinned ORT parity harness digest is $actual_harness_sha256, expected $KAPSL_ORT_PARITY_HARNESS_SHA256" >&2
  exit 1
fi
actual_model_sha256="$(sha256sum "$source_model" | awk '{ print $1 }')"
if [ "$actual_model_sha256" != "$KAPSL_ORT_PARITY_MODEL_SHA256" ]; then
  echo "Pinned ORT parity model digest is $actual_model_sha256, expected $KAPSL_ORT_PARITY_MODEL_SHA256" >&2
  exit 1
fi

engine_ref="$(git -C "$repo_root" rev-parse HEAD)"
if [[ ! "$engine_ref" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Engine checkout did not resolve to an exact commit." >&2
  exit 1
fi

KAPSL_VERSION="$KAPSL_VERSION" cargo build \
  --manifest-path "$repo_root/kapsl-runtime/Cargo.toml" \
  --package kapsl \
  --release \
  --locked

model_input_dir="$work_root/model-input"
mkdir -p "$model_input_dir"
model_copy="$model_input_dir/identity_logits.onnx"
cp "$source_model" "$model_copy"
model_package="$work_root/identity-logits.aimod"
"$engine_binary" build "$model_copy" \
  --output "$model_package" \
  --project-name ort-cpu-parity \
  --format onnx \
  --model-type opaque \
  --task forward \
  --version 1.0.0
model_package_sha256="$(sha256sum "$model_package" | awk '{ print $1 }')"

bundle_path="$work_root/ort-cpu-parity.kapsl-bundle"
KAPSL_BACKEND_CACHE_DIR="$work_root/bundle-builder-backends" \
KAPSL_BACKEND_INDEX_PATH="$index_path" \
KAPSL_BACKEND_PUBLIC_KEYS="$KAPSL_ORT_PARITY_PUBLIC_KEY" \
KAPSL_LAZY_ONNX_PACKS=1 \
  "$engine_binary" bundle "$model_package" \
    --output "$bundle_path" \
    --target linux-x86_64-cpu \
    --backend-artifacts-dir "$artifacts_dir"

candidate_backend_cache="$work_root/candidate-backends"
candidate_bundle_cache="$work_root/candidate-bundles"
KAPSL_BACKEND_CACHE_DIR="$candidate_backend_cache" \
KAPSL_BUNDLE_CACHE_DIR="$candidate_bundle_cache" \
KAPSL_BACKEND_PUBLIC_KEYS="$KAPSL_ORT_PARITY_PUBLIC_KEY" \
KAPSL_GENERIC_NATIVE_PACKS=1 \
KAPSL_LAZY_ONNX_PACKS=1 \
  "$engine_binary" backend ensure "$bundle_path" --offline \
    >"$work_root/backend-ensure.log" 2>&1
KAPSL_BACKEND_CACHE_DIR="$candidate_backend_cache" \
KAPSL_BACKEND_PUBLIC_KEYS="$KAPSL_ORT_PARITY_PUBLIC_KEY" \
  "$engine_binary" backend list --json >"$work_root/backend-list.json"

python3 - "$work_root/backend-list.json" <<'PY'
import json
import pathlib
import sys

entries = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
matches = [
    entry
    for entry in entries
    if entry["backend"] == "onnx" and entry["profile"] == "cpu" and entry["valid"]
]
if len(matches) != 1:
    raise SystemExit("offline bundle did not install exactly one valid onnx/cpu pack")
PY

baseline_backend_cache="$work_root/baseline-backends"
shared_model_cache="$work_root/model-cache"
config_path="$work_root/ort-cpu-parity.json"
python3 - \
  "$config_path" \
  "$engine_binary" \
  "$repo_root" \
  "$model_package" \
  "$model_package_sha256" \
  "$engine_ref" \
  "$KAPSL_ORT_INTEGRATIONS_REF" \
  "$KAPSL_ORT_PARITY_PUBLIC_KEY" \
  "$baseline_backend_cache" \
  "$candidate_backend_cache" \
  "$shared_model_cache" \
  "$conformance_mode" <<'PY'
import json
import pathlib
import sys

(
    config_path,
    engine_binary,
    repo_root,
    model_package,
    model_package_sha256,
    engine_ref,
    integrations_ref,
    public_key,
    baseline_backend_cache,
    candidate_backend_cache,
    shared_model_cache,
    conformance_mode,
) = sys.argv[1:]

if conformance_mode == "smoke":
    workload = {
        "model_id": 0,
        "warmup_requests": 0,
        "requests_per_payload": 1,
        "trials": 1,
        "concurrency": [1],
        "timeout_seconds": 30,
        "readiness_timeout_seconds": 120,
        "cooldown_seconds": 0,
        "rss_sample_seconds": 0,
    }
    # The pinned integrations harness requires numeric performance fields.
    # PR smoke supplies non-restrictive sentinels, then enforces only route,
    # process, request-success, and tensor-correctness evidence below.
    gates = {
        "max_abs_error": 0.000001,
        "max_rel_error": 0.00001,
        "min_throughput_ratio": 0,
        "max_p95_latency_ratio": 1_000_000_000,
        "max_p99_latency_ratio": 1_000_000_000,
        "max_model_memory_ratio": 1_000_000_000,
        "max_model_memory_increase_bytes": 9_223_372_036_854_775_807,
        "max_peak_rss_ratio": 1_000_000_000,
        "max_peak_rss_increase_bytes": 9_223_372_036_854_775_807,
        "max_startup_ratio": 1_000_000_000,
        "require_zero_failures": True,
        "require_model_memory": False,
        "require_process_identity": True,
        "require_process_rss": False,
        "require_route_evidence": True,
        "require_startup_evidence": False,
    }
    sequence = ["baseline", "candidate", "candidate", "baseline"]
else:
    workload = {
        "model_id": 0,
        "warmup_requests": 40,
        "requests_per_payload": 1000,
        "trials": 3,
        "concurrency": [1, 4],
        "timeout_seconds": 30,
        "readiness_timeout_seconds": 120,
        "cooldown_seconds": 0.25,
        "rss_sample_seconds": 0.05,
    }
    gates = {
        "max_abs_error": 0.000001,
        "max_rel_error": 0.00001,
        "min_throughput_ratio": 0.8,
        "max_p95_latency_ratio": 1.25,
        "max_p99_latency_ratio": 1.35,
        "max_model_memory_ratio": 1.2,
        "max_model_memory_increase_bytes": 67108864,
        "max_peak_rss_ratio": 1.2,
        "max_peak_rss_increase_bytes": 268435456,
        "max_startup_ratio": 1.5,
        "require_zero_failures": True,
        "require_model_memory": False,
        "require_process_identity": True,
        "require_process_rss": True,
        "require_route_evidence": True,
        "require_startup_evidence": True,
    }
    # Four samples per route keep a single host-scheduler startup outlier from
    # controlling the median while preserving balanced ABBA ordering.
    sequence = ["baseline", "candidate", "candidate", "baseline"] * 2

command = [
    engine_binary,
    "run",
    "--model",
    "{model_path}",
    "--http-bind",
    "127.0.0.1",
    "--metrics-port",
    "19095",
    "--socket",
    "{output_dir}/kapsl.sock",
    "--offline",
]
common_env = {
    "KAPSL_BACKEND_PUBLIC_KEYS": public_key,
    "KAPSL_MODEL_CACHE_DIR": shared_model_cache,
    "RUST_LOG": "warn,kapsl=debug,kapsl_backends=info",
}
config = {
    "schema_version": 1,
    "suite_id": f"ort-cpu-forward-{conformance_mode}",
    "task_profile": "forward",
    "identity": {
        "engine_commit": engine_ref,
        "integrations_commit": integrations_ref,
        "model_path": model_package,
        "model_sha256": model_package_sha256,
    },
    "payloads": [
        {
            "id": "identity-logits-f32",
            "request": {
                "input": {
                    "shape": [1, 3],
                    "dtype": "float32",
                    "data": [0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64],
                }
            },
        }
    ],
    "workload": workload,
    "gates": gates,
    "sequence": sequence,
    "allowed_variant_env_differences": [
        "KAPSL_BACKEND_CACHE_DIR",
        "KAPSL_GENERIC_NATIVE_PACKS",
        "KAPSL_LAZY_ONNX_PACKS",
    ],
    "baseline": {
        "base_url": "http://127.0.0.1:19095",
        "command": command,
        "cwd": repo_root,
        "env": {
            **common_env,
            "KAPSL_BACKEND_CACHE_DIR": baseline_backend_cache,
            "KAPSL_GENERIC_NATIVE_PACKS": "0",
            "KAPSL_LAZY_ONNX_PACKS": "0",
        },
        "required_log_markers": [
            "Using embedded ORT rollback for model `ort-cpu-parity`",
            "Activating embedded ORT rollback route for model `ort-cpu-parity`",
        ],
        "forbidden_log_markers": [
            "Activated signed native backend pack onnx/cpu",
            "Selected signed native backend route onnx/cpu",
            "Activating signed backend route onnx/cpu",
        ],
    },
    "candidate": {
        "base_url": "http://127.0.0.1:19095",
        "command": command,
        "cwd": repo_root,
        "env": {
            **common_env,
            "KAPSL_BACKEND_CACHE_DIR": candidate_backend_cache,
            "KAPSL_GENERIC_NATIVE_PACKS": "1",
            "KAPSL_LAZY_ONNX_PACKS": "1",
        },
        "required_log_markers": [
            "Activated signed native backend pack onnx/cpu",
            "Selected signed native backend route onnx/cpu",
            "Activating signed backend route onnx/cpu",
        ],
        "forbidden_log_markers": [
            "Using embedded ORT rollback",
            "Activating embedded ORT rollback route",
        ],
    },
}
pathlib.Path(config_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

certification_status=0
PYTHONDONTWRITEBYTECODE=1 python3 "$parity_harness" certify \
  --config "$config_path" \
  --output-dir "$evidence_dir" || certification_status=$?

mkdir -p "$evidence_dir"
rm -f "$evidence_dir/kapsl.sock"
cp "$work_root/backend-ensure.log" "$evidence_dir/backend-ensure.log"
cp "$work_root/backend-list.json" "$evidence_dir/backend-list.json"
cp "$config_path" "$evidence_dir/certification-config.json"

if [ "$conformance_mode" = "smoke" ]; then
  smoke_validation_status=0
  python3 - \
    "$evidence_dir" \
    "$evidence_dir/smoke-validation.json" \
    "$certification_status" <<'PY' || smoke_validation_status=$?
import json
import pathlib
import sys

evidence_dir = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
harness_status = int(sys.argv[3])
failures = []

if harness_status not in (0, 1):
    failures.append(f"conformance harness exited before comparison: {harness_status}")

try:
    report = json.loads((evidence_dir / "report.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    report = {}
    failures.append(f"read conformance report: {error}")

if report.get("suite_id") != "ort-cpu-forward-smoke":
    failures.append("conformance report is not the PR correctness-smoke suite")
if not report.get("process_identity", {}).get("verified", False):
    failures.append("baseline and packaged routes did not use one owned process identity")
correctness = report.get("correctness", {})
if not correctness or any(
    not result.get("passed", False) for result in correctness.values()
):
    failures.append("embedded and packaged ORT outputs were not equivalent")
captures = report.get("captures", {})
if captures.get("baseline_sessions") != 2 or captures.get("candidate_sessions") != 2:
    failures.append("smoke did not complete the balanced load/unload lifecycle")

capture_paths = sorted(evidence_dir.glob("[0-9][0-9]-*.json"))
if len(capture_paths) != 4:
    failures.append("smoke did not retain all four lifecycle captures")
for capture_path in capture_paths:
    try:
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"read {capture_path.name}: {error}")
        continue
    if not capture.get("route_evidence", {}).get("verified", False):
        failures.append(f"{capture_path.name} did not prove its selected route")
    if capture.get("warmup_failures"):
        failures.append(f"{capture_path.name} recorded warmup failures")
    if any(int(trial.get("failures", 0)) != 0 for trial in capture.get("trials", [])):
        failures.append(f"{capture_path.name} recorded request failures")

validation = {
    "schema_version": 1,
    "status": "passed" if not failures else "failed",
    "harness_comparison_status": harness_status,
    "enforced": [
        "signed packaged route activation",
        "owned-process identity",
        "zero request failures",
        "tensor correctness",
        "balanced load and unload lifecycle",
    ],
    "not_enforced_on_pull_requests": [
        "throughput",
        "p95 latency",
        "p99 latency",
        "startup latency",
        "model memory",
        "process RSS",
    ],
    "failures": failures,
}
output_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
if failures:
    raise SystemExit("; ".join(failures))
PY
  certification_status=$smoke_validation_status
fi

python3 - \
  "$evidence_dir/certification-inputs.json" \
  "$engine_ref" \
  "$KAPSL_ORT_INTEGRATIONS_REF" \
  "$KAPSL_ORT_PARITY_HARNESS_PATH" \
  "$KAPSL_ORT_PARITY_HARNESS_SHA256" \
  "$KAPSL_ORT_PARITY_MODEL_REF" \
  "$KAPSL_ORT_PARITY_MODEL_PATH" \
  "$KAPSL_ORT_PARITY_MODEL_SHA256" \
  "$model_package_sha256" \
  "$(sha256sum "$archive_path" | awk '{ print $1 }')" <<'PY'
import json
import pathlib
import sys

(
    output,
    engine_commit,
    integrations_commit,
    conformance_path,
    conformance_sha256,
    model_source_commit,
    model_source_path,
    model_source_sha256,
    model_package_sha256,
    adapter_archive_sha256,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "engine_commit": engine_commit,
    "integrations_commit": integrations_commit,
    "conformance_harness": {
        "repository": "kapsl-runtime/kapsl-integrations",
        "commit": integrations_commit,
        "path": conformance_path,
        "sha256": conformance_sha256,
    },
    "model_source": {
        "commit": model_source_commit,
        "path": model_source_path,
        "sha256": model_source_sha256,
    },
    "model_package_sha256": model_package_sha256,
    "adapter_archive_sha256": adapter_archive_sha256,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "ORT CPU host $conformance_mode evidence: $evidence_dir"
exit "$certification_status"
