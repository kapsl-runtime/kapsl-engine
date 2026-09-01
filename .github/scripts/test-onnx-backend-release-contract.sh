#!/usr/bin/env bash
set -euo pipefail

require_literal() {
  file="$1"
  literal="$2"
  if ! grep -Fq -- "$literal" "$file"; then
    echo "$file is missing ONNX backend release contract: $literal" >&2
    exit 1
  fi
}

manager="kapsl-runtime/crates/kapsl-cli/src/backend/manager.rs"
activator="kapsl-runtime/crates/kapsl-cli/src/backend/onnx.rs"
packager=".github/scripts/package-linux-onnx-backend-packs.sh"
cpu_packager=".github/scripts/package-linux-ort-cpu-backend.sh"
accelerator_packager=".github/scripts/package-linux-ort-accelerator-backends.sh"
integration_verifier=".github/scripts/verify-ort-integration-checkout.sh"
index_generator=".github/scripts/generate-backend-index.py"
integration_lock=".github/ort-integration.lock"
parity_lock=".github/ort-cpu-parity.lock.json"
parity_certifier=".github/scripts/certify-ort-cpu-parity.sh"
parity_workflow=".github/workflows/ort-cpu-conformance.yml"
installer_workflow=".github/workflows/installer-smoke.yml"
runtime_backend="kapsl-runtime/crates/kapsl-cli/src/runtime/model/backend.rs"
native_host="kapsl-runtime/crates/kapsl-cli/src/backend/native.rs"
bundle="kapsl-runtime/crates/kapsl-cli/src/backend/bundle.rs"
cli_manifest="kapsl-runtime/crates/kapsl-cli/Cargo.toml"

require_literal "$manager" 'pub(crate) const ONNX_CPU_PACK_PROFILE: &str = "cpu";'
require_literal "$manager" 'pub(crate) const ONNX_CUDA12_PACK_PROFILE: &str = "cuda12";'
require_literal "$manager" 'pub(crate) const ONNX_TENSORRT10_PACK_PROFILE: &str = "tensorrt10";'
require_literal "$activator" 'libloading::os::unix::Library::open'
require_literal "$activator" 'libc::RTLD_NOW | libc::RTLD_GLOBAL'
require_literal "$activator" 'TensorRT may only be selected when the .aimod explicitly declares it'
require_literal "$activator" '!generic_native_backend_packs_enabled()?'
require_literal "$activator" 'pack_plan.manifest.adapter_abi.as_deref()'
require_literal "$activator" 'STANDARD_NATIVE_ADAPTER_ABI'
require_literal "$activator" 'activate_native_backend_pack(&pack_plan.manifest, &installed)?;'
require_literal "$runtime_backend" 'LLMBackend::with_device(provider.to_owned(), device_id as i32)'
require_literal "$runtime_backend" 'native_backend_pack_active_for_provider("onnx", provider)?'
require_literal "$runtime_backend" 'return create_native_backend_pack_engine('
require_literal "$native_host" 'const GENERIC_NATIVE_PACKS_ENV: &str = "KAPSL_GENERIC_NATIVE_PACKS";'
require_literal "$native_host" 'KAPSL_BACKEND_ENTRYPOINT_SYMBOL'
require_literal "$native_host" 'KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR'
require_literal "$native_host" 'GpuDevicePool'
require_literal "$native_host" '"pack_root": pack.root'
require_literal "$native_host" '"onnx_tuning": tuning.map'
require_literal "$native_host" 'pointer.cast::<KapslBackendApiPrefixV1>().read()'
require_literal "$native_host" 'pack.api.shutdown'
require_literal "$cli_manifest" 'kapsl-backend-abi = "=0.1.0"'
require_literal "$packager" 'package_profile cuda12 cuda 2'
require_literal "$packager" 'package_profile tensorrt10 tensorrt 3'
require_literal "$packager" '"execution_mode": "native"'
require_literal "$packager" '"entrypoint": "libkapsl_backend_onnx.so"'
require_literal "$cpu_packager" ': "${KAPSL_ORT_INTEGRATIONS_REF:?'
require_literal "$cpu_packager" 'verify-ort-integration-checkout.sh'
require_literal "$cpu_packager" 'integrations/ort/packaging/build_cpu_pack.sh'
require_literal "$cpu_packager" '"adapter_abi": "kapsl-backend-v1"'
require_literal "$cpu_packager" 'runtime_soname = "libonnxruntime.so.1"'
require_literal "$cpu_packager" 'v1.23.2/onnxruntime-linux-x64-1.23.2.tgz'
require_literal "$cpu_packager" '1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b'
require_literal "$cpu_packager" 'maximum_permitted_glibc'
require_literal "$cpu_packager" 'engine index publisher owns the release key'
require_literal "$accelerator_packager" ': "${KAPSL_ORT_INTEGRATIONS_REF:?'
require_literal "$accelerator_packager" 'verify-ort-integration-checkout.sh'
require_literal "$accelerator_packager" 'integrations/ort/packaging/build_accelerator_packs.sh'
require_literal "$accelerator_packager" 'for profile in cuda12 tensorrt10'
require_literal "$accelerator_packager" '"adapter_abi": "kapsl-backend-v1"'
require_literal "$accelerator_packager" 'v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz'
require_literal "$accelerator_packager" '2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53'
require_literal "$accelerator_packager" 'pack contains host driver libraries'
require_literal "$accelerator_packager" 'engine index publisher owns the release key'
require_literal "$integration_verifier" '^[0-9a-f]{40}$'
require_literal "$integration_verifier" '--untracked-files=all'
require_literal "$integration_verifier" '--ignored=matching'
require_literal "$index_generator" 'adapter_abi != "kapsl-backend-v1"'
require_literal "$bundle" 'backend_artifacts_dir'
require_literal "$bundle" 'manager.verify_pack_archive(pack, &candidate)?;'
require_literal "$manager" 'let model_paths = crate::backend::expand_run_bundles(&args.model, &device_info)?;'
require_literal "$parity_certifier" '--target linux-x86_64-cpu'
require_literal "$parity_certifier" 'KAPSL_GENERIC_NATIVE_PACKS=1'
require_literal "$parity_certifier" 'PYTHONDONTWRITEBYTECODE=1'
require_literal "$parity_certifier" 'KAPSL_ORT_PARITY_HARNESS_PATH'
require_literal "$parity_certifier" '"warmup_requests": 40'
require_literal "$parity_certifier" '"requests_per_payload": 1000'
require_literal "$parity_certifier" 'certification_status=0'
require_literal "$parity_certifier" 'rm -f "$evidence_dir/kapsl.sock"'
require_literal "$parity_certifier" 'exit "$certification_status"'
if ! grep -Eq '^[0-9a-f]{40}$' "$integration_lock" \
  || [ "$(wc -l < "$integration_lock" | tr -d ' ')" != "1" ]; then
  echo "$integration_lock must contain exactly one lowercase 40-hex commit." >&2
  exit 1
fi

python3 - "$parity_lock" <<'PY'
import json
import pathlib
import re
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert lock.get("schema_version") == 1
assert lock.get("conformance", {}).get("repository") == "kapsl-runtime/kapsl-integrations"
assert lock.get("model", {}).get("repository") == "kapsl-runtime/kapsl-sdk"
assert re.fullmatch(r"[0-9a-f]{64}", lock["conformance"]["entrypoint_sha256"])
assert re.fullmatch(r"[0-9a-f]{40}", lock["model"]["commit"])
assert re.fullmatch(r"[0-9a-f]{64}", lock["model"]["sha256"])
for entry in (lock["conformance"], lock["model"]):
    path = pathlib.PurePosixPath(entry["path"])
    assert not path.is_absolute() and ".." not in path.parts
PY

if grep -Fq 'package_profile cpu ' "$packager"; then
  echo "$packager must not publish the legacy provider-only CPU bundle" >&2
  exit 1
fi

python3 - "$runtime_backend" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
generic = source.index(
    'native_backend_pack_active_for_provider("onnx", provider)?'
)
embedded = source.index("if engine_kind.is_onnx_generate()")
if generic >= embedded:
    raise SystemExit("generic native ONNX selection must precede every embedded ORT constructor")
PY

if grep -Fq 'LD_LIBRARY_PATH' "$packager"; then
  echo "$packager must not modify LD_LIBRARY_PATH" >&2
  exit 1
fi

for workflow in \
  .github/workflows/beta-runtime-installers.yml \
  .github/workflows/release-runtime-installers.yml; do
  require_literal "$workflow" '.github/scripts/package-linux-onnx-backend-packs.sh'
  require_literal "$workflow" '.github/scripts/package-linux-ort-cpu-backend.sh'
  require_literal "$workflow" '.github/ort-integration.lock'
  require_literal "$workflow" 'ref: ${{ steps.ort-integrations.outputs.ref }}'
  require_literal "$workflow" 'repository: kapsl-runtime/kapsl-integrations'
  require_literal "$workflow" 'verify-ort-integration-checkout.sh'
  require_literal "$workflow" 'Install certified ORT packaging toolchain'
  require_literal "$workflow" 'rustup toolchain install "$toolchain" --profile minimal'
  require_literal "$workflow" '.github/scripts/collect-linux-tensorrt-runtime.sh'
done

require_literal "$parity_workflow" 'name: ORT CPU Conformance'
require_literal "$parity_workflow" 'cancel-in-progress: true'
require_literal "$parity_workflow" '.github/ort-cpu-parity.lock.json'
require_literal "$parity_workflow" 'repository: kapsl-runtime/kapsl-sdk'
require_literal "$parity_workflow" '.github/scripts/certify-ort-cpu-parity.sh'
require_literal "$parity_certifier" '"sequence": ["baseline", "candidate", "candidate", "baseline"] * 2'
require_literal "$installer_workflow" '.github/scripts/test-package-linux-ort-accelerator-backends.sh'
if grep -Fq 'Certify embedded versus packaged ORT CPU parity' "$installer_workflow"; then
  echo "$installer_workflow must not run ORT performance conformance." >&2
  exit 1
fi

python3 - "$parity_lock" "$parity_workflow" <<'PY'
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
workflow = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
conformance_dir = pathlib.PurePosixPath(lock["conformance"]["path"]).parent
expected = f"uses: ./kapsl-integrations-ort/{conformance_dir}"
if expected not in workflow:
    raise SystemExit("ORT CPU conformance must use parity from the locked integrations checkout")
if "kapsl-runtime/kapsl-benchmarks" in workflow:
    raise SystemExit("public engine CI must not depend on a private benchmark action")
PY

if grep -Eq 'KAPSL_ORT_INTEGRATIONS_REF:-|ref:.*feature/ort|ref:.*(main|develop)' \
  "$cpu_packager" \
  .github/workflows/beta-runtime-installers.yml \
  .github/workflows/release-runtime-installers.yml; then
  echo "ORT release paths must use an exact configured integrations commit without a branch fallback." >&2
  exit 1
fi

echo "ONNX backend release contract tests passed."
