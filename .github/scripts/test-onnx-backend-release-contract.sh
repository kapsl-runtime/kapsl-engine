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
runtime_backend="kapsl-runtime/crates/kapsl-cli/src/runtime/model/backend.rs"
native_host="kapsl-runtime/crates/kapsl-cli/src/backend/native.rs"
cli_manifest="kapsl-runtime/crates/kapsl-cli/Cargo.toml"

require_literal "$manager" 'pub(crate) const ONNX_CPU_PACK_PROFILE: &str = "cpu";'
require_literal "$manager" 'pub(crate) const ONNX_CUDA12_PACK_PROFILE: &str = "cuda12";'
require_literal "$manager" 'pub(crate) const ONNX_TENSORRT10_PACK_PROFILE: &str = "tensorrt10";'
require_literal "$activator" 'libloading::os::unix::Library::open'
require_literal "$activator" 'libc::RTLD_NOW | libc::RTLD_GLOBAL'
require_literal "$activator" 'TensorRT may only be selected when the .aimod explicitly declares it'
require_literal "$activator" 'if generic_native_backend_packs_enabled()?'
require_literal "$activator" 'activate_native_backend_pack(&pack_plan.manifest, &installed)?;'
require_literal "$runtime_backend" 'LLMBackend::with_device(provider.to_owned(), device_id as i32)'
require_literal "$runtime_backend" 'engine_kind.uses_onnx_session() && generic_native_backend_packs_enabled()?'
require_literal "$runtime_backend" 'return create_native_backend_pack_engine('
require_literal "$native_host" 'const GENERIC_NATIVE_PACKS_ENV: &str = "KAPSL_GENERIC_NATIVE_PACKS";'
require_literal "$native_host" 'KAPSL_BACKEND_ENTRYPOINT_SYMBOL'
require_literal "$native_host" 'KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR'
require_literal "$native_host" 'GpuDevicePool'
require_literal "$native_host" 'pointer.cast::<KapslBackendApiPrefixV1>().read()'
require_literal "$native_host" 'pack.api.shutdown'
require_literal "$cli_manifest" 'kapsl-backend-abi = "=0.1.0"'
require_literal "$packager" 'package_profile cpu cpu 1'
require_literal "$packager" 'package_profile cuda12 cuda 2'
require_literal "$packager" 'package_profile tensorrt10 tensorrt 3'
require_literal "$packager" '"execution_mode": "native"'
require_literal "$packager" '"entrypoint": "libkapsl_backend_onnx.so"'

python3 - "$runtime_backend" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
generic = source.index(
    "engine_kind.uses_onnx_session() && generic_native_backend_packs_enabled()?"
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
  require_literal "$workflow" '.github/scripts/collect-linux-tensorrt-runtime.sh'
done

echo "ONNX backend release contract tests passed."
