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

manager="kapsl-runtime/crates/kapsl-cli/src/backend_manager.rs"
activator="kapsl-runtime/crates/kapsl-cli/src/onnx_backend_pack.rs"
packager=".github/scripts/package-linux-onnx-backend-packs.sh"
runtime_backend="kapsl-runtime/crates/kapsl-cli/src/runtime/model/backend.rs"

require_literal "$manager" 'pub(crate) const ONNX_CPU_PACK_PROFILE: &str = "cpu";'
require_literal "$manager" 'pub(crate) const ONNX_CUDA12_PACK_PROFILE: &str = "cuda12";'
require_literal "$manager" 'pub(crate) const ONNX_TENSORRT10_PACK_PROFILE: &str = "tensorrt10";'
require_literal "$activator" 'libloading::os::unix::Library::open'
require_literal "$activator" 'libc::RTLD_NOW | libc::RTLD_GLOBAL'
require_literal "$activator" 'TensorRT may only be selected when the .aimod explicitly declares it'
require_literal "$runtime_backend" 'LLMBackend::with_device(provider.to_owned(), device_id as i32)'
require_literal "$packager" 'package_profile cpu cpu 1'
require_literal "$packager" 'package_profile cuda12 cuda 2'
require_literal "$packager" 'package_profile tensorrt10 tensorrt 3'
require_literal "$packager" '"execution_mode": "native"'
require_literal "$packager" '"entrypoint": "libkapsl_backend_onnx.so"'

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
