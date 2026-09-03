#!/usr/bin/env bash
set -euo pipefail

require_literal() {
  file="$1"
  literal="$2"
  if ! grep -Fq -- "$literal" "$file"; then
    echo "$file is missing llama.cpp backend release contract: $literal" >&2
    exit 1
  fi
}

header="kapsl-runtime/include/kapsl_llama_cpp_backend.h"
manager="kapsl-runtime/crates/kapsl-cli/src/backend/manager.rs"
loader="kapsl-runtime/crates/kapsl-cli/src/backend/llama_cpp/mod.rs"
packager=".github/scripts/package-linux-llama-cpp-backend-packs.sh"
manifest="kapsl-runtime/crates/kapsl-backend-llama-cpp/Cargo.toml"
cli_manifest="kapsl-runtime/crates/kapsl-cli/Cargo.toml"

require_literal "$manifest" 'kapsl-backend-abi = "=0.2.0"'
require_literal "$cli_manifest" 'kapsl-backend-abi = "=0.2.0"'
require_literal "$header" '#define KAPSL_LLAMA_CPP_ABI_VERSION 1u'
require_literal "$header" 'const kapsl_llama_cpp_api_v1 *kapsl_llama_cpp_backend_v1(void);'
require_literal "$manager" 'pub(crate) const LLAMA_CPP_CPU_PACK_PROFILE: &str = "cpu";'
require_literal "$manager" 'pub(crate) const LLAMA_CPP_CUDA12_PACK_PROFILE: &str = "cuda12";'
require_literal "$loader" 'KAPSL_LLAMA_CPP_ALLOW_NATIVE_KV'
require_literal "$loader" 'preliminary memory admission rejected llama.cpp/'
require_literal "$manifest" 'kapsl-engine-api = "0.3.0"'
require_literal "$manifest" 'kapsl-llm = { version = "0.3.0"'
require_literal "$packager" 'package_profile cpu cpu cpu'
require_literal "$packager" 'package_profile cuda12 cuda cuda12-shared-pool "${KAPSL_LLAMA_CUDA_LIBRARY:-}" shared_pool'
require_literal "$packager" '--locked'
require_literal "$packager" '"kv_mode": kv_mode'
require_literal "$packager" 'KAPSL_LLAMA_CPP_KV_MODE=$kv_mode'
require_literal "$packager" 'copy_runtime_dependencies "$library" "$root/lib" "$accelerator"'
require_literal "$packager" 'NVIDIA-CONTAINER-LICENSE'
require_literal "$packager" 'NVIDIA driver libraries must not be bundled'
require_literal "$packager" '"execution_mode": "native"'
require_literal "$packager" '"entrypoint": "lib/libkapsl_backend_llama_cpp.so"'

for workflow in \
  .github/workflows/beta-runtime-installers.yml \
  .github/workflows/release-runtime-installers.yml; do
  require_literal "$workflow" '.github/scripts/package-linux-llama-cpp-backend-packs.sh'
  require_literal "$workflow" 'Package llama.cpp backend packs from published SDK crates'
  require_literal "$workflow" 'cargo build --manifest-path kapsl-runtime/Cargo.toml --locked'
done

if grep -Eq 'KAPSL_LLAMA_SDK_(DIR|REF)|sdk-llama|patch\.crates-io\.kapsl-(llm|engine-api)' \
  "$packager" \
  .github/workflows/beta-runtime-installers.yml \
  .github/workflows/release-runtime-installers.yml; then
  echo "llama.cpp release paths must resolve published SDK crates without path patches." >&2
  exit 1
fi

require_literal ".github/workflows/lazy-llama-cpp-backend-pack-gpu-certification.yml" \
  'KAPSL_GPU_TEST_REQUIRE_LAZY_LLAMA_PACK'
require_literal ".github/workflows/lazy-llama-cpp-backend-pack-gpu-certification.yml" \
  'certify-llama-cpp-backend-pack-performance.sh'

echo "llama.cpp backend release contract tests passed."
