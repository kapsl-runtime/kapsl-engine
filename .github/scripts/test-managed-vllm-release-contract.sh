#!/usr/bin/env bash
set -euo pipefail

lock_file=".github/scripts/managed-vllm-cu130.lock"
bootstrap=".github/scripts/bootstrap-vllm-backend.sh"
packager=".github/scripts/package-linux-vllm-backend.sh"
runtime="kapsl-runtime/crates/kapsl-cli/src/runtime/managed_vllm.rs"
sdk_ref="3a4e626f919e11287e0a19bb720c547ec9216f7f"

require_literal() {
  local file="$1"
  local literal="$2"
  if ! grep -Fq -- "$literal" "$file"; then
    echo "$file is missing certified release contract: $literal" >&2
    exit 1
  fi
}

for pin in \
  'torch==2.13.0+cu130' \
  'torchvision==0.28.0+cu130' \
  'torchaudio==2.11.0+cu130' \
  'vllm==0.26.1rc1.dev1130+g2ec6f0d71' \
  'kapsl-vllm-connector==0.5.0'; do
  require_literal "$lock_file" "$pin"
  require_literal "$bootstrap" "$pin"
done

require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_TORCH_VERSION: &str = "2.13.0+cu130";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_TORCHVISION_VERSION: &str = "0.28.0+cu130";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_TORCHAUDIO_VERSION: &str = "2.11.0+cu130";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_CUDA_RUNTIME_VERSION: &str = "13.0";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_PROFILE_ID: &str = "vllm-v1-packed-cuda-ipc/flash-attn";'
require_literal "$packager" '--constraint "$requirements_lock"'
require_literal "$bootstrap" '--constraint "$requirements_lock"'

for workflow in \
  .github/workflows/release-runtime-installers.yml \
  .github/workflows/beta-runtime-installers.yml \
  .github/workflows/vllm-shared-pool-conformance.yml; do
  require_literal "$workflow" "$sdk_ref"
done
require_literal "$packager" "$sdk_ref"

awk -F '==' '
  /^[[:space:]]*($|#)/ { next }
  NF != 2 || $1 !~ /^[A-Za-z0-9_.-]+$/ || $2 == "" {
    print "invalid lock entry: " $0 > "/dev/stderr"
    failed = 1
    next
  }
  {
    key = tolower($1)
    gsub(/[._-]+/, "-", key)
    if (seen[key]++) {
      print "duplicate normalized package in lock: " key > "/dev/stderr"
      failed = 1
    }
  }
  END { exit failed }
' "$lock_file"

echo "Managed-vLLM release contract tests passed."
