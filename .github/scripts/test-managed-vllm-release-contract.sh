#!/usr/bin/env bash
set -euo pipefail

lock_file=".github/scripts/managed-vllm-cu130.lock"
bootstrap=".github/scripts/bootstrap-vllm-backend.sh"
packager=".github/scripts/package-linux-vllm-backend.sh"
index_generator=".github/scripts/generate-backend-index.py"
runtime="kapsl-runtime/crates/kapsl-cli/src/runtime/managed_vllm.rs"
sdk_verifier=".github/scripts/verify-managed-vllm-sdk-checkout.sh"
wheel_verifier=".github/scripts/verify-managed-vllm-connector-wheel.py"
connector_version="0.7.0"
planner_schema_version="1"
kv_abi_major="1"
kv_abi_minor="5"

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
  "kapsl-vllm-connector==$connector_version"; do
  require_literal "$lock_file" "$pin"
  require_literal "$bootstrap" "$pin"
done

require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_TORCH_VERSION: &str = "2.13.0+cu130";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_TORCHVISION_VERSION: &str = "0.28.0+cu130";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_TORCHAUDIO_VERSION: &str = "2.11.0+cu130";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_CUDA_RUNTIME_VERSION: &str = "13.0";'
require_literal "$runtime" "pub(crate) const MANAGED_VLLM_ADAPTER_VERSION: &str = \"$connector_version\";"
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_PROFILE_ID: &str = "vllm-v1-packed-cuda-ipc/flash-attn";'
require_literal "$runtime" 'pub(crate) const MANAGED_VLLM_ELASTIC_PROFILE_ID: &str = "vllm-v1-packed-cuda-vmm/flash-attn-blnhc";'
require_literal "$runtime" "pub(crate) const MANAGED_VLLM_KV_ABI_MAJOR: u64 = $kv_abi_major;"
require_literal "$runtime" "pub(crate) const MANAGED_VLLM_KV_ABI_MINOR: u64 = $kv_abi_minor;"
require_literal "$runtime" "planner_schema_version: $planner_schema_version,"
require_literal "$packager" '--constraint "$requirements_lock"'
require_literal "$bootstrap" '--constraint "$requirements_lock"'
require_literal "$packager" ': "${KAPSL_VLLM_SDK_REF:?'
require_literal "$packager" 'verify-managed-vllm-sdk-checkout.sh "$sdk_dir" "$sdk_ref"'
require_literal "$packager" 'verify-managed-vllm-connector-wheel.py'
require_literal "$packager" "connector_version=\"$connector_version\""
require_literal "$packager" "planner_schema_version=\"$planner_schema_version\""
require_literal "$packager" "kv_abi_major=\"$kv_abi_major\""
require_literal "$packager" "kv_abi_minor=\"$kv_abi_minor\""
require_literal "$packager" '"connector_distribution": "$connector_version"'
require_literal "$packager" '"elastic_profile": "$connector_elastic_profile"'
require_literal "$packager" '"kv_abi": {"major": $kv_abi_major, "minor": $kv_abi_minor}'
require_literal "$packager" '"sdk_ref": "$sdk_ref"'
require_literal "$bootstrap" "\"connector_distribution\": \"$connector_version\""
require_literal "$bootstrap" "\"planner_schema_version\": $planner_schema_version"
require_literal "$bootstrap" '"elastic_profile": "vllm-v1-packed-cuda-vmm/flash-attn-blnhc"'
require_literal "$bootstrap" "\"kv_abi\": {\"major\": $kv_abi_major, \"minor\": $kv_abi_minor}"
require_literal "$packager" '"profile": "cu130-flash-attn"'
require_literal "$packager" '"runtime_abi": 1'
require_literal "$sdk_verifier" '^[0-9a-f]{40}$'
require_literal "$sdk_verifier" '--untracked-files=all'
require_literal "$sdk_verifier" '--ignored=matching'
require_literal "$wheel_verifier" '"planner_entry_point": "kapsl_vllm_connector.plan:main"'
require_literal "$index_generator" 'kapsl-backend-index-v1\0'
require_literal "$index_generator" 'kapsl-backend-artifact-v1\0'

for workflow in \
  .github/workflows/release-runtime-installers.yml \
  .github/workflows/beta-runtime-installers.yml \
  .github/workflows/vllm-shared-pool-conformance.yml; do
  require_literal "$workflow" 'KAPSL_VLLM_SDK_REF'
  require_literal "$workflow" 'verify-managed-vllm-sdk-checkout.sh'
  if [ "$workflow" != ".github/workflows/vllm-shared-pool-conformance.yml" ]; then
    require_literal "$workflow" '${{ vars.KAPSL_VLLM_SDK_REF }}'
    require_literal "$workflow" '--expected-public-key "$KAPSL_BACKEND_PUBLIC_KEYS"'
  else
    require_literal "$workflow" "EXPECTED_CONNECTOR_VERSION: \"$connector_version\""
    require_literal "$workflow" "EXPECTED_PLANNER_SCHEMA_VERSION: \"$planner_schema_version\""
    require_literal "$workflow" 'EXPECTED_ELASTIC_CONNECTOR_PROFILE: "vllm-v1-packed-cuda-vmm/flash-attn-blnhc"'
    require_literal "$workflow" "EXPECTED_KV_ABI_MAJOR: \"$kv_abi_major\""
    require_literal "$workflow" "EXPECTED_KV_ABI_MINOR: \"$kv_abi_minor\""
    require_literal "$workflow" 'EXPECTED_TORCHCODEC_VERSION: "0.16.0+cu130"'
    require_literal "$workflow" '"torchcodec==$EXPECTED_TORCHCODEC_VERSION"'
    require_literal "$workflow" 'ninja-build'
    require_literal "$workflow" '--index-url https://pypi.org/simple'
    require_literal "$workflow" '--extra-index-url "$PYTORCH_INDEX_URL"'
    require_literal "$workflow" 'engine/.cargo/config.toml'
    require_literal "$workflow" 'key: ${{ inputs.sdk_ref }}'
    require_literal "$workflow" '${{ runner.temp }}/kapsl-vllm-${{ github.run_id }}-${{ github.run_attempt }}/*.json'
    require_literal "$workflow" '${{ runner.temp }}/kapsl-vllm-${{ github.run_id }}-${{ github.run_attempt }}/*.log'
    require_literal "$workflow" '${{ runner.temp }}/kapsl-vllm-${{ github.run_id }}-${{ github.run_attempt }}/*.txt'
    require_literal "$workflow" '${{ runner.temp }}/kapsl-vllm-${{ github.run_id }}-${{ github.run_attempt }}/*.sha256'
    require_literal "$workflow" '${{ runner.temp }}/kapsl-vllm-${{ github.run_id }}-${{ github.run_attempt }}/wheels/*.whl'
    require_literal "$workflow" 'mixed-backend report did not pass'
    require_literal "$workflow" 'llama_owner_usage_bytes'
    require_literal "$workflow" 'general_pool_allocated_bytes'
  fi
done

if grep -Eq "grep .*kv_path=(shared-kv|native).*mixed-runtime\\.log" \
  .github/workflows/vllm-shared-pool-conformance.yml; then
  echo "Managed-vLLM conformance must gate mixed memory from metrics evidence, not log strings." >&2
  exit 1
fi

if [[ "$(grep -F -c -- '--trials 15' \
  .github/workflows/vllm-shared-pool-conformance.yml)" != "2" ]]; then
  echo "Managed-vLLM conformance must run 15 independent trials for each bridge target." >&2
  exit 1
fi

if grep -Fxq '            ${{ runner.temp }}/kapsl-vllm-${{ github.run_id }}-${{ github.run_attempt }}' \
  .github/workflows/vllm-shared-pool-conformance.yml; then
  echo "Managed-vLLM conformance must upload an evidence allowlist, not the artifact root." >&2
  exit 1
fi

if grep -Fq -- '--index-url "$PYTORCH_INDEX_URL"' \
  .github/workflows/vllm-shared-pool-conformance.yml; then
  echo "Managed-vLLM conformance must keep PyPI available for ordinary dependencies." >&2
  exit 1
fi

if grep -Eq 'KAPSL_VLLM_SDK_REF:-|sdk_ref:.*default:' "$packager" \
  .github/workflows/release-runtime-installers.yml \
  .github/workflows/beta-runtime-installers.yml \
  .github/workflows/vllm-shared-pool-conformance.yml; then
  echo "Managed-vLLM release paths must not carry a fallback SDK ref." >&2
  exit 1
fi

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
