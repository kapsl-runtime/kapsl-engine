#!/usr/bin/env bash
set -euo pipefail

verifier=".github/scripts/verify-managed-vllm-connector-wheel.py"
profile="vllm-v1-packed-cuda-ipc/flash-attn"
elastic_profile="vllm-v1-packed-cuda-vmm/flash-attn-blnhc"
test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

make_wheel() {
  local path="$1"
  local distribution_version="$2"
  local adapter_version="$3"
  local planner_schema="$4"
  local planner_entry_point="$5"
  local emitted_elastic_profile="${6:-$elastic_profile}"
  local abi_major="${7:-1}"
  local abi_minor="${8:-5}"
  python3 - "$path" "$distribution_version" "$adapter_version" "$planner_schema" "$planner_entry_point" "$emitted_elastic_profile" "$abi_major" "$abi_minor" <<'PY'
import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])
(
    distribution_version,
    adapter_version,
    planner_schema,
    planner_entry_point,
    elastic_profile,
    abi_major,
    abi_minor,
) = sys.argv[2:]
dist_info = f"kapsl_vllm_connector-{distribution_version}.dist-info"
with zipfile.ZipFile(path, "w") as wheel:
    wheel.writestr(
        f"{dist_info}/METADATA",
        "Metadata-Version: 2.4\n"
        "Name: kapsl-vllm-connector\n"
        f"Version: {distribution_version}\n",
    )
    wheel.writestr(
        "kapsl_vllm_connector/connector.py",
        f'ADAPTER_VERSION = "{adapter_version}"\n'
        'ADAPTER_PROFILE_ID = "vllm-v1-packed-cuda-ipc/flash-attn"\n'
        f'ELASTIC_ADAPTER_PROFILE_ID = "{elastic_profile}"\n',
    )
    wheel.writestr(
        "kapsl_vllm_connector/contract.py",
        f'ABI_VERSION = {{"major": {abi_major}, "minor": {abi_minor}}}\n',
    )
    wheel.writestr(
        "kapsl_vllm_connector/planning.py",
        f"PLANNER_SCHEMA_VERSION = {planner_schema}\n",
    )
    wheel.writestr(
        f"{dist_info}/entry_points.txt",
        "[console_scripts]\n"
        f"kapsl-vllm-plan = {planner_entry_point}\n",
    )
PY
}

verify() {
  "$verifier" \
    --wheel "$1" \
    --connector-version 0.7.0 \
    --profile "$profile" \
    --elastic-profile "$elastic_profile" \
    --planner-schema 1 \
    --kv-abi-major 1 \
    --kv-abi-minor 5
}

expect_failure() {
  if verify "$1" >/dev/null 2>&1; then
    echo "Managed-vLLM wheel verifier unexpectedly accepted $1" >&2
    exit 1
  fi
}

valid="$test_root/valid.whl"
make_wheel "$valid" 0.7.0 0.7.0 1 kapsl_vllm_connector.plan:main
verify "$valid"

wrong_distribution="$test_root/wrong-distribution.whl"
make_wheel "$wrong_distribution" 0.6.0 0.7.0 1 kapsl_vllm_connector.plan:main
expect_failure "$wrong_distribution"

wrong_module="$test_root/wrong-module.whl"
make_wheel "$wrong_module" 0.7.0 0.6.0 1 kapsl_vllm_connector.plan:main
expect_failure "$wrong_module"

wrong_schema="$test_root/wrong-schema.whl"
make_wheel "$wrong_schema" 0.7.0 0.7.0 2 kapsl_vllm_connector.plan:main
expect_failure "$wrong_schema"

wrong_entry_point="$test_root/wrong-entry-point.whl"
make_wheel "$wrong_entry_point" 0.7.0 0.7.0 1 kapsl_vllm_connector.other:main
expect_failure "$wrong_entry_point"

wrong_elastic_profile="$test_root/wrong-elastic-profile.whl"
make_wheel "$wrong_elastic_profile" 0.7.0 0.7.0 1 kapsl_vllm_connector.plan:main vllm-v1-wrong 1 5
expect_failure "$wrong_elastic_profile"

wrong_abi="$test_root/wrong-abi.whl"
make_wheel "$wrong_abi" 0.7.0 0.7.0 1 kapsl_vllm_connector.plan:main "$elastic_profile" 1 4
expect_failure "$wrong_abi"

echo "Managed-vLLM connector wheel verifier tests passed."
