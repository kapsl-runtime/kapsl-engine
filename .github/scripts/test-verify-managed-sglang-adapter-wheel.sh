#!/usr/bin/env bash
set -euo pipefail

# Build fake adapter wheels and check that the verifier accepts the release
# tuple and refuses each kind of drift.

test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

make_wheel() { # path [field=value ...]
  python3 - "$@" <<'PY'
import sys
import zipfile

path = sys.argv[1]
fields = {
    "name": "kapsl-sglang-adapter",
    "version": "0.2.0",
    "adapter": "0.2.0",
    "abi": '{"major": 1, "minor": 5}',
    "pins": '("0.5.23",)',
    "env": "KAPSL_KV_PARTICIPANT_CONFIG",
    "plugin": "kapsl_sglang_adapter.hooks:activate",
    "launcher": "kapsl_sglang_adapter.launch:main",
    "launch_module": "yes",
}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    fields[key] = value
package = "kapsl_sglang_adapter"
info = f"kapsl_sglang_adapter-{fields['version']}.dist-info"
with zipfile.ZipFile(path, "w") as wheel:
    wheel.writestr(
        f"{info}/METADATA",
        f"Metadata-Version: 2.1\nName: {fields['name']}\nVersion: {fields['version']}\n",
    )
    wheel.writestr(
        f"{info}/entry_points.txt",
        "[console_scripts]\n"
        f"kapsl-sglang-serve = {fields['launcher']}\n\n"
        "[sglang.srt.plugins]\n"
        f"kapsl_kv = {fields['plugin']}\n",
    )
    wheel.writestr(f"{package}/__init__.py", f'ADAPTER_VERSION = "{fields["adapter"]}"\n')
    wheel.writestr(
        f"{package}/contract.py",
        f"ABI_VERSION: dict[str, int] = {fields['abi']}\n",
    )
    wheel.writestr(
        f"{package}/capacity.py",
        f"PINNED_SGLANG_VERSIONS: tuple[str, ...] = {fields['pins']}\n",
    )
    wheel.writestr(
        f"{package}/participant_config.py",
        f'PARTICIPANT_CONFIG_ENV = "{fields["env"]}"\n',
    )
    wheel.writestr(f"{package}/hooks.py", "def activate():\n    pass\n")
    if fields["launch_module"] == "yes":
        wheel.writestr(f"{package}/launch.py", "def main():\n    return 0\n")
PY
}
verify() { # wheel
  .github/scripts/verify-managed-sglang-adapter-wheel.py \
    --wheel "$1" \
    --adapter-version 0.2.0 \
    --sglang-version 0.5.23 \
    --kv-abi-major 1 \
    --kv-abi-minor 5
}
expect_rejected() { # label field=value
  local wheel="$test_root/$1.whl"
  make_wheel "$wheel" "$2"
  if verify "$wheel" >/dev/null 2>&1; then
    echo "The adapter wheel verifier accepted $1." >&2
    exit 1
  fi
}

make_wheel "$test_root/good.whl"
verify "$test_root/good.whl"

expect_rejected other-distribution name=kapsl-vllm-connector
expect_rejected other-version version=0.1.0
expect_rejected other-adapter-constant adapter=0.1.0
expect_rejected other-kv-abi 'abi={"major": 1, "minor": 4}'
expect_rejected unpinned-sglang 'pins=("0.5.20",)'
expect_rejected other-config-env env=KAPSL_SGLANG_CONFIG
expect_rejected no-plugin plugin=other.module:activate
expect_rejected no-launcher launcher=other.module:main
expect_rejected no-launch-module launch_module=no

echo "Managed-SGLang adapter wheel verifier tests passed."
