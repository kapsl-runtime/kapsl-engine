#!/usr/bin/env bash
set -euo pipefail

# Keep the managed-SGLang pins identical across the certified dependency lock,
# the packager, the pack-local bootstrap, the runtime, and the release build.

lock_file=".github/scripts/managed-sglang-cu130.lock"
integration_lock=".github/sglang-integration.lock"
bootstrap=".github/scripts/bootstrap-sglang-backend.sh"
packager=".github/scripts/package-linux-sglang-backend.sh"
checkout_verifier=".github/scripts/verify-sglang-integration-checkout.sh"
wheel_verifier=".github/scripts/verify-managed-sglang-adapter-wheel.py"
runtime="kapsl-runtime/crates/kapsl-cli/src/runtime/managed/sglang.rs"
manager="kapsl-runtime/crates/kapsl-cli/src/backend/manager.rs"
build_workflow=".github/workflows/build-linux-accelerators.yml"
build_inputs=".github/scripts/release_build_inputs.py"
adapter_version="0.2.0"
sglang_version="0.5.23"
torch_version="2.14.1"
torchvision_version="0.29.2"
torchaudio_version="2.11.0"
kv_abi_major="1"
kv_abi_minor="5"
participant_config_env="KAPSL_KV_PARTICIPANT_CONFIG"
pack_profile="cu130"

require_literal() {
  local file="$1"
  local literal="$2"
  if ! grep -Fq -- "$literal" "$file"; then
    echo "$file is missing certified release contract: $literal" >&2
    exit 1
  fi
}

require_line() {
  local file="$1"
  local line="$2"
  if ! grep -Fxq -- "$line" "$file"; then
    echo "$file is missing certified release pin: $line" >&2
    exit 1
  fi
}

for pin in \
  "torch==$torch_version" \
  "torchvision==$torchvision_version" \
  "torchaudio==$torchaudio_version" \
  "sglang==$sglang_version"; do
  require_line "$lock_file" "$pin"
  require_literal "$bootstrap" "\"$pin\""
done
# The adapter is built from the pinned kapsl-integrations commit, never
# resolved from an index, so only the bootstrap names it.
require_literal "$bootstrap" "\"kapsl-sglang-adapter==$adapter_version\""
if grep -Eiq '^kapsl[-_.]sglang[-_.]adapter==' "$lock_file"; then
  echo "$lock_file must not pin the locally built Kapsl SGLang adapter." >&2
  exit 1
fi

# The packager installs into the pack's own Python, which the bootstrap copies
# into the installed environment, so everything it installs ships. Each
# package must carry the certified lock's exact pin, and setuptools, which
# builds the adapter, must be among them.
packager_installs="$(awk '
  /-m pip install/ { collecting = 1; sub(/.*-m pip install/, "") }
  collecting {
    continued = sub(/\\[[:space:]]*$/, "")
    for (i = 1; i <= NF; i++) {
      word = $i
      gsub(/"/, "", word)
      if (word !~ /^-/) print word
    }
    if (!continued) collecting = 0
  }
' "$packager")"
if [ -z "$packager_installs" ]; then
  echo "$packager installs nothing into the pack's Python; expected the lock's setuptools." >&2
  exit 1
fi
while IFS= read -r requirement; do
  if [[ ! "$requirement" =~ ^[A-Za-z0-9_.-]+==[^=]+$ ]]; then
    echo "$packager installs $requirement without an exact name==version pin." >&2
    exit 1
  fi
  if ! grep -Fxq -- "$requirement" "$lock_file"; then
    echo "$packager installs $requirement into the pack, but $lock_file does not pin it." >&2
    exit 1
  fi
done <<< "$packager_installs"
lock_setuptools="$(grep -E '^setuptools==' "$lock_file" || true)"
if [ -z "$lock_setuptools" ] || ! grep -Fxq -- "$lock_setuptools" <<< "$packager_installs"; then
  echo "$packager must install the lock's setuptools pin (${lock_setuptools:-none in the lock})." >&2
  exit 1
fi

if ! grep -Exq '[0-9a-f]{40}' "$integration_lock" \
  || [ "$(grep -c . "$integration_lock")" != "1" ]; then
  echo "$integration_lock must contain exactly one lowercase 40-hex commit." >&2
  exit 1
fi

require_literal "$runtime" "pub(crate) const MANAGED_SGLANG_ADAPTER_VERSION: &str = \"$adapter_version\";"
require_literal "$runtime" "pub(crate) const MANAGED_SGLANG_BACKEND_VERSION: &str = \"$sglang_version\";"
require_literal "$manager" "pub(crate) const MANAGED_SGLANG_PACK_PROFILE: &str = \"$pack_profile\";"

require_literal "$packager" ': "${KAPSL_SGLANG_INTEGRATIONS_REF:?'
require_literal "$packager" 'verify-sglang-integration-checkout.sh "$integrations_dir" "$integrations_ref"'
require_literal "$packager" 'verify-managed-sglang-adapter-wheel.py'
require_literal "$packager" '--constraint "$requirements_lock"'
require_literal "$packager" "requirements_lock=\"$lock_file\""
require_literal "$packager" "adapter_version=\"$adapter_version\""
require_literal "$packager" "sglang_version=\"$sglang_version\""
require_literal "$packager" "torch_version=\"$torch_version\""
require_literal "$packager" "torchvision_version=\"$torchvision_version\""
require_literal "$packager" "torchaudio_version=\"$torchaudio_version\""
require_literal "$packager" "kv_abi_major=\"$kv_abi_major\""
require_literal "$packager" "kv_abi_minor=\"$kv_abi_minor\""
require_literal "$packager" "participant_config_env=\"$participant_config_env\""
require_literal "$packager" "pack_profile=\"$pack_profile\""
require_literal "$packager" '"backend": "sglang"'
require_literal "$packager" '"runtime_abi": 1'
require_literal "$packager" '"execution_mode": "external"'
require_literal "$packager" '"integrations_ref": "$integrations_ref"'
require_literal "$packager" '"kv_abi": {"major": $kv_abi_major, "minor": $kv_abi_minor}'
require_literal "$bootstrap" '--constraint "$requirements_lock"'
require_literal "$bootstrap" "\"kv_abi\": {\"major\": $kv_abi_major, \"minor\": $kv_abi_minor}"
require_literal "$bootstrap" "\"participant_config_env\": \"$participant_config_env\""
require_literal "$bootstrap" "\"sglang\": \"$sglang_version\""
require_literal "$bootstrap" "\"adapter\": \"$adapter_version\""
require_literal "$checkout_verifier" '^[0-9a-f]{40}$'
require_literal "$checkout_verifier" '--untracked-files=all'
require_literal "$checkout_verifier" '--ignored=matching'
require_literal "$wheel_verifier" '"plugin_entry_point": "kapsl_sglang_adapter.hooks:activate"'
require_literal "$wheel_verifier" '"launcher_entry_point": "kapsl_sglang_adapter.launch:main"'

require_literal "$build_workflow" "$integration_lock"
require_literal "$build_workflow" 'ref: ${{ steps.sglang-integrations.outputs.ref }}'
require_literal "$build_workflow" 'verify-sglang-integration-checkout.sh kapsl-integrations-sglang "$KAPSL_SGLANG_INTEGRATIONS_REF"'
require_literal "$build_workflow" '.github/scripts/package-linux-sglang-backend.sh'
require_literal "$build_inputs" '{"component": "sglang", "target": "", "profile": ""}'

if grep -Eq 'KAPSL_SGLANG_INTEGRATIONS_REF:-' "$packager" "$build_workflow"; then
  echo "Managed-SGLang release paths must not carry a fallback integrations ref." >&2
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

echo "Managed-SGLang release contract tests passed."
