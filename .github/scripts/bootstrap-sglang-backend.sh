#!/usr/bin/env bash
set -euo pipefail

# Install the managed-SGLang backend from its offline wheelhouse into
# TARGET_ROOT, check every pinned version, and swap it in atomically.

if [ "$#" -ne 2 ]; then
  echo "usage: bootstrap-sglang-backend.sh BOOTSTRAP_ROOT TARGET_ROOT" >&2
  exit 2
fi

bootstrap_root="$(cd "$1" && pwd)"
target_root="$2"
python_source="$bootstrap_root/python"
wheelhouse="$bootstrap_root/wheels"
checksums="$bootstrap_root/SHA256SUMS"
requirements_lock="$bootstrap_root/requirements.lock"
installed_manifest="$bootstrap_root/installed-manifest.json"

for required in \
  "$python_source/bin/python3.12" \
  "$requirements_lock" \
  "$installed_manifest" \
  "$checksums"; do
  if [ ! -e "$required" ]; then
    echo "Incomplete managed-SGLang bootstrap: missing $required" >&2
    exit 1
  fi
done
if [ ! -d "$wheelhouse" ]; then
  echo "Incomplete managed-SGLang bootstrap: missing $wheelhouse" >&2
  exit 1
fi

(cd "$bootstrap_root" && sha256sum --check SHA256SUMS)

target_parent="$(dirname "$target_root")"
target_name="$(basename "$target_root")"
mkdir -p "$target_parent"
staging_root="$(mktemp -d "$target_parent/.${target_name}.install.XXXXXX")"
backup_root=""

cleanup() {
  if [ -n "$staging_root" ] && [ -d "$staging_root" ]; then
    rm -rf "$staging_root"
  fi
  if [ -n "$backup_root" ] && [ -d "$backup_root" ] && [ ! -e "$target_root" ]; then
    mv "$backup_root" "$target_root"
  fi
}
trap cleanup EXIT INT TERM

cp -a "$python_source/." "$staging_root/"
if [ -d "$bootstrap_root/licenses" ]; then
  cp -a "$bootstrap_root/licenses" "$staging_root/licenses"
fi
ln -sfn python3.12 "$staging_root/bin/python3"
ln -sfn python3.12 "$staging_root/bin/python"

staging_python="$staging_root/bin/python"
"$staging_python" - "$installed_manifest" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": 1,
    "python": "3.12.3",
    "torch": "2.14.1",
    "torchvision": "0.29.2",
    "torchaudio": "2.11.0",
    "cuda_runtime": "13.0",
    "sglang": "0.5.23",
    "adapter_distribution": "0.2.0",
    "adapter": "0.2.0",
    "kv_abi": {"major": 1, "minor": 5},
    "participant_config_env": "KAPSL_KV_PARTICIPANT_CONFIG",
}
actual = {key: manifest.get(key) for key in expected}
if actual != expected:
    raise SystemExit(
        f"managed-SGLang installed manifest mismatch: {actual!r} != {expected!r}"
    )
ref = manifest.get("integrations_ref")
if not isinstance(ref, str) or re.fullmatch(r"[0-9a-f]{40}", ref) is None:
    raise SystemExit(
        "managed-SGLang installed manifest integrations_ref must be an exact "
        f"lowercase 40-hex commit, got {ref!r}"
    )
PY
"$staging_python" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-index \
  --find-links "$wheelhouse" \
  --constraint "$requirements_lock" \
  "torch==2.14.1" \
  "torchvision==0.29.2" \
  "torchaudio==2.11.0" \
  "sglang==0.5.23" \
  "kapsl-sglang-adapter==0.2.0"
"$staging_python" -m pip check

# Importing the adapter does not import SGLang; torch is imported only for
# its CUDA runtime version, which needs no GPU.
"$staging_python" <<'PY'
import importlib.metadata as md
import json
import platform

import kapsl_sglang_adapter as adapter
import torch

actual = {
    "python": platform.python_version(),
    "torch": md.version("torch"),
    "torchvision": md.version("torchvision"),
    "torchaudio": md.version("torchaudio"),
    "sglang": md.version("sglang"),
    "adapter_distribution": md.version("kapsl-sglang-adapter"),
    "adapter": adapter.ADAPTER_VERSION,
    "kv_abi": dict(adapter.ABI_VERSION),
    "cuda_runtime": str(torch.version.cuda),
    "participant_config_env": adapter.PARTICIPANT_CONFIG_ENV,
    "pinned_sglang": "0.5.23" in adapter.PINNED_SGLANG_VERSIONS,
}
expected = {
    "python": "3.12.3",
    "torch": "2.14.1",
    "torchvision": "0.29.2",
    "torchaudio": "2.11.0",
    "sglang": "0.5.23",
    "adapter_distribution": "0.2.0",
    "adapter": "0.2.0",
    "kv_abi": {"major": 1, "minor": 5},
    "cuda_runtime": "13.0",
    "participant_config_env": "KAPSL_KV_PARTICIPANT_CONFIG",
    "pinned_sglang": True,
}
if actual != expected:
    raise SystemExit(f"managed-SGLang bundle mismatch: {actual!r} != {expected!r}")
print(json.dumps(actual, sort_keys=True))
PY

# pip-generated entry points embed the temporary absolute prefix. Kapsl puts
# this bin directory first on the child PATH, so env-based shebangs remain
# relocatable after the atomic rename below.
while IFS= read -r script; do
  if [ "$(head -c 2 "$script" 2>/dev/null || true)" != '#!' ]; then
    continue
  fi
  first_line="$(head -n 1 "$script")"
  case "$first_line" in
    '#!'*python*) sed -i '1c #!/usr/bin/env python3' "$script" ;;
  esac
done < <(find "$staging_root/bin" -maxdepth 1 -type f -perm -u+x -print)

find "$staging_root" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$staging_root" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

cp "$installed_manifest" "$staging_root/kapsl-sglang-backend.json"

if [ -e "$target_root" ]; then
  backup_root="$(mktemp -d "$target_parent/.${target_name}.previous.XXXXXX")"
  rmdir "$backup_root"
  mv "$target_root" "$backup_root"
fi
mv "$staging_root" "$target_root"
staging_root=""
if [ -n "$backup_root" ] && [ -d "$backup_root" ]; then
  rm -rf "$backup_root"
fi
backup_root=""
trap - EXIT INT TERM

echo "Installed certified managed-SGLang backend at $target_root"
