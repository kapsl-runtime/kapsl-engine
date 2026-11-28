#!/usr/bin/env bash
set -euo pipefail

# Package the certified managed-SGLang backend: a standalone Python, an offline
# wheelhouse resolved from .github/scripts/managed-sglang-cu130.lock, and the
# Kapsl SGLang adapter built from the exact kapsl-integrations commit in
# .github/sglang-integration.lock. Kapsl installs it with bootstrap.sh.

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"
: "${KAPSL_SGLANG_INTEGRATIONS_REF:?KAPSL_SGLANG_INTEGRATIONS_REF is required and must be an exact 40-hex commit}"

if [ "${RUNNER_OS:-}" != "Linux" ] || [ "${RUNNER_ARCH:-}" != "X64" ]; then
  echo "The managed-SGLang backend bundle is supported only on Linux x86_64." >&2
  exit 1
fi

integrations_dir="${KAPSL_SGLANG_INTEGRATIONS_DIR:-kapsl-integrations-sglang}"
integrations_ref="$KAPSL_SGLANG_INTEGRATIONS_REF"
adapter_version="0.2.0"
kv_abi_major="1"
kv_abi_minor="5"
participant_config_env="KAPSL_KV_PARTICIPANT_CONFIG"
sglang_version="0.5.23"
torch_version="2.14.1"
torchvision_version="0.29.2"
torchaudio_version="2.11.0"
cuda_runtime="13.0"
pack_profile="cu130"
adapter_root="$integrations_dir/integrations/sglang"
requirements_lock=".github/scripts/managed-sglang-cu130.lock"
.github/scripts/verify-sglang-integration-checkout.sh "$integrations_dir" "$integrations_ref"
if [ ! -f "$adapter_root/pyproject.toml" ]; then
  echo "Missing Kapsl SGLang adapter checkout at $adapter_root" >&2
  exit 1
fi
if [ ! -f "$requirements_lock" ]; then
  echo "Missing managed-SGLang dependency lock at $requirements_lock" >&2
  exit 1
fi
# The same pinned standalone Python as the managed-vLLM pack. SGLang
# 0.5.23 requires Python 3.11 or newer.
python_archive_name="cpython-3.12.3+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
python_archive_url="https://github.com/astral-sh/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
python_archive_sha256="a73ba777b5d55ca89edef709e6b8521e3f3d4289581f174c8699adfb608d09d6"

runner_temp="${RUNNER_TEMP:-/tmp}"
build_root="$(mktemp -d "$runner_temp/kapsl-sglang-package.XXXXXX")"
cleanup() {
  rm -rf "$build_root"
}
trap cleanup EXIT INT TERM

downloads="$build_root/downloads"
payload="$build_root/payload/backends/sglang-bootstrap"
python_extract="$build_root/python-extract"
mkdir -p "$downloads" "$payload/wheels" "$python_extract" dist
mkdir -p "$payload/licenses"
cp LICENSE NOTICE "$payload/licenses/"

python_archive="$downloads/$python_archive_name"
curl --fail --location --retry 3 --output "$python_archive" "$python_archive_url"
printf '%s  %s\n' "$python_archive_sha256" "$python_archive" | sha256sum --check -
tar -xzf "$python_archive" -C "$python_extract"
if [ ! -x "$python_extract/python/bin/python3.12" ]; then
  echo "The pinned standalone Python archive has an unexpected layout." >&2
  exit 1
fi
mv "$python_extract/python" "$payload/python"
ln -sfn python3.12 "$payload/python/bin/python3"
ln -sfn python3.12 "$payload/python/bin/python"

bootstrap_python="$payload/python/bin/python"
"$bootstrap_python" -m ensurepip --upgrade
# Whatever is installed here ships in the pack, so it carries the certified
# lock's pins. The adapter build needs only setuptools; setuptools >= 70.1
# builds wheels without the separate `wheel` package.
"$bootstrap_python" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  "setuptools==84.2.1"

"$bootstrap_python" -m pip wheel \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "$payload/wheels" \
  "$adapter_root"

shopt -s nullglob
adapter_wheels=("$payload/wheels/kapsl_sglang_adapter-${adapter_version}-"*.whl)
shopt -u nullglob
if [ "${#adapter_wheels[@]}" -ne 1 ]; then
  echo "Expected exactly one Kapsl SGLang adapter $adapter_version wheel, found ${#adapter_wheels[@]}" >&2
  exit 1
fi
.github/scripts/verify-managed-sglang-adapter-wheel.py \
  --wheel "${adapter_wheels[0]}" \
  --adapter-version "$adapter_version" \
  --sglang-version "$sglang_version" \
  --kv-abi-major "$kv_abi_major" \
  --kv-abi-minor "$kv_abi_minor"

# SGLang and its CUDA 13 wheels come from PyPI. The lock pins every package
# of the environment that passed the GPU certification run.
"$bootstrap_python" -m pip download \
  --disable-pip-version-check \
  --no-cache-dir \
  --only-binary=:all: \
  --constraint "$requirements_lock" \
  --dest "$payload/wheels" \
  "sglang==${sglang_version}" \
  "torch==${torch_version}" \
  "torchvision==${torchvision_version}" \
  "torchaudio==${torchaudio_version}"

for required_glob in \
  "kapsl_sglang_adapter-${adapter_version}-*.whl" \
  "sglang-${sglang_version}-*.whl" \
  "torch-${torch_version}-*.whl" \
  "torchvision-${torchvision_version}-*.whl" \
  "torchaudio-${torchaudio_version}-*.whl"; do
  if ! compgen -G "$payload/wheels/$required_glob" >/dev/null; then
    echo "Managed-SGLang wheelhouse is missing $required_glob" >&2
    exit 1
  fi
done

rm -f "$python_archive"

cp .github/scripts/bootstrap-sglang-backend.sh "$payload/bootstrap.sh"
cp "$requirements_lock" "$payload/requirements.lock"
chmod 755 "$payload/bootstrap.sh"
requirements_lock_sha256="$(sha256sum "$requirements_lock" | awk '{ print $1 }')"
cat > "$payload/installed-manifest.json" <<EOF
{
  "schema_version": 1,
  "integrations_ref": "$integrations_ref",
  "python": "3.12.3",
  "torch": "$torch_version",
  "torchvision": "$torchvision_version",
  "torchaudio": "$torchaudio_version",
  "cuda_runtime": "$cuda_runtime",
  "sglang": "$sglang_version",
  "adapter_distribution": "$adapter_version",
  "adapter": "$adapter_version",
  "kv_abi": {"major": $kv_abi_major, "minor": $kv_abi_minor},
  "participant_config_env": "$participant_config_env"
}
EOF
(cd "$payload" && {
  find python wheels -type f -print0
  printf 'requirements.lock\0installed-manifest.json\0'
} | sort -z | xargs -0 sha256sum > SHA256SUMS)

cat > "$payload/manifest.json" <<EOF
{
  "schema_version": 1,
  "kapsl_runtime_version": "$KAPSL_VERSION",
  "integrations_ref": "$integrations_ref",
  "requirements_lock_sha256": "$requirements_lock_sha256",
  "python": "3.12.3",
  "python_archive_sha256": "$python_archive_sha256",
  "torch": "$torch_version",
  "torchvision": "$torchvision_version",
  "torchaudio": "$torchaudio_version",
  "cuda_runtime": "$cuda_runtime",
  "sglang": "$sglang_version",
  "adapter_distribution": "$adapter_version",
  "adapter": "$adapter_version",
  "kv_abi": {"major": $kv_abi_major, "minor": $kv_abi_minor}
}
EOF

cat > "$payload/backend-pack.json" <<EOF
{
  "schema_version": 1,
  "backend": "sglang",
  "profile": "$pack_profile",
  "pack_version": "$sglang_version",
  "runtime_abi": 1,
  "platform": "linux-x86_64",
  "execution_mode": "external",
  "entrypoint": "bin/python"
}
EOF

installed_bytes="$($bootstrap_python - "$payload/python" "$payload/wheels" <<'PY'
import pathlib
import sys
import zipfile

python_root = pathlib.Path(sys.argv[1])
wheel_root = pathlib.Path(sys.argv[2])
total = sum(path.stat().st_size for path in python_root.rglob("*") if path.is_file())
for wheel in wheel_root.glob("*.whl"):
    with zipfile.ZipFile(wheel) as archive:
        total += sum(item.file_size for item in archive.infolist() if not item.is_dir())
print(total)
PY
)"
python_entrypoint_sha256="$(sha256sum "$payload/python/bin/python3.12" | awk '{ print $1 }')"
installed_manifest_sha256="$(sha256sum "$payload/installed-manifest.json" | awk '{ print $1 }')"
license_sha256="$(sha256sum "$payload/licenses/LICENSE" | awk '{ print $1 }')"
notice_sha256="$(sha256sum "$payload/licenses/NOTICE" | awk '{ print $1 }')"

archive_name="kapsl-backend-sglang-${KAPSL_VERSION}-linux-x86_64.tar.gz"
archive_path="dist/$archive_name"
# Wheels are already compressed; gzip level 1 is much faster at almost no
# size cost, and --remove-files keeps peak disk close to one bundle.
tar --remove-files -C "$build_root/payload" -I 'gzip -1' -cf "$archive_path" .
(cd dist && sha256sum "$archive_name" > "$archive_name.sha256")

cat > "dist/${archive_name}.manifest.json" <<EOF
{
  "schema_version": 1,
  "backend": "sglang",
  "profile": "$pack_profile",
  "pack_version": "$sglang_version",
  "runtime_abi": 1,
  "compatible_kapsl": "=$KAPSL_VERSION",
  "platform": "linux-x86_64",
  "architecture": "x86_64",
  "accelerator_profile": "cuda",
  "minimum_cuda": "$cuda_runtime",
  "minimum_driver": "580.65.06",
  "execution_mode": "external",
  "entrypoint": "bin/python",
  "installed_bytes": $installed_bytes,
  "memory": {
    "host_bytes": 536870912,
    "accelerator_bytes": 0,
    "workspace_weight_ppm": 125000,
    "minimum_workspace_bytes": 268435456
  },
  "installer": {
    "kind": "bootstrap",
    "path": "bootstrap.sh"
  },
  "files": {
    "bin/python": "$python_entrypoint_sha256",
    "kapsl-sglang-backend.json": "$installed_manifest_sha256",
    "licenses/LICENSE": "$license_sha256",
    "licenses/NOTICE": "$notice_sha256"
  },
  "licenses": [
    {"name": "Kapsl", "path": "licenses/LICENSE"},
    {"name": "Kapsl notices", "path": "licenses/NOTICE"}
  ],
  "priority": 100
}
EOF

echo "Packaged $archive_path"
du -h "$archive_path"
