#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"
: "${KAPSL_VLLM_SDK_REF:?KAPSL_VLLM_SDK_REF is required and must be an exact 40-hex commit}"

if [ "${RUNNER_OS:-}" != "Linux" ] || [ "${RUNNER_ARCH:-}" != "X64" ]; then
  echo "The managed-vLLM backend bundle is supported only on Linux x86_64." >&2
  exit 1
fi

sdk_dir="${KAPSL_VLLM_SDK_DIR:-sdk-vllm}"
sdk_ref="$KAPSL_VLLM_SDK_REF"
connector_version="0.6.0"
connector_profile="vllm-v1-packed-cuda-ipc/flash-attn"
planner_schema_version="1"
connector_root="$sdk_dir/integrations/vllm"
requirements_lock=".github/scripts/managed-vllm-cu130.lock"
.github/scripts/verify-managed-vllm-sdk-checkout.sh "$sdk_dir" "$sdk_ref"
if [ ! -f "$connector_root/pyproject.toml" ]; then
  echo "Missing kapsl-vLLM connector checkout at $connector_root" >&2
  exit 1
fi
if [ ! -f "$requirements_lock" ]; then
  echo "Missing managed-vLLM dependency lock at $requirements_lock" >&2
  exit 1
fi
python_archive_name="cpython-3.12.3+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
python_archive_url="https://github.com/astral-sh/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
python_archive_sha256="a73ba777b5d55ca89edef709e6b8521e3f3d4289581f174c8699adfb608d09d6"
vllm_wheel_name="vllm-0.26.1rc1.dev1130+g2ec6f0d71-cp38-abi3-manylinux_2_28_x86_64.whl"
vllm_wheel_url="https://wheels.vllm.ai/2ec6f0d71ea3b350952630e310efcda1c744ff4d/vllm-0.26.1rc1.dev1130%2Bg2ec6f0d71-cp38-abi3-manylinux_2_28_x86_64.whl"
vllm_wheel_sha256="3bc943a7ba18c547d8777b14c15640b6f4d8f7bd268a4cec76a1fbbf8d8d3c70"
pytorch_index_url="https://download.pytorch.org/whl/cu130"

runner_temp="${RUNNER_TEMP:-/tmp}"
build_root="$(mktemp -d "$runner_temp/kapsl-vllm-package.XXXXXX")"
cleanup() {
  rm -rf "$build_root"
}
trap cleanup EXIT INT TERM

downloads="$build_root/downloads"
payload="$build_root/payload/backends/vllm-bootstrap"
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
"$bootstrap_python" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  "setuptools==80.10.2" \
  "wheel==0.45.1"

vllm_wheel="$downloads/$vllm_wheel_name"
curl --fail --location --retry 3 --output "$vllm_wheel" "$vllm_wheel_url"
printf '%s  %s\n' "$vllm_wheel_sha256" "$vllm_wheel" | sha256sum --check -

"$bootstrap_python" -m pip wheel \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "$payload/wheels" \
  "$connector_root"

shopt -s nullglob
connector_wheels=("$payload/wheels/kapsl_vllm_connector-${connector_version}-"*.whl)
shopt -u nullglob
if [ "${#connector_wheels[@]}" -ne 1 ]; then
  echo "Expected exactly one kapsl-vLLM connector $connector_version wheel, found ${#connector_wheels[@]}" >&2
  exit 1
fi
connector_wheel="${connector_wheels[0]}"
.github/scripts/verify-managed-vllm-connector-wheel.py \
  --wheel "$connector_wheel" \
  --connector-version "$connector_version" \
  --profile "$connector_profile" \
  --planner-schema "$planner_schema_version"

"$bootstrap_python" -m pip download \
  --disable-pip-version-check \
  --no-cache-dir \
  --only-binary=:all: \
  --constraint "$requirements_lock" \
  --extra-index-url "$pytorch_index_url" \
  --dest "$payload/wheels" \
  "$vllm_wheel" \
  "torch==2.13.0+cu130" \
  "torchvision==0.28.0+cu130" \
  "torchaudio==2.11.0+cu130"

for required_glob in \
  "kapsl_vllm_connector-${connector_version}-*.whl" \
  'torch-2.13.0+cu130-*.whl' \
  'torchvision-0.28.0+cu130-*.whl' \
  'torchaudio-2.11.0+cu130-*.whl' \
  'vllm-0.26.1rc1.dev1130+g2ec6f0d71-*.whl'; do
  if ! compgen -G "$payload/wheels/$required_glob" >/dev/null; then
    echo "Managed-vLLM wheelhouse is missing $required_glob" >&2
    exit 1
  fi
done

# The pinned wheel is now present in the payload and standalone Python has
# already been extracted. Remove duplicate downloads before constructing the
# multi-gigabyte archive so hosted runners retain bounded scratch usage.
rm -f "$python_archive" "$vllm_wheel"

cp .github/scripts/bootstrap-vllm-backend.sh "$payload/bootstrap.sh"
cp "$requirements_lock" "$payload/requirements.lock"
chmod 755 "$payload/bootstrap.sh"
requirements_lock_sha256="$(sha256sum "$requirements_lock" | awk '{ print $1 }')"
cat > "$payload/installed-manifest.json" <<EOF
{
  "schema_version": 1,
  "sdk_ref": "$sdk_ref",
  "python": "3.12.3",
  "torch": "2.13.0+cu130",
  "torchvision": "0.28.0+cu130",
  "torchaudio": "2.11.0+cu130",
  "cuda_runtime": "13.0",
  "vllm": "0.26.1rc1.dev1130+g2ec6f0d71",
  "connector_distribution": "$connector_version",
  "connector": "$connector_version",
  "profile": "$connector_profile",
  "planner_schema_version": $planner_schema_version
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
  "sdk_ref": "$sdk_ref",
  "requirements_lock_sha256": "$requirements_lock_sha256",
  "python": "3.12.3",
  "python_archive_sha256": "$python_archive_sha256",
  "torch": "2.13.0+cu130",
  "torchvision": "0.28.0+cu130",
  "torchaudio": "2.11.0+cu130",
  "cuda_runtime": "13.0",
  "vllm": "0.26.1rc1.dev1130+g2ec6f0d71",
  "vllm_wheel_sha256": "$vllm_wheel_sha256",
  "connector_distribution": "$connector_version",
  "connector": "$connector_version",
  "profile": "$connector_profile",
  "planner_schema_version": $planner_schema_version
}
EOF

cat > "$payload/backend-pack.json" <<EOF
{
  "schema_version": 1,
  "backend": "vllm",
  "profile": "cu130-flash-attn",
  "pack_version": "0.26.1rc1.dev1130+g2ec6f0d71",
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

archive_name="kapsl-backend-vllm-${KAPSL_VERSION}-linux-x86_64.tar.gz"
archive_path="dist/$archive_name"
# Wheels are already compressed, so gzip level 1 is materially faster with
# negligible size cost. GNU tar removes each temporary payload file after it
# has entered the archive, keeping peak disk close to one bundle rather than
# retaining both a full wheelhouse and full tarball.
tar --remove-files -C "$build_root/payload" -I 'gzip -1' -cf "$archive_path" .
(cd dist && sha256sum "$archive_name" > "$archive_name.sha256")

cat > "dist/${archive_name}.manifest.json" <<EOF
{
  "schema_version": 1,
  "backend": "vllm",
  "profile": "cu130-flash-attn",
  "pack_version": "0.26.1rc1.dev1130+g2ec6f0d71",
  "runtime_abi": 1,
  "compatible_kapsl": "=$KAPSL_VERSION",
  "platform": "linux-x86_64",
  "architecture": "x86_64",
  "accelerator_profile": "cuda",
  "minimum_cuda": "13.0",
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
    "kapsl-vllm-backend.json": "$installed_manifest_sha256",
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
