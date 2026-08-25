#!/usr/bin/env bash
set -euo pipefail

if [ "${RUNNER_OS:-Linux}" != "Linux" ] || [ "${RUNNER_ARCH:-X64}" != "X64" ]; then
  echo "TensorRT 10 runtime collection is supported only on Linux x86_64." >&2
  exit 1
fi

python_bin="${PYTHON:-python3}"
version="${KAPSL_TENSORRT_PYPI_VERSION:-10.16.1.11}"
runtime_output="${KAPSL_TENSORRT_RUNTIME_DIR:-tensorrt-runtime-libs}"
license_output="${KAPSL_TENSORRT_LICENSE_DIR:-tensorrt-license-files}"
if [ -e "$runtime_output" ] || [ -e "$license_output" ]; then
  echo "TensorRT output already exists; refusing to overwrite $runtime_output or $license_output" >&2
  exit 1
fi

scratch="$(mktemp -d "${RUNNER_TEMP:-/tmp}/kapsl-tensorrt.XXXXXX")"
cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT INT TERM

PIP_BREAK_SYSTEM_PACKAGES=1 "$python_bin" -m pip install \
  --disable-pip-version-check \
  --no-compile \
  --only-binary=:all: \
  --extra-index-url https://pypi.nvidia.com \
  --target "$scratch/site" \
  "tensorrt-cu12-libs==${version}"

mkdir -p "$runtime_output" "$license_output"
"$python_bin" - "$scratch/site" "$runtime_output" "$license_output" <<'PY'
import hashlib
import pathlib
import shutil
import sys

source = pathlib.Path(sys.argv[1])
runtime = pathlib.Path(sys.argv[2])
licenses = pathlib.Path(sys.argv[3])

def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()

libraries = sorted(
    path for path in source.rglob("*")
    if path.is_file() and ".so" in path.name
)
if not libraries:
    raise SystemExit("TensorRT PyPI package contained no shared libraries")
for path in libraries:
    destination = runtime / path.name
    if destination.exists():
        if digest(path) != digest(destination):
            raise SystemExit(f"conflicting TensorRT library basename: {path.name}")
        continue
    shutil.copy2(path, destination)

license_names = {"license", "license.txt", "third_party_notices", "third_party_notices.txt"}
license_files = sorted(
    path for path in source.rglob("*")
    if path.is_file()
    and (
        path.name.lower() in license_names
        or path.name.lower().startswith("license.")
        or path.name.lower().startswith("third_party_notices.")
    )
)
for position, path in enumerate(license_files, 1):
    shutil.copy2(path, licenses / f"{position:03}-{path.name}")
PY

if ! find "$runtime_output" -maxdepth 1 -type f -name 'libnvinfer.so*' -print -quit | grep -q .; then
  echo "TensorRT runtime closure is missing libnvinfer.so" >&2
  exit 1
fi
if ! find "$runtime_output" -maxdepth 1 -type f -name 'libnvonnxparser.so*' -print -quit | grep -q .; then
  echo "TensorRT runtime closure is missing libnvonnxparser.so" >&2
  exit 1
fi

find "$runtime_output" -maxdepth 1 -type f -print | sort
