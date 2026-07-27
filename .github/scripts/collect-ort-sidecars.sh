#!/usr/bin/env bash
set -euo pipefail

rm -rf ort-core-libs ort-cuda-libs ort-tensorrt-libs ort-rocm-libs
mkdir -p ort-core-libs ort-cuda-libs ort-tensorrt-libs ort-rocm-libs

search_roots=(
  "kapsl-runtime/target/release"
  "kapsl-runtime/target/release/deps"
  "${HOME}/.cache/ort.pyke.io/dfbin"
  "${HOME}/Library/Caches/ort.pyke.io/dfbin"
)

existing_roots=()
for root in "${search_roots[@]}"; do
  if [ -d "$root" ]; then
    existing_roots+=("$root")
  fi
done

if [ "${#existing_roots[@]}" -eq 0 ]; then
  echo "No ONNX Runtime search roots found; continuing without sidecar libraries."
  exit 0
fi

while IFS= read -r lib; do
  [ -n "$lib" ] || continue
  name="$(basename "$lib")"
  case "$name" in
    *onnxruntime_providers_cuda*)
      destination="ort-cuda-libs"
      ;;
    *onnxruntime_providers_tensorrt*)
      destination="ort-tensorrt-libs"
      ;;
    *onnxruntime_providers_rocm*)
      destination="ort-rocm-libs"
      ;;
    *)
      destination="ort-core-libs"
      ;;
  esac
  echo "Staging ONNX Runtime sidecar for ${destination}: ${lib}"
  cp -L "$lib" "${destination}/${name}"
done < <(
  find "${existing_roots[@]}" -type f \( \
    -name 'libonnxruntime*.so' -o \
    -name 'libonnxruntime*.so.*' -o \
    -name 'libonnxruntime*.dylib' -o \
    -name 'libonnxruntime*.dylib.*' \
  \) 2>/dev/null | sort -u
)

if [ "${RUNNER_OS:-}" = "Linux" ] && [ "${RUNNER_ARCH:-}" = "X64" ]; then
  if [ ! -f "ort-core-libs/libonnxruntime_providers_shared.so" ]; then
    echo "Missing required core ONNX Runtime sidecar: libonnxruntime_providers_shared.so" >&2
    exit 1
  fi
fi

for directory in ort-core-libs ort-cuda-libs ort-tensorrt-libs ort-rocm-libs; do
  if [ -n "$(find "$directory" -type f -print -quit)" ]; then
    find "$directory" -maxdepth 1 -type f -print | sort
  else
    echo "No sidecars staged in ${directory}."
  fi
done
