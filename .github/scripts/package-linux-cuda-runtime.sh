#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"

if [ "${RUNNER_OS:-}" != "Linux" ] || [ "${RUNNER_ARCH:-}" != "X64" ]; then
  echo "The CUDA runtime bundle is supported only on Linux x86_64." >&2
  exit 1
fi

binary="kapsl-runtime/target/release/kapsl"
if [ ! -x "$binary" ]; then
  echo "Missing CUDA runtime binary: $binary" >&2
  exit 1
fi

# The accelerator image must never silently receive the portable build again.
# libcuda is provided by the NVIDIA driver at container runtime, so inspect the
# dependency without requiring a GPU or driver on the release runner.
if ! readelf -d "$binary" | grep -q 'libcuda.so.1'; then
  echo "CUDA runtime binary does not depend on libcuda.so.1; was it built with --features cuda?" >&2
  exit 1
fi

bundle_name="kapsl-${KAPSL_VERSION}-linux-x86_64-cuda12"
bundle_root="${RUNNER_TEMP}/${bundle_name}"
mkdir -p dist "$bundle_root"
install -m 755 "$binary" "$bundle_root/kapsl"

if [ -d ort-core-libs ]; then
  cp -L ort-core-libs/* "$bundle_root/" 2>/dev/null || true
fi

tar_path="dist/${bundle_name}.tar.gz"
tar -C "$bundle_root" -czf "$tar_path" .
sha256sum "$tar_path" > "${tar_path}.sha256"

find dist -maxdepth 1 -type f -name "${bundle_name}*" -print | sort
