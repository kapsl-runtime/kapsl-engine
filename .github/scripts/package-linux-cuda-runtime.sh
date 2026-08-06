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

# Read the dynamic dependencies once. Piping readelf straight into `grep -q`
# races against pipefail: grep exits on the first match, readelf takes SIGPIPE,
# and the pipeline reports 141 even though the library is present. It only
# passes today because readelf's output is small enough to flush first.
needed_libs="$(readelf -d "$binary" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p')"

# The accelerator image must never silently receive the portable build again.
# libcuda is provided by the NVIDIA driver at container runtime, so inspect the
# dependency without requiring a GPU or driver on the release runner.
case "$needed_libs" in
  *libcuda.so.1*) ;;
  *)
    echo "CUDA runtime binary does not depend on libcuda.so.1; was it built with --features cuda?" >&2
    exit 1
    ;;
esac

bundle_name="kapsl-${KAPSL_VERSION}-linux-x86_64-cuda12"
bundle_root="${RUNNER_TEMP}/${bundle_name}"
mkdir -p dist "$bundle_root"
install -m 755 "$binary" "$bundle_root/kapsl"

if [ -d ort-core-libs ]; then
  cp -L ort-core-libs/* "$bundle_root/" 2>/dev/null || true
fi

# Everything the binary needs that a bare GPU host will not already have has to
# travel with it, next to the binary, where the $ORIGIN RUNPATH already looks.
#
# libcuda.so.1 is deliberately excluded: it ships with the NVIDIA driver and
# has to match it, so bundling a copy would break the host. The base C/C++
# runtime is assumed present, exactly as for the portable build. What is left
# is the CUDA-stack libraries this devel build image happens to carry but a
# driver-only host does not — libnccl.so.2 today, linked because ggml turns on
# GGML_CUDA_NCCL whenever CMake finds NCCL. That one is why a driver check
# alone never proved the binary could actually start: it installed fine and
# then exited 127 on every run.
resolved_deps="$(ldd "$binary")"
bundled_libs=()
unresolved_libs=()
while read -r dep; do
  [ -n "$dep" ] || continue
  case "$dep" in
    libcuda.so.* | libc.so.* | libm.so.* | libdl.so.* | librt.so.* | \
      libpthread.so.* | libgcc_s.so.* | libstdc++.so.* | ld-linux*)
      continue
      ;;
  esac

  # Fed by here-string, not a pipe: awk exits at the first match, which would
  # SIGPIPE a writer on the other end and trip pipefail.
  resolved="$(awk -v dep="$dep" '$1 == dep { print $3; exit }' <<< "$resolved_deps")"
  if [ -z "$resolved" ] || [ ! -e "$resolved" ]; then
    unresolved_libs+=("$dep")
    continue
  fi

  cp -L "$resolved" "$bundle_root/"
  bundled_libs+=("$dep")
done <<< "$needed_libs"

if [ "${#unresolved_libs[@]}" -gt 0 ]; then
  echo "Cannot resolve these dependencies to bundle: ${unresolved_libs[*]}" >&2
  echo "They would be missing on any host that does not already provide them." >&2
  exit 1
fi

# Shipping libcuda would pin users to this image's driver version.
if [ -e "$bundle_root/libcuda.so.1" ]; then
  echo "libcuda.so.1 must not be bundled; it belongs to the host driver." >&2
  exit 1
fi

if [ "${#bundled_libs[@]}" -gt 0 ]; then
  echo "Bundled host-missing libraries: ${bundled_libs[*]}"
fi

tar_path="dist/${bundle_name}.tar.gz"
tar -C "$bundle_root" -czf "$tar_path" .
sha256sum "$tar_path" > "${tar_path}.sha256"

find dist -maxdepth 1 -type f -name "${bundle_name}*" -print | sort
