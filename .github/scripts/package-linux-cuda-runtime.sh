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

cuda_provider="ort-cuda-libs/libonnxruntime_providers_cuda.so"
shared_provider="ort-core-libs/libonnxruntime_providers_shared.so"
for required_sidecar in "$cuda_provider" "$shared_provider"; do
  if [ ! -f "$required_sidecar" ]; then
    echo "Missing ONNX Runtime CUDA bundle sidecar: $required_sidecar" >&2
    exit 1
  fi
done

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
cp -L ort-core-libs/* "$bundle_root/"
cp -L ort-cuda-libs/* "$bundle_root/"

nvidia_license="${KAPSL_NVIDIA_LICENSE_FILE:-/NGC-DL-CONTAINER-LICENSE}"
if [ ! -f "$nvidia_license" ]; then
  echo "Missing NVIDIA redistribution license: $nvidia_license" >&2
  exit 1
fi
cp "$nvidia_license" "$bundle_root/NVIDIA-CONTAINER-LICENSE"

# cuDNN 9 loads its split libraries at runtime rather than declaring all of
# them as ELF dependencies, so ldd cannot discover the complete family from
# libcudnn.so.9 alone. Treat every runtime component as a dependency root.
mapfile -t cudnn_libs < <(
  find /usr \( -type f -o -type l \) 2>/dev/null \
    | grep '/libcudnn[^/]*\.so\.9$' \
    | sort -u
)
if [ "${#cudnn_libs[@]}" -eq 0 ]; then
  echo "The CUDA bundle cannot find cuDNN 9; use a cuDNN release image." >&2
  exit 1
fi

# Everything the GGUF CUDA binary and ONNX CUDA provider need that a bare GPU
# host will not already have has to travel with them, next to the binary, where
# the $ORIGIN RUNPATH already looks.
#
# libcuda.so.1 and libnvidia-* are deliberately excluded: they ship with the
# NVIDIA driver and have to match it, so bundling copies would break the host.
# The base C/C++ runtime is assumed present, exactly as for the portable build.
# The release image supplies the remaining CUDA, cuDNN, and NCCL libraries, making this one
# archive the Linux equivalent of a self-contained Triton GPU image.
resolved_deps="$({
  ldd "$binary"
  ldd "$cuda_provider"
  for cudnn_lib in "${cudnn_libs[@]}"; do
    ldd "$cudnn_lib"
  done
})"
bundled_libs=()
unresolved_libs=()
while IFS= read -r dep; do
  [ -n "$dep" ] || continue
  case "$dep" in
    libcuda.so.* | libnvidia-*.so.* | libc.so.* | libm.so.* | libdl.so.* | librt.so.* | \
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

  # Preserve the requested SONAME even when the resolved file is a more
  # specific version; that is the name the loader will look for at runtime.
  cp -L "$resolved" "$bundle_root/$dep"
  bundled_libs+=("$dep")
done < <(
  {
    printf '%s\n' "$needed_libs"
    printf '%s\n' "$resolved_deps" | sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>.*/\1/p'
  } | sort -u
)

if [ "${#unresolved_libs[@]}" -gt 0 ]; then
  echo "Cannot resolve these dependencies to bundle: ${unresolved_libs[*]}" >&2
  echo "They would be missing on any host that does not already provide them." >&2
  exit 1
fi

# Shipping driver libraries would pin users to this image's driver version.
driver_library="$(find "$bundle_root" -maxdepth 1 -type f \
  \( -name 'libcuda.so*' -o -name 'libnvidia-*.so*' \) -print -quit)"
if [ -n "$driver_library" ]; then
  echo "NVIDIA driver libraries must not be bundled; they belong to the host." >&2
  exit 1
fi

if [ "${#bundled_libs[@]}" -gt 0 ]; then
  echo "Bundled host-missing libraries: ${bundled_libs[*]}"
fi

# Ship every cuDNN runtime component under its loader-visible SONAME.
for cudnn_lib in "${cudnn_libs[@]}"; do
  cudnn_name="$(basename "$cudnn_lib")"
  cp -L "$cudnn_lib" "$bundle_root/$cudnn_name"
done

if [ ! -f "$bundle_root/libcudnn.so.9" ]; then
  echo "The CUDA bundle is missing libcudnn.so.9; use a cuDNN release image." >&2
  exit 1
fi

# ORT's published CUDA provider carries a build-machine RUNPATH, while a bare
# install puts all libraries beside kapsl. Normalize every bundled ELF to look
# in its own directory so the archive works outside a container or ldconfig
# setup as well.
if ! command -v patchelf >/dev/null 2>&1; then
  echo "patchelf is required to make the CUDA archive relocatable." >&2
  exit 1
fi
while IFS= read -r elf_file; do
  patchelf --set-rpath '$ORIGIN' "$elf_file"
done < <(find "$bundle_root" -maxdepth 1 -type f \( -name kapsl -o -name 'lib*.so*' \) | sort)

# This marker activates the ONNX CUDA execution provider. Keeping it in the
# same archive as the CUDA-compiled binary is what guarantees that one install
# accelerates both ONNX and GGUF models.
provider_files=""
while IFS= read -r provider_file; do
  escaped="$(basename "$provider_file" | sed 's/\\/\\\\/g; s/"/\\"/g')"
  if [ -n "$provider_files" ]; then
    provider_files="${provider_files},"
  fi
  provider_files="${provider_files}\"${escaped}\""
done < <(find "$bundle_root" -maxdepth 1 -type f ! -name kapsl | sort)

cat > "$bundle_root/kapsl-provider-cuda12.json" <<EOF
{
  "schema_version": 1,
  "provider": "cuda",
  "provider_version": "12",
  "runtime_version": "${KAPSL_VERSION}",
  "platform": "linux-x86_64",
  "requires": [],
  "files": [${provider_files}],
  "system_requirements": "A compatible NVIDIA driver must be installed."
}
EOF

for required_bundle_file in \
  kapsl \
  kapsl-provider-cuda12.json \
  libonnxruntime_providers_cuda.so \
  libonnxruntime_providers_shared.so \
  libcudnn.so.9; do
  if [ ! -f "$bundle_root/$required_bundle_file" ]; then
    echo "Incomplete CUDA bundle: missing $required_bundle_file" >&2
    exit 1
  fi
done

cuda_provider_runpath="$(patchelf --print-rpath "$bundle_root/libonnxruntime_providers_cuda.so")"
if [ "$cuda_provider_runpath" != '$ORIGIN' ]; then
  echo "CUDA provider is not relocatable: RUNPATH=${cuda_provider_runpath}" >&2
  exit 1
fi

tar_path="dist/${bundle_name}.tar.gz"
tar -C "$bundle_root" -czf "$tar_path" .
sha256sum "$tar_path" > "${tar_path}.sha256"

find dist -maxdepth 1 -type f -name "${bundle_name}*" -print | sort
