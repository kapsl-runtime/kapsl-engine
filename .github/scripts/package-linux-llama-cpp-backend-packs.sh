#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"

host_os="${RUNNER_OS:-$(uname -s)}"
host_arch="${RUNNER_ARCH:-$(uname -m)}"
if [ "$host_os" != "Linux" ] || { [ "$host_arch" != "X64" ] && [ "$host_arch" != "x86_64" ]; }; then
  echo "llama.cpp native packs are currently published only for Linux x86_64." >&2
  exit 1
fi

for command_name in cargo file ldd nm patchelf python3 sha256sum tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to package llama.cpp backend packs." >&2
    exit 1
  fi
done

header="kapsl-runtime/include/kapsl_llama_cpp_backend.h"
llama_license=".github/licenses/LLAMA-CPP-LICENSE"
for required in "$header" "$llama_license" LICENSE NOTICE kapsl-runtime/Cargo.toml; do
  if [ ! -f "$required" ]; then
    echo "Missing required llama.cpp pack input: $required" >&2
    exit 1
  fi
done

work_parent="${RUNNER_TEMP:-/tmp}"
work_root="$(mktemp -d "${work_parent%/}/kapsl-llama-packs.XXXXXX")"
cleanup() {
  rm -rf "$work_root"
}
trap cleanup EXIT INT TERM
mkdir -p dist

build_profile() {
  profile="$1"
  feature="$2"
  override="$3"
  kv_mode="$4"
  if [ -n "$override" ]; then
    if [ ! -f "$override" ]; then
      echo "Configured $profile llama.cpp library does not exist: $override" >&2
      exit 1
    fi
    printf '%s\n' "$override"
    return
  fi

  target_dir="$work_root/target-$profile"
  cargo_args=(
    build
    --manifest-path kapsl-runtime/Cargo.toml
    -p kapsl-backend-llama-cpp
    --release
    --locked
    --no-default-features
    --features "$feature"
    --target-dir "$target_dir"
  )
  # The backend entrypoint is a cdylib which statically links llama.cpp's
  # C/CUDA archives. Force every CMake target (including nvcc objects) to be
  # position independent so the final shared object is linkable on Linux.
  CMAKE_POSITION_INDEPENDENT_CODE=ON cargo "${cargo_args[@]}"
  printf '%s\n' "$target_dir/release/libkapsl_backend_llama_cpp.so"
}

copy_runtime_dependencies() {
  library="$1"
  destination="$2"
  accelerator="$3"
  resolved_deps="$(ldd "$library")"
  while IFS= read -r row; do
    dep="$(awk '{print $1}' <<< "$row")"
    marker="$(awk '{print $2}' <<< "$row")"
    resolved="$(awk '{print $3}' <<< "$row")"
    [ "$marker" = "=>" ] || continue
    if [ "$accelerator" = "cpu" ]; then
      case "$dep" in
        libcudart.so* | libcublas.so* | libcublasLt.so* | libcufft.so* | \
          libcurand.so* | libcusolver.so* | libcusparse.so* | libnvrtc.so* | \
          libnvJitLink.so* | libnccl.so*)
          echo "CPU llama.cpp pack unexpectedly depends on CUDA library $dep" >&2
          exit 1
          ;;
      esac
    fi
    case "$dep" in
      libcuda.so* | libnvidia-*.so*)
        if [ "$accelerator" != "cuda" ]; then
          echo "CPU llama.cpp pack unexpectedly depends on NVIDIA driver library $dep" >&2
          exit 1
        fi
        # Driver libraries belong to the host and must match its driver.
        continue
        ;;
      libc.so* | libm.so* | libdl.so* | librt.so* | libpthread.so* | \
        libgcc_s.so* | libstdc++.so* | ld-linux*)
        # Use the deployment platform's baseline C/C++ runtime.
        continue
        ;;
    esac
    if [ "$resolved" = "not" ] || [ ! -f "$resolved" ]; then
      echo "Cannot resolve llama.cpp CUDA dependency $dep from: $row" >&2
      exit 1
    fi
    if [ -f "$destination/$dep" ] && ! cmp -s "$resolved" "$destination/$dep"; then
      echo "Conflicting llama.cpp CUDA dependency named $dep" >&2
      exit 1
    fi
    cp -L "$resolved" "$destination/$dep"
  done <<< "$resolved_deps"

  driver_library="$(find "$destination" -maxdepth 1 -type f \
    \( -name 'libcuda.so*' -o -name 'libnvidia-*.so*' \) -print -quit)"
  if [ -n "$driver_library" ]; then
    echo "NVIDIA driver libraries must not be bundled in a llama.cpp pack." >&2
    exit 1
  fi
}

package_profile() {
  profile="$1"
  accelerator="$2"
  feature="$3"
  override="$4"
  kv_mode="$5"
  root="$work_root/payload-$profile"
  mkdir -p "$root/lib" "$root/include" "$root/licenses"

  library="$(build_profile "$profile" "$feature" "$override" "$kv_mode")"
  if [ ! -f "$library" ]; then
    echo "llama.cpp $profile build did not produce $library" >&2
    exit 1
  fi
  if ! file "$library" | grep -q 'ELF .* shared object'; then
    echo "llama.cpp $profile entrypoint is not an ELF shared object: $library" >&2
    exit 1
  fi
  dynamic_symbols="$(nm -D "$library")"
  case "$dynamic_symbols" in
    *kapsl_llama_cpp_backend_v1*) ;;
    *)
      echo "llama.cpp $profile entrypoint is missing kapsl_llama_cpp_backend_v1" >&2
      exit 1
      ;;
  esac
  case "$dynamic_symbols" in
    *" U __isoc23_"*)
      echo "llama.cpp $profile entrypoint requires glibc C23 symbols outside the Linux release baseline." >&2
      nm -D --undefined-only "$library" | grep '__isoc23_' >&2 || true
      exit 1
      ;;
  esac
  if ! grep -aFq "KAPSL_LLAMA_CPP_KV_MODE=$kv_mode" "$library"; then
    echo "llama.cpp $profile binary marker does not match signed KV mode $kv_mode" >&2
    exit 1
  fi

  cp "$library" "$root/lib/libkapsl_backend_llama_cpp.so"
  cp "$header" "$root/include/"
  cp LICENSE "$root/licenses/KAPSL-LICENSE"
  cp NOTICE "$root/licenses/KAPSL-NOTICE"
  cp "$llama_license" "$root/licenses/LLAMA-CPP-LICENSE"
  copy_runtime_dependencies "$library" "$root/lib" "$accelerator"
  gcc_runtime="$(find "$root/lib" -maxdepth 1 -type f \
    \( -name 'libgomp.so*' -o -name 'libatomic.so*' -o -name 'libquadmath.so*' \) \
    -print -quit)"
  if [ -n "$gcc_runtime" ]; then
    gcc_license="${KAPSL_GCC_RUNTIME_LICENSE_FILE:-/usr/share/doc/libgomp1/copyright}"
    if [ ! -f "$gcc_license" ]; then
      echo "Missing GCC runtime redistribution notice for $gcc_runtime." >&2
      exit 1
    fi
    cp "$gcc_license" "$root/licenses/GCC-RUNTIME-COPYRIGHT"
  fi
  if [ "$accelerator" = "cuda" ]; then
    nvidia_license="${KAPSL_NVIDIA_LICENSE_FILE:-${KAPSL_CUDA_RUNTIME_ROOT:-}/NVIDIA-CONTAINER-LICENSE}"
    if [ -z "$nvidia_license" ] || [ ! -f "$nvidia_license" ]; then
      echo "Missing NVIDIA redistribution license for llama.cpp CUDA pack." >&2
      exit 1
    fi
    cp "$nvidia_license" "$root/licenses/NVIDIA-CONTAINER-LICENSE"
  fi
  while IFS= read -r -d '' elf; do
    if file "$elf" | grep -q 'ELF .* shared object'; then
      patchelf --set-rpath '$ORIGIN' "$elf"
    fi
  done < <(find "$root/lib" -maxdepth 1 -type f -print0 | sort -z)

  cat > "$root/backend-pack.json" <<EOF
{
  "schema_version": 1,
  "backend": "llama-cpp",
  "profile": "$profile",
  "pack_version": "llama.cpp-0.1.146-kapsl.1",
  "runtime_abi": 1,
  "platform": "linux-x86_64",
  "execution_mode": "native",
  "kv_mode": "$kv_mode",
  "entrypoint": "lib/libkapsl_backend_llama_cpp.so"
}
EOF

  archive="dist/kapsl-backend-llama-cpp-${profile}-${KAPSL_VERSION}-linux-x86_64.tar.gz"
  template="${archive}.manifest.json"
  python3 - "$root" "$template" "$profile" "$accelerator" "$KAPSL_VERSION" "$kv_mode" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
template_path = pathlib.Path(sys.argv[2])
profile = sys.argv[3]
accelerator = sys.argv[4]
version = sys.argv[5]
kv_mode = sys.argv[6]

files = {}
installed_bytes = 0
licenses = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root).as_posix()
    payload = path.read_bytes()
    installed_bytes += len(payload)
    files[relative] = hashlib.sha256(payload).hexdigest()
    if relative.startswith("licenses/"):
        licenses.append({"name": path.name, "path": relative})

template = {
    "schema_version": 1,
    "backend": "llama-cpp",
    "profile": profile,
    "pack_version": "llama.cpp-0.1.146-kapsl.1",
    "runtime_abi": 1,
    "compatible_kapsl": f"={version}",
    "platform": "linux-x86_64",
    "architecture": "x86_64",
    "accelerator_profile": accelerator,
    "execution_mode": "native",
    "kv_mode": kv_mode,
    "entrypoint": "lib/libkapsl_backend_llama_cpp.so",
    "installed_bytes": max(installed_bytes, 1),
    "memory": {
        "host_bytes": 67108864,
        "accelerator_bytes": 0 if accelerator == "cpu" else 67108864,
        "workspace_weight_ppm": 125000,
        "minimum_workspace_bytes": 268435456,
    },
    "installer": {"kind": "extract"},
    "files": files,
    "licenses": licenses,
    "priority": 100,
}
if accelerator == "cuda":
    template["minimum_cuda"] = "12.0"
    template["minimum_driver"] = "560.28.03"
template_path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
PY

  tar -C "$root" -czf "$archive" .
  sha256sum "$archive" > "${archive}.sha256"
  echo "Packaged $archive"
}

package_profile cpu cpu cpu "${KAPSL_LLAMA_CPU_LIBRARY:-}" native
package_profile cuda12 cuda cuda12-shared-pool "${KAPSL_LLAMA_CUDA_LIBRARY:-}" shared_pool
