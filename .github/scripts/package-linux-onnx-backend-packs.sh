#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"

if [ "${RUNNER_OS:-Linux}" != "Linux" ] || [ "${RUNNER_ARCH:-X64}" != "X64" ]; then
  echo "ONNX native backend packs are currently published only for Linux x86_64." >&2
  exit 1
fi

for command_name in cc file patchelf python3 sha256sum tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to package ONNX backend packs." >&2
    exit 1
  fi
done

core_dir="${KAPSL_ORT_CORE_DIR:-ort-core-libs}"
cuda_dir="${KAPSL_ORT_CUDA_DIR:-ort-cuda-libs}"
tensorrt_provider_dir="${KAPSL_ORT_TENSORRT_DIR:-ort-tensorrt-libs}"
cuda_runtime_root="${KAPSL_CUDA_RUNTIME_ROOT:-${RUNNER_TEMP:?RUNNER_TEMP is required}/kapsl-${KAPSL_VERSION}-linux-x86_64-cuda12}"
tensorrt_runtime_dir="${KAPSL_TENSORRT_RUNTIME_DIR:-tensorrt-runtime-libs}"
tensorrt_license_dir="${KAPSL_TENSORRT_LICENSE_DIR:-tensorrt-license-files}"
entrypoint_source=".github/scripts/onnx-backend-pack-entrypoint.c"
onnx_license=".github/licenses/ONNX-RUNTIME-LICENSE"

for required in \
  "$core_dir/libonnxruntime_providers_shared.so" \
  "$cuda_dir/libonnxruntime_providers_cuda.so" \
  "$tensorrt_provider_dir/libonnxruntime_providers_tensorrt.so" \
  "$cuda_runtime_root/libcudnn.so.9" \
  "$entrypoint_source" \
  "$onnx_license" \
  LICENSE; do
  if [ ! -f "$required" ]; then
    echo "Missing required ONNX pack input: $required" >&2
    exit 1
  fi
done
if [ ! -d "$tensorrt_runtime_dir" ] || [ -z "$(find "$tensorrt_runtime_dir" -maxdepth 1 -type f -print -quit)" ]; then
  echo "TensorRT runtime dependency directory is empty: $tensorrt_runtime_dir" >&2
  exit 1
fi

work_parent="${RUNNER_TEMP:-/tmp}"
work_root="$(mktemp -d "${work_parent%/}/kapsl-onnx-packs.XXXXXX")"
cleanup() {
  rm -rf "$work_root"
}
trap cleanup EXIT INT TERM
mkdir -p dist

copy_flat_files() {
  source_dir="$1"
  destination="$2"
  [ -d "$source_dir" ] || return 0
  while IFS= read -r -d '' source; do
    name="$(basename "$source")"
    if [ -f "$destination/$name" ]; then
      if ! cmp -s "$source" "$destination/$name"; then
        echo "Conflicting pack dependency named $name" >&2
        exit 1
      fi
      continue
    fi
    cp -L "$source" "$destination/$name"
  done < <(find "$source_dir" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
}

copy_cuda_runtime_libraries() {
  destination="$1"
  while IFS= read -r -d '' source; do
    name="$(basename "$source")"
    case "$name" in
      libcuda.so* | libnvidia-*.so*)
        echo "Refusing to bundle host NVIDIA driver library $name" >&2
        exit 1
        ;;
      lib*.so*)
        if [ -f "$destination/$name" ] && ! cmp -s "$source" "$destination/$name"; then
          echo "Conflicting CUDA dependency named $name" >&2
          exit 1
        fi
        cp -L "$source" "$destination/$name"
        ;;
    esac
  done < <(find "$cuda_runtime_root" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
}

write_provider_marker() {
  root="$1"
  provider="$2"
  version="$3"
  requires="$4"
  python3 - "$root" "$provider" "$version" "$requires" "$KAPSL_VERSION" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
provider = sys.argv[2]
provider_version = sys.argv[3]
requires = [item for item in sys.argv[4].split(",") if item]
runtime_version = sys.argv[5]
marker_name = f"kapsl-provider-{provider}{provider_version}.json"
files = sorted(
    path.name
    for path in root.iterdir()
    if path.is_file() and not path.name.startswith("kapsl-provider-")
)
marker = {
    "schema_version": 1,
    "provider": provider,
    "provider_version": provider_version,
    "runtime_version": runtime_version,
    "platform": "linux-x86_64",
    "requires": requires,
    "files": files,
    "system_requirements": "A compatible host NVIDIA driver is required; all non-driver user-space dependencies are pack-local.",
}
(root / marker_name).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
PY
}

normalize_runpaths() {
  root="$1"
  while IFS= read -r -d '' candidate; do
    if file "$candidate" | grep -q 'ELF .* shared object'; then
      patchelf --set-rpath '$ORIGIN' "$candidate"
    fi
  done < <(find "$root" -maxdepth 1 -type f -name '*.so*' -print0 | sort -z)
}

package_profile() {
  profile="$1"
  accelerator="$2"
  profile_id="$3"
  root="$work_root/$profile"
  mkdir -p "$root/licenses"

  copy_flat_files "$core_dir" "$root"
  case "$profile" in
    cpu)
      ;;
    cuda12)
      copy_flat_files "$cuda_dir" "$root"
      copy_cuda_runtime_libraries "$root"
      ;;
    tensorrt10)
      copy_flat_files "$cuda_dir" "$root"
      copy_cuda_runtime_libraries "$root"
      copy_flat_files "$tensorrt_provider_dir" "$root"
      copy_flat_files "$tensorrt_runtime_dir" "$root"
      ;;
    *)
      echo "Unknown ONNX profile: $profile" >&2
      exit 1
      ;;
  esac

  cp LICENSE "$root/licenses/KAPSL-LICENSE"
  cp "$onnx_license" "$root/licenses/ONNX-RUNTIME-LICENSE"
  if [ "$profile" != "cpu" ]; then
    nvidia_license="${KAPSL_NVIDIA_LICENSE_FILE:-$cuda_runtime_root/NVIDIA-CONTAINER-LICENSE}"
    if [ ! -f "$nvidia_license" ]; then
      echo "Missing NVIDIA redistribution license: $nvidia_license" >&2
      exit 1
    fi
    cp "$nvidia_license" "$root/licenses/NVIDIA-CONTAINER-LICENSE"
  fi
  if [ "$profile" = "tensorrt10" ] && [ -d "$tensorrt_license_dir" ]; then
    license_number=0
    while IFS= read -r -d '' license; do
      license_number=$((license_number + 1))
      cp "$license" "$root/licenses/TENSORRT-${license_number}-$(basename "$license")"
    done < <(find "$tensorrt_license_dir" -type f -print0 | sort -z)
  fi

  cc -shared -fPIC -O2 -fvisibility=hidden \
    "-DKAPSL_ONNX_PROFILE=$profile_id" \
    "$entrypoint_source" \
    -o "$root/libkapsl_backend_onnx.so"

  case "$profile" in
    cpu)
      if find "$root" -maxdepth 1 -type f \
        \( -iname '*cuda*' -o -iname '*cudnn*' -o -iname '*tensorrt*' -o -iname '*nvinfer*' \) \
        -print -quit | grep -q .; then
        echo "CPU ONNX pack unexpectedly contains accelerator libraries." >&2
        exit 1
      fi
      ;;
    cuda12)
      write_provider_marker "$root" cuda 12 ""
      if find "$root" -maxdepth 1 -type f \
        \( -iname '*tensorrt*' -o -iname '*nvinfer*' \) -print -quit | grep -q .; then
        echo "CUDA ONNX pack unexpectedly contains TensorRT libraries." >&2
        exit 1
      fi
      ;;
    tensorrt10)
      write_provider_marker "$root" cuda 12 ""
      write_provider_marker "$root" tensorrt 10 "cuda12"
      ;;
  esac

  normalize_runpaths "$root"

  cat > "$root/backend-pack.json" <<EOF
{
  "schema_version": 1,
  "backend": "onnx",
  "profile": "${profile}",
  "pack_version": "ort-2.0.0-rc.11",
  "runtime_abi": 1,
  "platform": "linux-x86_64",
  "execution_mode": "native",
  "entrypoint": "libkapsl_backend_onnx.so"
}
EOF

  archive="dist/kapsl-backend-onnx-${profile}-${KAPSL_VERSION}-linux-x86_64.tar.gz"
  template="${archive}.manifest.json"
  python3 - "$root" "$template" "$profile" "$accelerator" "$KAPSL_VERSION" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
template_path = pathlib.Path(sys.argv[2])
profile = sys.argv[3]
accelerator = sys.argv[4]
version = sys.argv[5]

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

memory = {
    "host_bytes": 67108864,
    "accelerator_bytes": 0 if accelerator == "cpu" else 134217728,
    "workspace_weight_ppm": 250000,
    "minimum_workspace_bytes": 268435456,
}
template = {
    "schema_version": 1,
    "backend": "onnx",
    "profile": profile,
    "pack_version": "ort-2.0.0-rc.11",
    "runtime_abi": 1,
    "compatible_kapsl": f"={version}",
    "platform": "linux-x86_64",
    "architecture": "x86_64",
    "accelerator_profile": accelerator,
    "execution_mode": "native",
    "entrypoint": "libkapsl_backend_onnx.so",
    "installed_bytes": max(installed_bytes, 1),
    "memory": memory,
    "installer": {"kind": "extract"},
    "files": files,
    "licenses": licenses,
    "priority": 100,
}
if accelerator != "cpu":
    template["minimum_cuda"] = "12.0"
    template["minimum_driver"] = "560.28.03"
template_path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
PY

  tar -C "$root" -czf "$archive" .
  sha256sum "$archive" > "${archive}.sha256"
  echo "Packaged $archive"
}

package_profile cpu cpu 1
package_profile cuda12 cuda 2
package_profile tensorrt10 tensorrt 3
