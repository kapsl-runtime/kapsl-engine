#!/usr/bin/env bash
# Replace the staged ONNX Runtime CUDA / TensorRT provider libraries with Microsoft's official
# release builds, fetched from the pinned URL and verified by SHA-256 (archive, then member).
#
# usage: fetch-microsoft-ort-providers.sh <cuda|tensorrt> <destination-dir> [<cuda|tensorrt> <dir> ...]
# env:   KAPSL_MS_ORT_PINS_FILE  pins file to source (default: .github/ort-microsoft-providers.env)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pins_file="${KAPSL_MS_ORT_PINS_FILE:-$script_dir/../ort-microsoft-providers.env}"
if [ ! -f "$pins_file" ]; then
  echo "Missing pins file: $pins_file" >&2
  exit 1
fi
# shellcheck source=/dev/null
. "$pins_file"
for var in MS_ORT_VERSION MS_ORT_URL MS_ORT_ARCHIVE_SHA256 MS_ORT_ARCHIVE_BYTES MS_ORT_MEMBER_PREFIX; do
  if [ -z "${!var:-}" ]; then
    echo "Pins file $pins_file does not set $var" >&2
    exit 1
  fi
done

if [ "$#" -eq 0 ] || [ $(( $# % 2 )) -ne 0 ]; then
  echo "usage: $0 <cuda|tensorrt> <destination-dir> [<cuda|tensorrt> <dir> ...]" >&2
  exit 2
fi

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}
size_of() { wc -c < "$1" | tr -d ' '; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT INT TERM

archive="$work/onnxruntime-gpu.tgz"
echo "Fetching Microsoft ONNX Runtime ${MS_ORT_VERSION} GPU archive: $MS_ORT_URL"
curl -fsSL --retry 3 --retry-delay 5 -o "$archive" "$MS_ORT_URL"

actual_bytes="$(size_of "$archive")"
if [ "$actual_bytes" != "$MS_ORT_ARCHIVE_BYTES" ]; then
  echo "Microsoft ONNX Runtime archive is $actual_bytes bytes; expected $MS_ORT_ARCHIVE_BYTES" >&2
  exit 1
fi
actual_sha="$(sha256_of "$archive")"
if [ "$actual_sha" != "$MS_ORT_ARCHIVE_SHA256" ]; then
  echo "Microsoft ONNX Runtime archive SHA-256 is $actual_sha; expected $MS_ORT_ARCHIVE_SHA256" >&2
  exit 1
fi

while [ "$#" -gt 0 ]; do
  provider="$1"
  destination="$2"
  shift 2
  case "$provider" in
    cuda)
      library="libonnxruntime_providers_cuda.so"
      expected="${MS_ORT_CUDA_SHA256:-}"
      ;;
    tensorrt)
      library="libonnxruntime_providers_tensorrt.so"
      expected="${MS_ORT_TENSORRT_SHA256:-}"
      ;;
    *)
      echo "Unknown provider '$provider'; expected cuda or tensorrt" >&2
      exit 2
      ;;
  esac
  if [ -z "$expected" ]; then
    echo "Pins file does not set the SHA-256 for the $provider provider" >&2
    exit 1
  fi
  if [ ! -d "$destination" ]; then
    echo "Destination directory does not exist: $destination" >&2
    exit 1
  fi

  extracted="$work/$library"
  tar -xzf "$archive" -O "${MS_ORT_MEMBER_PREFIX}/${library}" > "$extracted"
  actual="$(sha256_of "$extracted")"
  if [ "$actual" != "$expected" ]; then
    echo "$library from the Microsoft archive has SHA-256 $actual; expected $expected" >&2
    exit 1
  fi

  # Install under a temporary name, then rename, so a failure never leaves a half-written library.
  install -m 755 "$extracted" "$destination/.${library}.new"
  mv -f "$destination/.${library}.new" "$destination/$library"
  echo "Installed Microsoft ${MS_ORT_VERSION} $library into $destination (sha256 $actual)"
done
