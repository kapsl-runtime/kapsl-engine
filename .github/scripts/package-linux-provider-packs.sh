#!/usr/bin/env bash
set -euo pipefail

if [ "${RUNNER_OS:-}" != "Linux" ] || [ "${RUNNER_ARCH:-}" != "X64" ]; then
  echo "NVIDIA provider packs are currently published only for Linux x86_64."
  exit 0
fi

mkdir -p dist

package_provider() {
  provider="$1"
  provider_version="$2"
  source_directory="$3"
  required_provider="$4"

  if [ -z "$(find "$source_directory" -type f -print -quit)" ]; then
    echo "Skipping ${provider} pack because no provider sidecars were built."
    return
  fi

  platform="linux-x86_64"
  pack_name="kapsl-provider-${provider}${provider_version}-${KAPSL_VERSION}-${platform}"
  pack_root="${RUNNER_TEMP}/${pack_name}"
  rm -rf "$pack_root"
  mkdir -p "$pack_root"
  cp -L "${source_directory}"/* "$pack_root/"

  files_json=""
  while IFS= read -r file; do
    escaped="$(basename "$file" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    if [ -n "$files_json" ]; then
      files_json="${files_json},"
    fi
    files_json="${files_json}\"${escaped}\""
  done < <(find "$pack_root" -maxdepth 1 -type f | sort)

  requires_json=""
  if [ -n "$required_provider" ]; then
    requires_json="\"${required_provider}\""
  fi

  marker="${pack_root}/kapsl-provider-${provider}${provider_version}.json"
  {
    echo "{"
    echo "  \"schema_version\": 1,"
    echo "  \"provider\": \"${provider}\","
    echo "  \"provider_version\": \"${provider_version}\","
    echo "  \"runtime_version\": \"${KAPSL_VERSION}\","
    echo "  \"platform\": \"${platform}\","
    echo "  \"requires\": [${requires_json}],"
    echo "  \"files\": [${files_json}],"
    echo "  \"system_requirements\": \"NVIDIA driver and compatible CUDA/cuDNN/TensorRT runtime libraries must be installed separately on Linux.\""
    echo "}"
  } > "$marker"

  tar_path="dist/${pack_name}.tar.gz"
  tar -C "$pack_root" -czf "$tar_path" .
  sha256sum "$tar_path" > "${tar_path}.sha256"

  if command -v dpkg-deb >/dev/null 2>&1 && [ -n "${KAPSL_DEB_VERSION:-}" ]; then
    deb_root="${RUNNER_TEMP}/kapsl-runtime-${provider}${provider_version}-deb"
    rm -rf "$deb_root"
    mkdir -p "$deb_root/DEBIAN" "$deb_root/usr/local/bin"
    cp -L "$pack_root"/* "$deb_root/usr/local/bin/"

    dependencies="kapsl-runtime (= ${KAPSL_DEB_VERSION})"
    if [ -n "$required_provider" ]; then
      dependencies="${dependencies}, kapsl-runtime-${required_provider} (= ${KAPSL_DEB_VERSION})"
    fi

    {
      echo "Package: kapsl-runtime-${provider}${provider_version}"
      echo "Version: ${KAPSL_DEB_VERSION}"
      echo "Section: libs"
      echo "Priority: optional"
      echo "Architecture: $(dpkg --print-architecture)"
      echo "Depends: ${dependencies}"
      echo "Maintainer: Kapsl Team <support@kapsl.ai>"
      echo "Description: Optional Kapsl ${provider}${provider_version} ONNX Runtime provider pack."
      echo " Requires compatible NVIDIA driver and system runtime libraries."
    } > "$deb_root/DEBIAN/control"

    deb_path="dist/kapsl-runtime-${provider}${provider_version}_${KAPSL_DEB_VERSION}_$(dpkg --print-architecture).deb"
    dpkg-deb --build "$deb_root" "$deb_path"
    sha256sum "$deb_path" > "${deb_path}.sha256"
    rm -rf "$deb_root"
  fi

  rm -rf "$pack_root"
}

package_provider "cuda" "12" "ort-cuda-libs" ""
package_provider "tensorrt" "10" "ort-tensorrt-libs" "cuda12"
