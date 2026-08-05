#!/bin/sh
set -eu

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"

accelerator="${1:-cpu}"
install_dir="${KAPSL_INSTALL_DIR:-/usr/local/bin}"
release_base="${KAPSL_RELEASE_BASE_URL:-https://github.com/kapsl-runtime/kapsl-engine/releases/download/v${KAPSL_VERSION}}"

case "$(uname -m)" in
    x86_64 | amd64)
        platform="linux-x86_64"
        ;;
    aarch64 | arm64)
        platform="linux-aarch64"
        ;;
    *)
        echo "Unsupported container architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

case "$accelerator" in
    cpu) ;;
    cuda | tensorrt)
        if [ "$platform" != "linux-x86_64" ]; then
            echo "${accelerator} images are supported only on Linux x86_64." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported accelerator: ${accelerator}" >&2
        exit 1
        ;;
esac

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

download() {
    url="$1"
    destination="$2"
    curl -fL \
        --connect-timeout 30 \
        --retry 10 \
        --retry-all-errors \
        --retry-delay 5 \
        "$url" \
        -o "$destination"
}

download_verified_asset() {
    asset_name="$1"
    asset_path="${tmp_dir}/${asset_name}"
    checksum_path="${asset_path}.sha256"

    download "${release_base}/${asset_name}" "$asset_path"
    download "${release_base}/${asset_name}.sha256" "$checksum_path"
    expected_checksum="$(awk '{print $1}' "$checksum_path")"
    actual_checksum="$(sha256sum "$asset_path" | awk '{print $1}')"
    if [ "$actual_checksum" != "$expected_checksum" ]; then
        echo "Checksum mismatch for ${asset_name}" >&2
        exit 1
    fi
}

extract_asset() {
    asset_name="$1"
    download_verified_asset "$asset_name"
    tar -xzf "${tmp_dir}/${asset_name}" -C "$install_dir"
}

mkdir -p "$install_dir"

case "$accelerator" in
    cpu)
        runtime_asset="kapsl-${KAPSL_VERSION}-${platform}.tar.gz"
        ;;
    cuda | tensorrt)
        # GGUF CUDA is statically compiled into this runtime. ONNX accelerator
        # providers remain separate packs installed below.
        runtime_asset="kapsl-${KAPSL_VERSION}-${platform}-cuda12.tar.gz"
        ;;
esac
extract_asset "$runtime_asset"

case "$accelerator" in
    cuda | tensorrt)
        extract_asset "kapsl-provider-cuda12-${KAPSL_VERSION}-${platform}.tar.gz"
        ;;
esac

if [ "$accelerator" = "tensorrt" ]; then
    extract_asset "kapsl-provider-tensorrt10-${KAPSL_VERSION}-${platform}.tar.gz"
fi

chmod +x "${install_dir}/kapsl"
