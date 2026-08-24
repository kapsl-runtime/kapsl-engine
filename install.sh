#!/usr/bin/env sh
# Kapsl CLI installer
# Usage: curl -fsSL https://downloads.kapsl.net/install.sh | sh
# CUDA: curl -fsSL https://downloads.kapsl.net/install-cuda.sh | sh
set -e

BASE_URL="${KAPSL_BASE_URL:-https://downloads.kapsl.net}"
BIN_NAME="kapsl"
INSTALL_DIR="${KAPSL_INSTALL_DIR:-$HOME/.local/bin}"
ACCELERATOR="${KAPSL_ACCELERATOR:-cpu}"
CHANNEL="${KAPSL_CHANNEL:-stable}"
VERSION="${KAPSL_VERSION:-}"

usage() {
    cat <<'EOF'
Install Kapsl

Usage:
  install.sh [--accelerator cpu|cuda|tensorrt] [--channel stable|beta]
             [--version VERSION] [--install-dir DIR] [--base-url URL]

Examples:
  curl -fsSL https://downloads.kapsl.net/install.sh | sh
  curl -fsSL https://downloads.kapsl.net/install-cuda.sh | sh
  curl -fsSL https://downloads.kapsl.net/install.sh | sh -s -- --accelerator tensorrt
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --accelerator)
            [ "$#" -ge 2 ] || { echo "--accelerator requires a value" >&2; exit 1; }
            ACCELERATOR="$2"
            shift 2
            ;;
        --channel)
            [ "$#" -ge 2 ] || { echo "--channel requires a value" >&2; exit 1; }
            CHANNEL="$2"
            shift 2
            ;;
        --version)
            [ "$#" -ge 2 ] || { echo "--version requires a value" >&2; exit 1; }
            VERSION="$2"
            shift 2
            ;;
        --install-dir)
            [ "$#" -ge 2 ] || { echo "--install-dir requires a value" >&2; exit 1; }
            INSTALL_DIR="$2"
            shift 2
            ;;
        --base-url)
            [ "$#" -ge 2 ] || { echo "--base-url requires a value" >&2; exit 1; }
            BASE_URL="$2"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown installer option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$CHANNEL" in
    stable) RUNTIME_PATH="runtime" ;;
    beta) RUNTIME_PATH="runtime/beta" ;;
    *)
        echo "Unsupported channel '${CHANNEL}'. Use stable or beta." >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Detect OS and arch
# ---------------------------------------------------------------------------
detect_platform() {
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)  os="linux" ;;
        Darwin) os="macos" ;;
        *)
            echo "Unsupported OS: $os" >&2
            exit 1
            ;;
    esac

    case "$arch" in
        x86_64 | amd64) arch="x86_64" ;;
        aarch64 | arm64) arch="aarch64" ;;
        *)
            echo "Unsupported architecture: $arch" >&2
            exit 1
            ;;
    esac

    echo "${os}-${arch}"
}

# ---------------------------------------------------------------------------
# Resolve latest version from R2
# ---------------------------------------------------------------------------
latest_version() {
    url="${BASE_URL}/${RUNTIME_PATH}/latest.txt"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$url"
    else
        echo "curl or wget is required" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Download a file
# ---------------------------------------------------------------------------
download() {
    url="$1"
    dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --progress-bar "$url" -o "$dest"
    else
        wget -q --show-progress "$url" -O "$dest"
    fi
}

# Install a bundle tarball into "$INSTALL_DIR". Returns non-zero (after
# reporting why) if it could not be downloaded, unpacked, or did not carry the
# binary, so the caller can try a different bundle.
install_bundle() {
    bundle_file="$1"
    bundle_url="${BASE_URL}/${RUNTIME_PATH}/v${VERSION}/${bundle_file}"
    bundle_tmp="${TMP_DIR}/${bundle_file}"
    extract_dir="${TMP_DIR}/extract-${bundle_file}"

    if ! download "$bundle_url" "$bundle_tmp"; then
        echo "Bundle ${bundle_file} is unavailable." >&2
        return 1
    fi

    mkdir -p "$extract_dir"
    if ! tar -xzf "$bundle_tmp" -C "$extract_dir"; then
        echo "Failed to extract ${bundle_file}." >&2
        return 1
    fi

    bundle_bin="$(find "$extract_dir" -type f -name "$BIN_NAME" | head -n 1)"
    if [ -z "$bundle_bin" ]; then
        echo "Bundle ${bundle_file} does not contain ${BIN_NAME}." >&2
        return 1
    fi

    cp -R "$(dirname "$bundle_bin")/." "$INSTALL_DIR/"
    chmod +x "${INSTALL_DIR}/${BIN_NAME}"
    INSTALLED_BUNDLE_DIR="$(dirname "$bundle_bin")"
}

install_provider_pack() {
    provider="$1"
    provider_version="$2"

    if [ "$PLATFORM" != "linux-x86_64" ]; then
        echo "The ${provider} provider pack is currently available only for Linux x86_64." >&2
        exit 1
    fi

    pack_file="kapsl-provider-${provider}${provider_version}-${VERSION}-${PLATFORM}.tar.gz"
    pack_url="${BASE_URL}/${RUNTIME_PATH}/v${VERSION}/${pack_file}"
    pack_tmp="${TMP_DIR}/${pack_file}"

    echo "Installing Kapsl ${provider}${provider_version} provider pack..."
    download "$pack_url" "$pack_tmp"
    tar -xzf "$pack_tmp" -C "$INSTALL_DIR"
}

install_vllm_backend_pack() {
    if ! command -v bash >/dev/null 2>&1; then
        echo "bash is required to install the certified managed-vLLM backend." >&2
        exit 1
    fi

    pack_file="kapsl-backend-vllm-${VERSION}-${PLATFORM}.tar.gz"
    pack_url="${BASE_URL}/${RUNTIME_PATH}/v${VERSION}/${pack_file}"
    pack_tmp="${TMP_DIR}/${pack_file}"
    checksum_tmp="${pack_tmp}.sha256"

    echo "Installing certified Kapsl managed-vLLM backend..."
    download "$pack_url" "$pack_tmp"
    download "${pack_url}.sha256" "$checksum_tmp"
    expected_hash="$(awk 'NR == 1 { print $1 }' "$checksum_tmp")"
    actual_hash="$(sha256sum "$pack_tmp" | awk '{ print $1 }')"
    if [ -z "$expected_hash" ] || [ "$actual_hash" != "$expected_hash" ]; then
        echo "Managed-vLLM backend checksum mismatch." >&2
        exit 1
    fi

    if ! tar -tzf "$pack_tmp" | awk '
        $0 == "./" || $0 == "./backends/" || $0 == "./backends/vllm-bootstrap/" { next }
        /^\.\/backends\/vllm-bootstrap\// && $0 !~ /(^|\/)\.\.(\/|$)/ { next }
        { bad = 1 }
        END { exit bad }
    '; then
        echo "Managed-vLLM backend archive contains an unexpected path." >&2
        exit 1
    fi

    mkdir -p "$INSTALL_DIR/backends"
    rm -rf "$INSTALL_DIR/backends/vllm-bootstrap"
    tar -xzf "$pack_tmp" -C "$INSTALL_DIR"
    rm -f "$pack_tmp"

    bootstrap_root="$INSTALL_DIR/backends/vllm-bootstrap"
    if [ ! -x "$bootstrap_root/bootstrap.sh" ]; then
        echo "Managed-vLLM backend archive is missing bootstrap.sh." >&2
        exit 1
    fi
    bash "$bootstrap_root/bootstrap.sh" \
        "$bootstrap_root" \
        "$INSTALL_DIR/backends/vllm"
    rm -rf "$bootstrap_root"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PLATFORM="$(detect_platform)"

case "$ACCELERATOR" in
    cpu | cuda | cuda12 | tensorrt | tensorrt10) ;;
    *)
        echo "Unsupported accelerator '${ACCELERATOR}'. Use cpu, cuda, or tensorrt." >&2
        exit 1
        ;;
esac

case "$ACCELERATOR" in
    cuda | cuda12 | tensorrt | tensorrt10)
        if [ "$PLATFORM" != "linux-x86_64" ]; then
            echo "The ${ACCELERATOR} runtime is currently available only for Linux x86_64." >&2
            exit 1
        fi
        ;;
esac

if [ -z "$VERSION" ]; then
    printf "Fetching latest version... "
    VERSION="$(latest_version)"
    echo "$VERSION"
fi

case "$ACCELERATOR" in
    cpu)
        BUNDLE_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}.tar.gz"
        ;;
    cuda | cuda12 | tensorrt | tensorrt10)
        # One archive contains the CUDA-compiled GGUF runtime, the ONNX CUDA
        # provider, and their user-space CUDA dependencies. Only the matching
        # NVIDIA driver remains a host prerequisite.
        BUNDLE_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}-cuda12.tar.gz"
        ;;
esac

BIN_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}"
DOWNLOAD_URL="${BASE_URL}/${RUNTIME_PATH}/v${VERSION}/${BIN_FILE}"

echo "Installing kapsl ${VERSION} (${CHANNEL}, ${PLATFORM}) to ${INSTALL_DIR}..."

mkdir -p "$INSTALL_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if ! install_bundle "$BUNDLE_FILE"; then
    if [ "$ACCELERATOR" != "cpu" ]; then
        echo "The requested ${ACCELERATOR} runtime is unavailable; no CPU runtime was substituted." >&2
        exit 1
    fi

    # Old CPU releases may only have the bare executable. Accelerator installs
    # deliberately do not use this fallback because it cannot run GGUF on CUDA.
    echo "Falling back to single executable." >&2
    tmp="${TMP_DIR}/${BIN_FILE}"
    download "$DOWNLOAD_URL" "$tmp"
    chmod +x "$tmp"
    mv "$tmp" "${INSTALL_DIR}/${BIN_NAME}"
fi

case "$ACCELERATOR" in
    cuda | cuda12 | tensorrt | tensorrt10)
        # Compatibility for pre-merged releases: new CUDA archives already
        # contain this marker and need no second download.
        if [ ! -f "${INSTALLED_BUNDLE_DIR}/kapsl-provider-cuda12.json" ]; then
            echo "Installing legacy split ONNX CUDA provider pack..."
            install_provider_pack "cuda" "12"
        fi
        ;;
esac
case "$ACCELERATOR" in
    tensorrt | tensorrt10)
        install_provider_pack "tensorrt" "10"
        ;;
esac
case "$ACCELERATOR" in
    cuda | cuda12 | tensorrt | tensorrt10)
        # New shared-pool builds carry this certified profile marker. Install
        # their exact Python/vLLM wheelhouse and materialize it beside Kapsl;
        # old CUDA releases remain installable without an asset they predate.
        if grep -aFq 'vllm-v1-packed-cuda-ipc/flash-attn' "${INSTALL_DIR}/${BIN_NAME}"; then
            install_vllm_backend_pack
        fi
        ;;
esac

echo "Installed to ${INSTALL_DIR}/${BIN_NAME}"

# Remind user to add to PATH if needed
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        echo ""
        echo "Add the following to your shell profile to use kapsl:"
        echo "  export PATH=\"\$PATH:${INSTALL_DIR}\""
        ;;
esac

echo ""
if [ "$ACCELERATOR" != "cpu" ]; then
    echo "Installed accelerator profile: ${ACCELERATOR}"
    echo "GGUF models: CUDA compiled into this runtime build."
    echo "ONNX models: CUDA execution provider installed."
    if [ -x "${INSTALL_DIR}/backends/vllm/bin/python" ]; then
        echo "SafeTensors generation models: managed vLLM backend installed."
    fi
    echo "A compatible NVIDIA driver is required."
fi
echo "Run 'kapsl --help' to get started."
