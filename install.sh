#!/usr/bin/env sh
# Kapsl CLI installer
# Usage: curl -fsSL https://downloads.kapsl.net/install.sh | sh
# CUDA: curl -fsSL https://downloads.kapsl.net/install.sh | sh -s -- --accelerator cuda
set -e

BASE_URL="${KAPSL_BASE_URL:-https://downloads.kapsl.net}"
BIN_NAME="kapsl"
INSTALL_DIR="${KAPSL_INSTALL_DIR:-$HOME/.local/bin}"
ACCELERATOR="${KAPSL_ACCELERATOR:-cpu}"
CHANNEL="stable"
VERSION="${KAPSL_VERSION:-}"

usage() {
    cat <<'EOF'
Install Kapsl

Usage:
  install.sh [--accelerator cpu|cuda|tensorrt] [--channel stable|beta]
             [--version VERSION] [--install-dir DIR] [--base-url URL]

Examples:
  curl -fsSL https://downloads.kapsl.net/install.sh | sh
  curl -fsSL https://downloads.kapsl.net/install.sh | sh -s -- --accelerator cuda
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

if [ -z "$VERSION" ]; then
    printf "Fetching latest version... "
    VERSION="$(latest_version)"
    echo "$VERSION"
fi

BUNDLE_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}.tar.gz"
BUNDLE_URL="${BASE_URL}/${RUNTIME_PATH}/v${VERSION}/${BUNDLE_FILE}"
BIN_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}"
DOWNLOAD_URL="${BASE_URL}/${RUNTIME_PATH}/v${VERSION}/${BIN_FILE}"

echo "Installing kapsl ${VERSION} (${CHANNEL}, ${PLATFORM}) to ${INSTALL_DIR}..."

mkdir -p "$INSTALL_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

BUNDLE_TMP="${TMP_DIR}/${BUNDLE_FILE}"
EXTRACT_DIR="${TMP_DIR}/bundle"

if download "$BUNDLE_URL" "$BUNDLE_TMP"; then
    mkdir -p "$EXTRACT_DIR"
    if tar -xzf "$BUNDLE_TMP" -C "$EXTRACT_DIR"; then
        BUNDLE_BIN="$(find "$EXTRACT_DIR" -type f -name "$BIN_NAME" | head -n 1)"
        if [ -n "$BUNDLE_BIN" ]; then
            cp -R "$(dirname "$BUNDLE_BIN")/." "$INSTALL_DIR/"
            chmod +x "${INSTALL_DIR}/${BIN_NAME}"
        else
            echo "Downloaded bundle does not contain ${BIN_NAME}; falling back to single executable." >&2
            TMP="${TMP_DIR}/${BIN_FILE}"
            download "$DOWNLOAD_URL" "$TMP"
            chmod +x "$TMP"
            mv "$TMP" "${INSTALL_DIR}/${BIN_NAME}"
        fi
    else
        echo "Failed to extract bundle; falling back to single executable." >&2
        TMP="${TMP_DIR}/${BIN_FILE}"
        download "$DOWNLOAD_URL" "$TMP"
        chmod +x "$TMP"
        mv "$TMP" "${INSTALL_DIR}/${BIN_NAME}"
    fi
else
    echo "Bundle unavailable; falling back to single executable." >&2
    TMP="${TMP_DIR}/${BIN_FILE}"
    download "$DOWNLOAD_URL" "$TMP"
    chmod +x "$TMP"
    mv "$TMP" "${INSTALL_DIR}/${BIN_NAME}"
fi

case "$ACCELERATOR" in
    cuda | cuda12 | tensorrt | tensorrt10)
        install_provider_pack "cuda" "12"
        ;;
esac
case "$ACCELERATOR" in
    tensorrt | tensorrt10)
        install_provider_pack "tensorrt" "10"
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
    echo "Linux accelerator packs require compatible NVIDIA system runtime libraries."
fi
echo "Run 'kapsl --help' to get started."
