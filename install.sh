#!/usr/bin/env sh
# Kapsl CLI installer
# Usage: curl -fsSL https://downloads.kapsl.net/install.sh | sh
set -e

BASE_URL="${KAPSL_BASE_URL:-https://downloads.kapsl.net}"
BIN_NAME="kapsl"
INSTALL_DIR="${KAPSL_INSTALL_DIR:-$HOME/.local/bin}"

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
    url="${BASE_URL}/runtime/latest.txt"
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
VERSION="${KAPSL_VERSION:-}"
PLATFORM="$(detect_platform)"

if [ -z "$VERSION" ]; then
    printf "Fetching latest version... "
    VERSION="$(latest_version)"
    echo "$VERSION"
fi

BIN_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}"
DOWNLOAD_URL="${BASE_URL}/runtime/v${VERSION}/${BIN_FILE}"

echo "Installing kapsl ${VERSION} (${PLATFORM}) to ${INSTALL_DIR}..."

mkdir -p "$INSTALL_DIR"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

download "$DOWNLOAD_URL" "$TMP"
chmod +x "$TMP"
mv "$TMP" "${INSTALL_DIR}/${BIN_NAME}"

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
echo "Run 'kapsl --help' to get started."
