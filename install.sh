#!/usr/bin/env sh
# Kapsl CLI installer
# Usage: curl -fsSL https://kapsl.ai/install.sh | sh
set -e

REPO="kapsl-runtime/kapsl-runtime"
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
# Resolve latest version from GitHub
# ---------------------------------------------------------------------------
latest_version() {
    url="https://api.github.com/repos/${REPO}/releases/latest"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" | grep '"tag_name"' | sed 's/.*"tag_name": *"v\([^"]*\)".*/\1/'
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$url" | grep '"tag_name"' | sed 's/.*"tag_name": *"v\([^"]*\)".*/\1/'
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
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/v${VERSION}/${BIN_FILE}"

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
