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
}

# A working driver is necessary but not sufficient for the CUDA build: it also
# carries DT_NEEDED entries the driver does not provide (libnccl.so.2 among
# them, which ships in the CUDA container images but is a separate package on
# a bare host), so it can install cleanly and still fail to load. Prove the
# binary starts before committing to it, and surface the loader's own error —
# it names the missing library, which is the one thing that tells the user how
# to get the GPU back.
runtime_starts() {
    start_error="$("${INSTALL_DIR}/${BIN_NAME}" --version 2>&1 >/dev/null)" && return 0
    if [ -n "$start_error" ]; then
        echo "  ${start_error}" >&2
    fi
    return 1
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

# GGUF/llama.cpp only gets GPU support if the binary was compiled with
# --features cuda, and that build links libcuda.so.1 directly (not dlopen'd),
# so it cannot exec at all on a host with no NVIDIA driver loaded. Only reach
# for it when the driver is confirmed working; otherwise keep installing the
# portable build so `kapsl` still runs, and say so.
USE_CUDA_RUNTIME=0
case "$ACCELERATOR" in
    cuda | cuda12 | tensorrt | tensorrt10)
        if [ "$PLATFORM" = "linux-x86_64" ]; then
            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
                USE_CUDA_RUNTIME=1
            else
                echo "Warning: no working NVIDIA driver detected (nvidia-smi missing or failed)." >&2
                echo "Installing the portable runtime instead; GGUF models will run on CPU until a driver is available. ONNX models still get the ${ACCELERATOR} provider pack." >&2
            fi
        fi
        ;;
esac

PORTABLE_BUNDLE_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}.tar.gz"
CUDA_BUNDLE_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}-cuda12.tar.gz"
BIN_FILE="${BIN_NAME}-${VERSION}-${PLATFORM}"
DOWNLOAD_URL="${BASE_URL}/${RUNTIME_PATH}/v${VERSION}/${BIN_FILE}"

echo "Installing kapsl ${VERSION} (${CHANNEL}, ${PLATFORM}) to ${INSTALL_DIR}..."

mkdir -p "$INSTALL_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

INSTALLED=0

# Prefer the CUDA-compiled runtime when the driver check passed, but never let
# a missing -cuda12 asset (older releases predate it) cost the user the ORT
# sidecars that the portable bundle carries — degrade to it instead.
if [ "$USE_CUDA_RUNTIME" = "1" ]; then
    if install_bundle "$CUDA_BUNDLE_FILE" && runtime_starts; then
        INSTALLED=1
    else
        echo "The CUDA runtime is unusable on this host; falling back to the portable runtime, so GGUF models will run on CPU." >&2
        USE_CUDA_RUNTIME=0
    fi
fi

if [ "$INSTALLED" = "0" ] && install_bundle "$PORTABLE_BUNDLE_FILE"; then
    INSTALLED=1
fi

# Last resort: the bare executable, which ships without the ORT sidecars and is
# never the CUDA build, so GGUF stays on the CPU whatever was requested.
if [ "$INSTALLED" = "0" ]; then
    echo "Falling back to single executable." >&2
    USE_CUDA_RUNTIME=0
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
    if [ "$USE_CUDA_RUNTIME" = "1" ]; then
        echo "GGUF models: GPU-accelerated (CUDA compiled into this runtime build)."
    else
        echo "GGUF models: CPU only for this install (ONNX models still get the ${ACCELERATOR} provider pack)."
    fi
fi
echo "Run 'kapsl --help' to get started."
