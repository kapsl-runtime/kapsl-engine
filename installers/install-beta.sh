#!/usr/bin/env sh
# Published one-command installer for the latest Kapsl beta. On Linux x86_64, select the
# CUDA runtime when a working NVIDIA driver is visible; all other hosts use the
# portable runtime. KAPSL_ACCELERATOR or --accelerator can override the choice.
set -e

base_url="${KAPSL_BASE_URL:-https://downloads.kapsl.net}"
accelerator="${KAPSL_ACCELERATOR:-}"

if [ -z "$accelerator" ]; then
    os="$(uname -s)"
    arch="$(uname -m)"
    if [ "$os" = "Linux" ] \
        && { [ "$arch" = "x86_64" ] || [ "$arch" = "amd64" ]; } \
        && command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi -L >/dev/null 2>&1; then
        accelerator="cuda"
    else
        accelerator="cpu"
    fi
fi

if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${base_url}/install-beta-base.sh" \
        | KAPSL_ACCELERATOR="$accelerator" KAPSL_CHANNEL=beta sh -s -- "$@"
elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${base_url}/install-beta-base.sh" \
        | KAPSL_ACCELERATOR="$accelerator" KAPSL_CHANNEL=beta sh -s -- "$@"
else
    echo "curl or wget is required" >&2
    exit 1
fi
