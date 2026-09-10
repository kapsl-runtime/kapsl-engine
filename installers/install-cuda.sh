#!/usr/bin/env sh
# Published one-command Linux x86_64 CUDA installer.
set -e

base_url="${KAPSL_BASE_URL:-https://downloads.kapsl.net}"

if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${base_url}/install.sh" | KAPSL_ACCELERATOR=cuda sh -s -- "$@"
elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${base_url}/install.sh" | KAPSL_ACCELERATOR=cuda sh -s -- "$@"
else
    echo "curl or wget is required" >&2
    exit 1
fi
