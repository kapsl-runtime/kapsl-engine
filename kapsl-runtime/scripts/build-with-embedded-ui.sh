#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EMBED_MODULE="$RUNTIME_DIR/crates/kapsl-cli/src/http/static_files.rs"

[[ -f "$EMBED_MODULE" ]] || {
  printf 'error: embedded UI module not found: %s\n' "$EMBED_MODULE" >&2
  exit 1
}

# Cargo's stable freshness checks are mtime-based. Source transfers such as
# `rsync -a` can preserve an asset timestamp older than an existing target
# artifact, so explicitly invalidate the Rust module that owns the embed.
touch "$EMBED_MODULE"
cd "$RUNTIME_DIR"
exec cargo build "$@"
