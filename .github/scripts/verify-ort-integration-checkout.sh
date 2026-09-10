#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: verify-ort-integration-checkout.sh INTEGRATIONS_DIR INTEGRATIONS_REF" >&2
  exit 2
fi

integrations_dir="$1"
integrations_ref="$2"

if [[ ! "$integrations_ref" =~ ^[0-9a-f]{40}$ ]]; then
  echo "KAPSL_ORT_INTEGRATIONS_REF must be an exact lowercase 40-hex commit, got: $integrations_ref" >&2
  exit 1
fi
if ! git -C "$integrations_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Kapsl integrations directory is not a Git checkout: $integrations_dir" >&2
  exit 1
fi

actual_ref="$(git -C "$integrations_dir" rev-parse --verify 'HEAD^{commit}')"
if [ "$actual_ref" != "$integrations_ref" ]; then
  echo "kapsl-integrations checkout is $actual_ref, expected $integrations_ref" >&2
  exit 1
fi

dirty="$(
  git -C "$integrations_dir" status \
    --porcelain=v1 \
    --untracked-files=all \
    --ignored=matching
)"
if [ -n "$dirty" ]; then
  echo "kapsl-integrations checkout must be clean before ORT packaging:" >&2
  printf '%s\n' "$dirty" >&2
  exit 1
fi
