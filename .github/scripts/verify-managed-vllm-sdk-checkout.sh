#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: verify-managed-vllm-sdk-checkout.sh SDK_DIR SDK_REF" >&2
  exit 2
fi

sdk_dir="$1"
sdk_ref="$2"

if [[ ! "$sdk_ref" =~ ^[0-9a-f]{40}$ ]]; then
  echo "KAPSL_VLLM_SDK_REF must be an exact lowercase 40-hex commit, got: $sdk_ref" >&2
  exit 1
fi
if ! git -C "$sdk_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Kapsl vLLM SDK directory is not a Git checkout: $sdk_dir" >&2
  exit 1
fi

actual_sdk_ref="$(git -C "$sdk_dir" rev-parse --verify 'HEAD^{commit}')"
if [ "$actual_sdk_ref" != "$sdk_ref" ]; then
  echo "kapsl-sdk checkout is $actual_sdk_ref, expected $sdk_ref" >&2
  exit 1
fi

dirty="$(
  git -C "$sdk_dir" status \
    --porcelain=v1 \
    --untracked-files=all \
    --ignored=matching
)"
if [ -n "$dirty" ]; then
  echo "kapsl-sdk checkout must be clean before managed-vLLM packaging:" >&2
  printf '%s\n' "$dirty" >&2
  exit 1
fi
