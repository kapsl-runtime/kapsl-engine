#!/usr/bin/env bash
set -euo pipefail

verifier=".github/scripts/verify-managed-vllm-sdk-checkout.sh"
test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

sdk_dir="$test_root/sdk"
mkdir -p "$sdk_dir"
git -C "$sdk_dir" init -q
git -C "$sdk_dir" config user.email test@example.invalid
git -C "$sdk_dir" config user.name "Kapsl test"
printf 'fixture\n' > "$sdk_dir/README.md"
printf 'ignored.txt\n' > "$sdk_dir/.gitignore"
git -C "$sdk_dir" add README.md .gitignore
git -C "$sdk_dir" commit -qm fixture
sdk_ref="$(git -C "$sdk_dir" rev-parse HEAD)"

"$verifier" "$sdk_dir" "$sdk_ref"

expect_failure() {
  if "$verifier" "$1" "$2" >/dev/null 2>&1; then
    echo "managed-vLLM SDK verifier unexpectedly accepted: dir=$1 ref=$2" >&2
    exit 1
  fi
}

expect_failure "$sdk_dir" main
expect_failure "$sdk_dir" "${sdk_ref:0:12}"
expect_failure "$sdk_dir" 0000000000000000000000000000000000000000

printf 'dirty\n' >> "$sdk_dir/README.md"
expect_failure "$sdk_dir" "$sdk_ref"
git -C "$sdk_dir" restore README.md

printf 'untracked\n' > "$sdk_dir/untracked.txt"
expect_failure "$sdk_dir" "$sdk_ref"
rm "$sdk_dir/untracked.txt"

printf 'ignored\n' > "$sdk_dir/ignored.txt"
expect_failure "$sdk_dir" "$sdk_ref"
rm "$sdk_dir/ignored.txt"

"$verifier" "$sdk_dir" "$sdk_ref"
echo "Managed-vLLM SDK checkout verifier tests passed."
