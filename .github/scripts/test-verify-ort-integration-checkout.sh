#!/usr/bin/env bash
set -euo pipefail

verifier=".github/scripts/verify-ort-integration-checkout.sh"
test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

integrations_dir="$test_root/integrations"
mkdir -p "$integrations_dir"
git -C "$integrations_dir" init -q
git -C "$integrations_dir" config user.email test@example.invalid
git -C "$integrations_dir" config user.name "Kapsl test"
printf 'fixture\n' > "$integrations_dir/README.md"
printf 'ignored.txt\n' > "$integrations_dir/.gitignore"
git -C "$integrations_dir" add README.md .gitignore
git -C "$integrations_dir" commit -qm fixture
integrations_ref="$(git -C "$integrations_dir" rev-parse HEAD)"

"$verifier" "$integrations_dir" "$integrations_ref"

expect_failure() {
  if "$verifier" "$1" "$2" >/dev/null 2>&1; then
    echo "ORT integrations verifier unexpectedly accepted: dir=$1 ref=$2" >&2
    exit 1
  fi
}

expect_failure "$integrations_dir" main
expect_failure "$integrations_dir" "${integrations_ref:0:12}"
expect_failure "$integrations_dir" 0000000000000000000000000000000000000000

printf 'dirty\n' >> "$integrations_dir/README.md"
expect_failure "$integrations_dir" "$integrations_ref"
git -C "$integrations_dir" restore README.md

printf 'untracked\n' > "$integrations_dir/untracked.txt"
expect_failure "$integrations_dir" "$integrations_ref"
rm "$integrations_dir/untracked.txt"

printf 'ignored\n' > "$integrations_dir/ignored.txt"
expect_failure "$integrations_dir" "$integrations_ref"
rm "$integrations_dir/ignored.txt"

"$verifier" "$integrations_dir" "$integrations_ref"
echo "ORT integrations checkout verifier tests passed."
