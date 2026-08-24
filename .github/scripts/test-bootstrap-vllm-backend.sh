#!/usr/bin/env bash
set -euo pipefail

test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

bootstrap_root="$test_root/bootstrap"
target_root="$test_root/install/backends/vllm"
mkdir -p "$bootstrap_root/python/bin" "$bootstrap_root/wheels" "$target_root"

cat > "$bootstrap_root/python/bin/python3.12" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 0 ]; then
  cat >/dev/null
  printf '%s\n' '{"connector":"0.5.0","profile":"vllm-v1-packed-cuda-ipc/flash-attn"}'
  exit 0
fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then
  exit 0
fi
echo "unexpected fake Python invocation: $*" >&2
exit 1
EOF
chmod 755 "$bootstrap_root/python/bin/python3.12"
printf 'fixture wheel\n' > "$bootstrap_root/wheels/fixture.whl"
printf 'fixture==1.0\n' > "$bootstrap_root/requirements.lock"
(cd "$bootstrap_root" && {
  find python wheels -type f -print0
  printf 'requirements.lock\0'
} | sort -z | xargs -0 sha256sum > SHA256SUMS)

printf 'old installation\n' > "$target_root/old-installation"
.github/scripts/bootstrap-vllm-backend.sh "$bootstrap_root" "$target_root"

test -x "$target_root/bin/python"
test -x "$target_root/bin/python3"
test -x "$target_root/bin/python3.12"
test -f "$target_root/kapsl-vllm-backend.json"
test ! -e "$target_root/old-installation"
grep -qF 'vllm-v1-packed-cuda-ipc/flash-attn' "$target_root/kapsl-vllm-backend.json"

printf 'corrupt\n' >> "$bootstrap_root/wheels/fixture.whl"
set +e
.github/scripts/bootstrap-vllm-backend.sh "$bootstrap_root" "$target_root" >/dev/null 2>&1
status=$?
set -e
if [ "$status" -eq 0 ]; then
  echo "A corrupted managed-vLLM bootstrap unexpectedly succeeded." >&2
  exit 1
fi
test -x "$target_root/bin/python"
test -f "$target_root/kapsl-vllm-backend.json"

echo "Managed-vLLM backend bootstrap tests passed."
