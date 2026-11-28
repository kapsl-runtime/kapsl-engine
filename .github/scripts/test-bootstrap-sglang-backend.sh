#!/usr/bin/env bash
set -euo pipefail

# Exercise bootstrap-sglang-backend.sh against a fake Python: a good install,
# a mismatched manifest that must leave the previous install in place, and a
# corrupted wheelhouse that must be refused.

test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

bootstrap_root="$test_root/bootstrap"
target_root="$test_root/install/backends/sglang"
mkdir -p "$bootstrap_root/python/bin" "$bootstrap_root/wheels" "$target_root"

cat > "$bootstrap_root/python/bin/python3.12" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 0 ]; then
  cat >/dev/null
  printf '%s\n' '{"adapter":"0.2.0","sglang":"0.5.23"}'
  exit 0
fi
if [ "${1:-}" = "-" ]; then
  exec python3 "$@"
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
cat > "$bootstrap_root/installed-manifest.json" <<'EOF'
{
  "schema_version": 1,
  "integrations_ref": "1111111111111111111111111111111111111111",
  "python": "3.12.3",
  "torch": "2.14.1",
  "torchvision": "0.29.2",
  "torchaudio": "2.11.0",
  "cuda_runtime": "13.0",
  "sglang": "0.5.23",
  "adapter_distribution": "0.2.0",
  "adapter": "0.2.0",
  "kv_abi": {"major": 1, "minor": 5},
  "participant_config_env": "KAPSL_KV_PARTICIPANT_CONFIG"
}
EOF
write_checksums() {
  (cd "$bootstrap_root" && {
    find python wheels -type f -print0
    printf 'requirements.lock\0installed-manifest.json\0'
  } | sort -z | xargs -0 sha256sum > SHA256SUMS)
}
set_manifest_field() { # key json-value
  python3 - "$bootstrap_root/installed-manifest.json" "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest[sys.argv[2]] = json.loads(sys.argv[3])
path.write_text(json.dumps(manifest), encoding="utf-8")
PY
}
expect_failure() { # message
  set +e
  .github/scripts/bootstrap-sglang-backend.sh "$bootstrap_root" "$target_root" >/dev/null 2>&1
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    echo "$1" >&2
    exit 1
  fi
}

write_checksums
printf 'old installation\n' > "$target_root/old-installation"
.github/scripts/bootstrap-sglang-backend.sh "$bootstrap_root" "$target_root"

test -x "$target_root/bin/python"
test -x "$target_root/bin/python3"
test -x "$target_root/bin/python3.12"
test -f "$target_root/kapsl-sglang-backend.json"
test ! -e "$target_root/old-installation"
grep -qF '"sglang": "0.5.23"' "$target_root/kapsl-sglang-backend.json"
grep -qF '"adapter": "0.2.0"' "$target_root/kapsl-sglang-backend.json"
grep -qF '"kv_abi": {"major": 1, "minor": 5}' "$target_root/kapsl-sglang-backend.json"
grep -qF '"participant_config_env": "KAPSL_KV_PARTICIPANT_CONFIG"' "$target_root/kapsl-sglang-backend.json"

# A manifest for another SGLang release is refused and the previous install
# stays in place.
set_manifest_field sglang '"0.5.20"'
write_checksums
expect_failure "A mismatched managed-SGLang installed manifest unexpectedly succeeded."
grep -qF '"sglang": "0.5.23"' "$target_root/kapsl-sglang-backend.json"

# So is an adapter for another KV ABI.
set_manifest_field sglang '"0.5.23"'
set_manifest_field kv_abi '{"major": 1, "minor": 4}'
write_checksums
expect_failure "A managed-SGLang manifest for another KV ABI unexpectedly succeeded."

# And an integrations ref that is not an exact commit.
set_manifest_field kv_abi '{"major": 1, "minor": 5}'
set_manifest_field integrations_ref '"feature/sglang-adapter"'
write_checksums
expect_failure "A managed-SGLang manifest without an exact integrations commit unexpectedly succeeded."

# A wheelhouse that no longer matches its checksums is refused.
set_manifest_field integrations_ref '"1111111111111111111111111111111111111111"'
write_checksums
printf 'corrupt\n' >> "$bootstrap_root/wheels/fixture.whl"
expect_failure "A corrupted managed-SGLang bootstrap unexpectedly succeeded."
test -x "$target_root/bin/python"
test -f "$target_root/kapsl-sglang-backend.json"

echo "Managed-SGLang backend bootstrap tests passed."
