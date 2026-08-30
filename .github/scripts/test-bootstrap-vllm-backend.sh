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
  printf '%s\n' '{"connector":"0.7.0","profile":"vllm-v1-packed-cuda-ipc/flash-attn","elastic_profile":"vllm-v1-packed-cuda-vmm/flash-attn-blnhc","kv_abi":{"major":1,"minor":5},"planner_schema_version":1}'
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
  "sdk_ref": "1111111111111111111111111111111111111111",
  "python": "3.12.3",
  "torch": "2.13.0+cu130",
  "torchvision": "0.28.0+cu130",
  "torchaudio": "2.11.0+cu130",
  "cuda_runtime": "13.0",
  "vllm": "0.26.1rc1.dev1130+g2ec6f0d71",
  "connector_distribution": "0.7.0",
  "connector": "0.7.0",
  "profile": "vllm-v1-packed-cuda-ipc/flash-attn",
  "elastic_profile": "vllm-v1-packed-cuda-vmm/flash-attn-blnhc",
  "kv_abi": {"major": 1, "minor": 5},
  "planner_schema_version": 1
}
EOF
(cd "$bootstrap_root" && {
  find python wheels -type f -print0
  printf 'requirements.lock\0installed-manifest.json\0'
} | sort -z | xargs -0 sha256sum > SHA256SUMS)

printf 'old installation\n' > "$target_root/old-installation"
.github/scripts/bootstrap-vllm-backend.sh "$bootstrap_root" "$target_root"

test -x "$target_root/bin/python"
test -x "$target_root/bin/python3"
test -x "$target_root/bin/python3.12"
test -f "$target_root/kapsl-vllm-backend.json"
test ! -e "$target_root/old-installation"
grep -qF 'vllm-v1-packed-cuda-ipc/flash-attn' "$target_root/kapsl-vllm-backend.json"
grep -qF 'vllm-v1-packed-cuda-vmm/flash-attn-blnhc' "$target_root/kapsl-vllm-backend.json"
grep -qF '"connector": "0.7.0"' "$target_root/kapsl-vllm-backend.json"
grep -qF '"kv_abi": {"major": 1, "minor": 5}' "$target_root/kapsl-vllm-backend.json"
grep -qF '"planner_schema_version": 1' "$target_root/kapsl-vllm-backend.json"

python3 - "$bootstrap_root/installed-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["connector"] = "0.5.0"
path.write_text(json.dumps(manifest), encoding="utf-8")
PY
(cd "$bootstrap_root" && {
  find python wheels -type f -print0
  printf 'requirements.lock\0installed-manifest.json\0'
} | sort -z | xargs -0 sha256sum > SHA256SUMS)
set +e
.github/scripts/bootstrap-vllm-backend.sh "$bootstrap_root" "$target_root" >/dev/null 2>&1
status=$?
set -e
if [ "$status" -eq 0 ]; then
  echo "A mismatched managed-vLLM installed manifest unexpectedly succeeded." >&2
  exit 1
fi
grep -qF '"connector": "0.7.0"' "$target_root/kapsl-vllm-backend.json"

python3 - "$bootstrap_root/installed-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["connector"] = "0.7.0"
path.write_text(json.dumps(manifest), encoding="utf-8")
PY
(cd "$bootstrap_root" && {
  find python wheels -type f -print0
  printf 'requirements.lock\0installed-manifest.json\0'
} | sort -z | xargs -0 sha256sum > SHA256SUMS)

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
