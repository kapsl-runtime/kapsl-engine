#!/usr/bin/env bash
set -euo pipefail

root="$(mktemp -d)"
server_pid=""
cleanup() {
  if [ -n "$server_pid" ]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$root"
}
trap cleanup EXIT INT TERM

mkdir -p "$root/artifacts/payload/bin" "$root/artifacts/payload/licenses"
printf '#!/bin/sh\n' > "$root/artifacts/payload/bin/python"
chmod 755 "$root/artifacts/payload/bin/python"
printf 'fixture license\n' > "$root/artifacts/payload/licenses/LICENSE"
entrypoint_sha256="$(sha256sum "$root/artifacts/payload/bin/python" | awk '{ print $1 }')"
license_sha256="$(sha256sum "$root/artifacts/payload/licenses/LICENSE" | awk '{ print $1 }')"
cat > "$root/artifacts/payload/backend-pack.json" <<'EOF'
{"schema_version":1,"backend":"vllm","profile":"cu130-flash-attn","pack_version":"test","runtime_abi":1,"platform":"linux-x86_64","execution_mode":"external","entrypoint":"bin/python"}
EOF
tar -czf "$root/artifacts/pack.tar.gz" -C "$root/artifacts/payload" .
cat > "$root/artifacts/pack.tar.gz.manifest.json" <<EOF
{"schema_version":1,"backend":"vllm","profile":"cu130-flash-attn","pack_version":"test","runtime_abi":1,"compatible_kapsl":"=1.2.3","platform":"linux-x86_64","architecture":"x86_64","accelerator_profile":"cuda","minimum_cuda":"13.0","minimum_driver":"580.65.06","execution_mode":"external","entrypoint":"bin/python","installed_bytes":1024,"memory":{},"installer":{"kind":"extract"},"files":{"bin/python":"$entrypoint_sha256","licenses/LICENSE":"$license_sha256"},"licenses":[{"name":"Fixture","path":"licenses/LICENSE"}],"priority":1}
EOF

mkdir -p "$root/artifacts/payload-macos/bin" "$root/artifacts/payload-macos/licenses"
cp "$root/artifacts/payload/bin/python" "$root/artifacts/payload-macos/bin/python"
cp "$root/artifacts/payload/licenses/LICENSE" "$root/artifacts/payload-macos/licenses/LICENSE"
cat > "$root/artifacts/payload-macos/backend-pack.json" <<'EOF'
{"schema_version":1,"backend":"vllm","profile":"cu130-flash-attn","pack_version":"test","runtime_abi":1,"platform":"macos-aarch64","execution_mode":"external","entrypoint":"bin/python"}
EOF
tar -czf "$root/artifacts/pack-macos.tar.gz" -C "$root/artifacts/payload-macos" .
cat > "$root/artifacts/pack-macos.tar.gz.manifest.json" <<EOF
{"schema_version":1,"backend":"vllm","profile":"cu130-flash-attn","pack_version":"test","runtime_abi":1,"compatible_kapsl":"=1.2.3","platform":"macos-aarch64","architecture":"aarch64","accelerator_profile":"cuda","minimum_cuda":"13.0","minimum_driver":"580.65.06","execution_mode":"external","entrypoint":"bin/python","installed_bytes":1024,"memory":{},"installer":{"kind":"extract"},"files":{"bin/python":"$entrypoint_sha256","licenses/LICENSE":"$license_sha256"},"licenses":[{"name":"Fixture","path":"licenses/LICENSE"}],"priority":1}
EOF
openssl genpkey -algorithm ED25519 -out "$root/key.pem" >/dev/null 2>&1
public_key="$(openssl pkey -in "$root/key.pem" -pubout -outform DER | tail -c 32 | base64 | tr -d '\n')"
public_base="http://127.0.0.1:18082"

.github/scripts/generate-backend-index.py \
  --version 1.2.3 \
  --artifacts-dir "$root/artifacts" \
  --output "$root/artifacts/backend-index.json" \
  --signing-key "$root/key.pem" \
  --expected-public-key "$public_key" \
  --base-url "$public_base" \
  --allow-insecure-test-url \
  --channel beta

python3 - "$root/artifacts/backend-index.json" <<'PY'
import json
import pathlib
import sys
index = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert index["runtime_version"] == "1.2.3"
assert len(index["packs"]) == 2
assert {pack["platform"] for pack in index["packs"]} == {"linux-x86_64", "macos-aarch64"}
assert all(pack["signature"].startswith("ed25519:") for pack in index["packs"])
assert any(pack["artifact"].endswith("/runtime/beta/v1.2.3/pack.tar.gz") for pack in index["packs"])
assert any(pack["artifact"].endswith("/runtime/beta/v1.2.3/pack-macos.tar.gz") for pack in index["packs"])
PY
grep -q '^ed25519:' "$root/artifacts/backend-index.json.sig"

cp "$root/artifacts/pack.tar.gz.manifest.json" "$root/original-pack-manifest.json"
python3 - "$root/artifacts/pack.tar.gz.manifest.json" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["files"]["bin/python"] = "0" * 64
path.write_text(json.dumps(manifest) + "\n")
PY
if .github/scripts/generate-backend-index.py \
  --version 1.2.3 \
  --artifacts-dir "$root/artifacts" \
  --output "$root/artifacts/tampered-index.json" \
  --signing-key "$root/key.pem" \
  --expected-public-key "$public_key" \
  --channel beta 2>"$root/tampered-file.log"; then
  echo "index generation unexpectedly accepted a mismatched installed-file digest" >&2
  exit 1
fi
grep -q 'installed file digest mismatch' "$root/tampered-file.log"
mv "$root/original-pack-manifest.json" "$root/artifacts/pack.tar.gz.manifest.json"

public_release="$root/public/runtime/beta/v1.2.3"
mkdir -p "$public_release"
cp "$root/artifacts/pack.tar.gz" "$public_release/pack.tar.gz"
cp "$root/artifacts/pack-macos.tar.gz" "$public_release/pack-macos.tar.gz"
cp "$root/artifacts/backend-index.json" "$public_release/backend-index.json"
cp "$root/artifacts/backend-index.json.sig" "$public_release/backend-index.json.sig"
python3 -m http.server 18082 --bind 127.0.0.1 --directory "$root/public" \
  >"$root/http.log" 2>&1 &
server_pid="$!"
attempt=1
while ! curl -fsSI "$public_base/runtime/beta/v1.2.3/backend-index.json" >/dev/null; do
  if [ "$attempt" -ge 20 ]; then
    cat "$root/http.log" >&2
    echo "Timed out waiting for backend release fixture server." >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 1
done
.github/scripts/verify-public-backend-release.sh \
  "$root/artifacts/backend-index.json" \
  "$public_base/runtime/beta/v1.2.3"

openssl genpkey -algorithm ED25519 -out "$root/wrong-key.pem" >/dev/null 2>&1
wrong_public_key="$(openssl pkey -in "$root/wrong-key.pem" -pubout -outform DER | tail -c 32 | base64 | tr -d '\n')"
if .github/scripts/generate-backend-index.py \
  --version 1.2.3 \
  --artifacts-dir "$root/artifacts" \
  --output "$root/artifacts/wrong-index.json" \
  --signing-key "$root/key.pem" \
  --expected-public-key "$wrong_public_key" \
  --channel beta 2>"$root/wrong-key.log"; then
  echo "index generation unexpectedly accepted a mismatched embedded public key" >&2
  exit 1
fi
grep -q 'does not match any public key' "$root/wrong-key.log"

python3 - <<'PY'
import copy
import importlib.util
import pathlib

script = pathlib.Path(".github/scripts/generate-backend-index.py")
spec = importlib.util.spec_from_file_location("generate_backend_index", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
source = pathlib.Path("standard-native-fixture.json")

accelerator = {
    "formats": ["onnx"],
    "model_types": [],
    "tasks": ["forward", "generate"],
    "accelerator_profile": "cuda",
    "capabilities": {
        "batching": True,
        "streaming": True,
        "cancellation": True,
        "memory_reporting": True,
        "governed_device_allocator": True,
        "scoped_device_allocator": True,
        "kv_participation": False,
        "concurrent_inference": True,
    },
    "accelerator_requirements": {
        "kind": "cuda",
        "execution_providers": ["cuda"],
        "implicit_cpu_fallback": False,
    },
    "memory_behavior": {
        "allocation_scope": "kapsl-scoped-device-allocator-v1",
        "device_allocation": "host-governed-scoped",
        "planned_reporting": True,
        "live_reporting": True,
        "request_reporting": True,
        "synchronize_before_free": True,
    },
}
module.validate_standard_native_contract(accelerator, source)

cpu = copy.deepcopy(accelerator)
cpu["accelerator_profile"] = "cpu"
cpu["accelerator_requirements"] = {
    "kind": "cpu",
    "execution_providers": ["cpu"],
    "implicit_cpu_fallback": False,
}
cpu["capabilities"]["governed_device_allocator"] = False
cpu["capabilities"]["scoped_device_allocator"] = False
cpu["memory_behavior"]["allocation_scope"] = None
cpu["memory_behavior"]["device_allocation"] = "none"
cpu["memory_behavior"]["synchronize_before_free"] = False
module.validate_standard_native_contract(cpu, source)


def expect_failure(template, message):
    try:
        module.validate_standard_native_contract(template, source)
    except SystemExit as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"standard native contract unexpectedly accepted: {message}")


fallback = copy.deepcopy(accelerator)
fallback["accelerator_requirements"]["implicit_cpu_fallback"] = True
expect_failure(fallback, "disable implicit CPU fallback")

unscoped = copy.deepcopy(accelerator)
unscoped["memory_behavior"]["allocation_scope"] = None
expect_failure(unscoped, "name their allocation scope")

unreported = copy.deepcopy(accelerator)
unreported["memory_behavior"]["request_reporting"] = False
expect_failure(unreported, "planned, live, and request memory reporting")

unspecified_allocation = copy.deepcopy(accelerator)
unspecified_allocation["memory_behavior"]["device_allocation"] = None
expect_failure(unspecified_allocation, "device allocation behavior")
PY

echo "backend index generation tests passed"
