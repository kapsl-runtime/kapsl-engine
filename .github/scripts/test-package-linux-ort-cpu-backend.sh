#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
packager="$repo_root/.github/scripts/package-linux-ort-cpu-backend.sh"
test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

integrations_dir="$test_root/integrations"
build_script="$integrations_dir/integrations/ort/packaging/build_cpu_pack.sh"
mkdir -p "$(dirname "$build_script")"
cat > "$build_script" <<'BUILD'
#!/usr/bin/env bash
set -euo pipefail
: "${KAPSL_VERSION:?}"
: "${KAPSL_ORT_PACK_OUTPUT_DIR:?}"
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source_ref="$(git -C "$repo_root" rev-parse HEAD)"
SOURCE_REF="$source_ref" python3 - <<'PY'
import gzip
import hashlib
import io
import json
import os
import pathlib
import tarfile

version = os.environ["KAPSL_VERSION"]
source_ref = os.environ["SOURCE_REF"]
output_dir = pathlib.Path(os.environ["KAPSL_ORT_PACK_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
adapter_abi = "invalid-adapter" if os.environ.get("FIXTURE_BAD_ADAPTER") else "kapsl-backend-v1"
payload = {
    "schema_version": 1,
    "backend": "onnx",
    "profile": "cpu",
    "pack_version": "0.1.0",
    "runtime_abi": 1,
    "adapter_abi": adapter_abi,
    "platform": "linux-x86_64",
    "execution_mode": "native",
    "entrypoint": "libkapsl_backend_ort.so",
}
provenance = {
    "schema_version": 1,
    "source_repository": "https://github.com/kapsl-runtime/kapsl-integrations",
    "source_commit": source_ref,
    "adapter": {"adapter_abi": adapter_abi},
}
entries = {
    "backend-pack.json": (json.dumps(payload, sort_keys=True) + "\n").encode(),
    "libkapsl_backend_ort.so": b"fixture standard ABI entrypoint\n",
    "licenses/FIXTURE-LICENSE": b"fixture license\n",
    "provenance.json": (json.dumps(provenance, sort_keys=True) + "\n").encode(),
}
name = f"kapsl-backend-onnx-cpu-{version}-linux-x86_64.tar.gz"
archive_path = output_dir / name
with archive_path.open("wb") as output:
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=1_700_000_000) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for relative, data in sorted(entries.items()):
                info = tarfile.TarInfo(relative)
                info.size = len(data)
                info.mode = 0o755 if relative.endswith(".so") else 0o644
                info.mtime = 1_700_000_000
                archive.addfile(info, io.BytesIO(data))
manifest = {
    **payload,
    "compatible_kapsl": f"={version}",
    "architecture": "x86_64",
    "accelerator_profile": "cpu",
    "installed_bytes": sum(map(len, entries.values())),
    "memory": {},
    "installer": {"kind": "extract"},
    "files": {key: hashlib.sha256(value).hexdigest() for key, value in entries.items()},
    "licenses": [{"name": "Fixture", "path": "licenses/FIXTURE-LICENSE"}],
    "priority": 200,
}
(output_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
(output_dir / f"{name}.sha256").write_text(f"{digest}  {name}\n")
PY
BUILD
chmod +x "$build_script"

git -C "$integrations_dir" init -q
git -C "$integrations_dir" config user.email test@example.invalid
git -C "$integrations_dir" config user.name "Kapsl test"
git -C "$integrations_dir" add .
git -C "$integrations_dir" commit -qm fixture
integrations_ref="$(git -C "$integrations_dir" rev-parse HEAD)"
output_dir="$test_root/dist"

run_packager() {
  RUNNER_OS=Linux \
  RUNNER_ARCH=X64 \
  RUNNER_TEMP="$test_root/runner" \
  KAPSL_VERSION=1.2.3 \
  KAPSL_ORT_INTEGRATIONS_DIR="$integrations_dir" \
  KAPSL_ORT_INTEGRATIONS_REF="$integrations_ref" \
  KAPSL_ORT_RELEASE_OUTPUT_DIR="$output_dir" \
    "$packager"
}

run_packager
archive="$output_dir/kapsl-backend-onnx-cpu-1.2.3-linux-x86_64.tar.gz"
[ -s "$archive" ]
[ -s "${archive}.manifest.json" ]
[ -s "${archive}.sha256" ]
[ ! -e "${archive}.sig" ]

openssl genpkey -algorithm ED25519 -out "$test_root/key.pem" >/dev/null 2>&1
public_key="$(openssl pkey -in "$test_root/key.pem" -pubout -outform DER \
  | tail -c 32 | base64 | tr -d '\n')"
"$repo_root/.github/scripts/generate-backend-index.py" \
  --version 1.2.3 \
  --artifacts-dir "$output_dir" \
  --output "$output_dir/backend-index.json" \
  --signing-key "$test_root/key.pem" \
  --expected-public-key "$public_key"
python3 - "$output_dir/backend-index.json" "$integrations_ref" <<'PY'
import json
import pathlib
import sys

index = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert len(index["packs"]) == 1
pack = index["packs"][0]
assert pack["adapter_abi"] == "kapsl-backend-v1"
assert pack["backend"] == "onnx" and pack["profile"] == "cpu"
assert pack["entrypoint"] == "libkapsl_backend_ort.so"
PY

if run_packager >/dev/null 2>&1; then
  echo "ORT CPU packager unexpectedly overwrote an existing release output" >&2
  exit 1
fi
rm -rf "$output_dir"
if FIXTURE_BAD_ADAPTER=1 run_packager >/dev/null 2>&1; then
  echo "ORT CPU packager unexpectedly accepted an unrecognized adapter ABI" >&2
  exit 1
fi

echo "Linux ORT CPU release handoff tests passed."
