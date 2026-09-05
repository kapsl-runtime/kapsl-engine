#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
packager="$repo_root/.github/scripts/package-linux-ort-accelerator-backends.sh"
test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

integrations_dir="$test_root/integrations"
build_script="$integrations_dir/integrations/ort/packaging/build_accelerator_packs.sh"
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
profiles = {
    "cuda12": ("cuda", False),
    "tensorrt10": ("tensorrt", True),
}
for profile, (accelerator, include_tensorrt) in profiles.items():
    libraries = {
        "libkapsl_backend_ort.so": b"fixture adapter\n",
        "libonnxruntime.so.1": b"fixture runtime\n",
        "libonnxruntime_providers_shared.so": b"fixture shared provider\n",
        "libonnxruntime_providers_cuda.so": b"fixture CUDA provider\n",
        "libcublas.so.12": b"fixture cuBLAS\n",
        "libcudart.so.12": b"fixture CUDA runtime\n",
        "libcudnn.so.9": b"fixture cuDNN\n",
    }
    if include_tensorrt:
        libraries.update(
            {
                "libonnxruntime_providers_tensorrt.so": b"fixture TensorRT provider\n",
                "libnvinfer.so.10": b"fixture nvinfer\n",
                "libnvonnxparser.so.10": b"fixture parser\n",
            }
        )
    adapter_abi = "invalid" if os.environ.get("FIXTURE_BAD_ADAPTER") else "kapsl-backend-v1"
    execution_providers = ["cuda"]
    if include_tensorrt:
        execution_providers = ["tensorrt", "cuda"]
    contract = {
        "formats": ["onnx"],
        "tasks": ["forward", "embed", "classify", "detect", "transcribe", "generate"],
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
            "kind": accelerator,
            "execution_providers": execution_providers,
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
    payload = {
        "schema_version": 1,
        "backend": "onnx",
        "profile": profile,
        "pack_version": "0.1.0",
        "runtime_abi": 1,
        "adapter_abi": adapter_abi,
        "platform": "linux-x86_64",
        "execution_mode": "native",
        "entrypoint": "libkapsl_backend_ort.so",
        **contract,
    }
    provenance_libraries = {
        name: {
            "sha256": hashlib.sha256(data).hexdigest(),
            "needed_libraries": (
                ["libonnxruntime.so.1"]
                if name == "libkapsl_backend_ort.so"
                else ["libc.so.6"]
            ),
            "maximum_required_glibc": "2.35",
            "runpath": "$ORIGIN",
        }
        for name, data in libraries.items()
    }
    provenance = {
        "schema_version": 1,
        "source_repository": "https://github.com/kapsl-runtime/kapsl-integrations",
        "source_commit": source_ref,
        "adapter": {"adapter_abi": adapter_abi},
        "profile": {"pack": profile},
        "onnx_runtime": {
            "version": "1.23.2",
            "distribution_url": (
                "https://github.com/microsoft/onnxruntime/releases/download/"
                "v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz"
            ),
            "distribution_sha256": (
                "2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53"
            ),
        },
        "libraries": provenance_libraries,
    }
    entries = {
        **libraries,
        "backend-pack.json": (json.dumps(payload, sort_keys=True) + "\n").encode(),
        "licenses/FIXTURE-LICENSE": b"fixture license\n",
        "provenance.json": (json.dumps(provenance, sort_keys=True) + "\n").encode(),
    }
    name = f"kapsl-backend-onnx-{profile}-{version}-linux-x86_64.tar.gz"
    archive_path = output_dir / name
    with archive_path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=1_700_000_000) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for relative, data in sorted(entries.items()):
                    info = tarfile.TarInfo(relative)
                    info.size = len(data)
                    info.mode = 0o755 if ".so" in relative else 0o644
                    info.mtime = 1_700_000_000
                    archive.addfile(info, io.BytesIO(data))
    manifest = {
        **payload,
        "compatible_kapsl": f"={version}",
        "architecture": "x86_64",
        "accelerator_profile": accelerator,
        "minimum_cuda": "12.0",
        "minimum_driver": "560.28.03",
        "installed_bytes": sum(map(len, entries.values())),
        "memory": {"accelerator_bytes": 1},
        "installer": {"kind": "extract"},
        "files": {key: hashlib.sha256(value).hexdigest() for key, value in entries.items()},
        "licenses": [{"name": "Fixture", "path": "licenses/FIXTURE-LICENSE"}],
        "priority": 200,
    }
    (output_dir / f"{name}.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
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
cuda_root="$test_root/cuda"
tensorrt_root="$test_root/tensorrt"
tensorrt_licenses="$test_root/tensorrt-licenses"
mkdir -p "$cuda_root" "$tensorrt_root" "$tensorrt_licenses" "$test_root/runner"
printf 'NVIDIA license\n' > "$cuda_root/NVIDIA-CONTAINER-LICENSE"
printf 'TensorRT runtime\n' > "$tensorrt_root/libnvinfer.so.10"
printf 'TensorRT license\n' > "$tensorrt_licenses/LICENSE.txt"

run_packager() {
  RUNNER_OS=Linux \
  RUNNER_ARCH=X64 \
  RUNNER_TEMP="$test_root/runner" \
  KAPSL_VERSION=1.2.3 \
  KAPSL_ORT_INTEGRATIONS_DIR="$integrations_dir" \
  KAPSL_ORT_INTEGRATIONS_REF="$integrations_ref" \
  KAPSL_ORT_RELEASE_OUTPUT_DIR="$output_dir" \
  KAPSL_CUDA_RUNTIME_ROOT="$cuda_root" \
  KAPSL_TENSORRT_RUNTIME_DIR="$tensorrt_root" \
  KAPSL_TENSORRT_LICENSE_DIR="$tensorrt_licenses" \
    "$packager"
}

run_packager
for profile in cuda12 tensorrt10; do
  artifact="kapsl-backend-onnx-${profile}-1.2.3-linux-x86_64.tar.gz"
  test -s "$output_dir/$artifact"
  test -s "$output_dir/$artifact.manifest.json"
  test -s "$output_dir/$artifact.sha256"
  test ! -e "$output_dir/$artifact.sig"
done

openssl genpkey -algorithm ED25519 -out "$test_root/key.pem" >/dev/null 2>&1
public_key="$(openssl pkey -in "$test_root/key.pem" -pubout -outform DER \
  | tail -c 32 | base64 | tr -d '\n')"
"$repo_root/.github/scripts/generate-backend-index.py" \
  --version 1.2.3 \
  --artifacts-dir "$output_dir" \
  --output "$output_dir/backend-index.json" \
  --signing-key "$test_root/key.pem" \
  --expected-public-key "$public_key"
python3 - "$output_dir/backend-index.json" <<'PY'
import json
import pathlib
import sys

index = json.loads(pathlib.Path(sys.argv[1]).read_text())
packs = {(pack["backend"], pack["profile"]): pack for pack in index["packs"]}
assert set(packs) == {("onnx", "cuda12"), ("onnx", "tensorrt10")}
assert all(pack["adapter_abi"] == "kapsl-backend-v1" for pack in packs.values())
assert all(pack["capabilities"]["scoped_device_allocator"] for pack in packs.values())
assert all(pack["memory_behavior"]["synchronize_before_free"] for pack in packs.values())
assert packs[("onnx", "tensorrt10")]["accelerator_requirements"]["execution_providers"][0] == "tensorrt"
PY

if run_packager >/dev/null 2>&1; then
  echo "ORT accelerator packager unexpectedly overwrote release outputs" >&2
  exit 1
fi
rm -rf "$output_dir"
if FIXTURE_BAD_ADAPTER=1 run_packager >/dev/null 2>&1; then
  echo "ORT accelerator packager accepted an invalid adapter ABI" >&2
  exit 1
fi

echo "Linux ORT accelerator release handoff tests passed."
