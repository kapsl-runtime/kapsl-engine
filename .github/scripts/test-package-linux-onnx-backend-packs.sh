#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
test_root="$(mktemp -d)"
cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

mkdir -p \
  "$test_root/repo/.github/scripts" \
  "$test_root/repo/.github/licenses" \
  "$test_root/repo/ort-core-libs" \
  "$test_root/repo/ort-cuda-libs" \
  "$test_root/repo/ort-tensorrt-libs" \
  "$test_root/repo/cuda-runtime" \
  "$test_root/repo/tensorrt-runtime-libs" \
  "$test_root/repo/tensorrt-license-files" \
  "$test_root/bin" \
  "$test_root/runner"

cp "$repo_root/.github/scripts/package-linux-onnx-backend-packs.sh" \
  "$test_root/repo/.github/scripts/"
cp "$repo_root/.github/scripts/onnx-backend-pack-entrypoint.c" \
  "$test_root/repo/.github/scripts/"
cp "$repo_root/.github/licenses/ONNX-RUNTIME-LICENSE" \
  "$test_root/repo/.github/licenses/"
cp "$repo_root/LICENSE" "$test_root/repo/LICENSE"

printf 'ORT shared\n' > "$test_root/repo/ort-core-libs/libonnxruntime_providers_shared.so"
printf 'ORT CUDA\n' > "$test_root/repo/ort-cuda-libs/libonnxruntime_providers_cuda.so"
printf 'ORT TensorRT\n' > "$test_root/repo/ort-tensorrt-libs/libonnxruntime_providers_tensorrt.so"
printf 'cuDNN\n' > "$test_root/repo/cuda-runtime/libcudnn.so.9"
printf 'CUDA runtime\n' > "$test_root/repo/cuda-runtime/libcudart.so.12"
# The full CUDA runtime archive also contains ORT sidecars. Use deliberately
# different bytes so this fixture catches accidental copying or conflicts with
# the authoritative provider-specific staging directories above.
printf 'Bundled ORT shared copy\n' > "$test_root/repo/cuda-runtime/libonnxruntime_providers_shared.so"
printf 'Bundled ORT CUDA copy\n' > "$test_root/repo/cuda-runtime/libonnxruntime_providers_cuda.so"
printf 'NVIDIA license\n' > "$test_root/repo/cuda-runtime/NVIDIA-CONTAINER-LICENSE"
printf 'TensorRT\n' > "$test_root/repo/tensorrt-runtime-libs/libnvinfer.so.10"
printf 'TensorRT parser\n' > "$test_root/repo/tensorrt-runtime-libs/libnvonnxparser.so.10"
printf 'TensorRT license\n' > "$test_root/repo/tensorrt-license-files/LICENSE.txt"

cat > "$test_root/bin/patchelf" <<'EOF'
#!/usr/bin/env sh
exit 0
EOF
chmod +x "$test_root/bin/patchelf"

(
  cd "$test_root/repo"
  PATH="$test_root/bin:$PATH" \
  RUNNER_OS=Linux \
  RUNNER_ARCH=X64 \
  RUNNER_TEMP="$test_root/runner" \
  KAPSL_VERSION=1.2.3 \
  KAPSL_CUDA_RUNTIME_ROOT="$test_root/repo/cuda-runtime" \
  .github/scripts/package-linux-onnx-backend-packs.sh
)

openssl genpkey -algorithm ED25519 -out "$test_root/backend-key.pem" >/dev/null 2>&1
public_key="$(openssl pkey -in "$test_root/backend-key.pem" -pubout -outform DER \
  | tail -c 32 | base64 | tr -d '\n')"
"$repo_root/.github/scripts/generate-backend-index.py" \
  --version 1.2.3 \
  --artifacts-dir "$test_root/repo/dist" \
  --output "$test_root/repo/dist/backend-index.json" \
  --signing-key "$test_root/backend-key.pem" \
  --expected-public-key "$public_key"

python3 - "$test_root/repo/dist" "$test_root/extracted" <<'PY'
import hashlib
import ctypes
import json
import pathlib
import sys
import tarfile

dist = pathlib.Path(sys.argv[1])
extracted = pathlib.Path(sys.argv[2])
profiles = {
    "cpu": "cpu",
    "cuda12": "cuda",
    "tensorrt10": "tensorrt",
}
index = json.loads((dist / "backend-index.json").read_text())
assert {(pack["backend"], pack["profile"]) for pack in index["packs"]} == {
    ("onnx", "cpu"),
    ("onnx", "cuda12"),
    ("onnx", "tensorrt10"),
}
for profile, accelerator in profiles.items():
    archive = dist / f"kapsl-backend-onnx-{profile}-1.2.3-linux-x86_64.tar.gz"
    template_path = pathlib.Path(str(archive) + ".manifest.json")
    assert archive.is_file(), archive
    assert pathlib.Path(str(archive) + ".sha256").is_file()
    template = json.loads(template_path.read_text())
    assert template["backend"] == "onnx"
    assert template["profile"] == profile
    assert template["accelerator_profile"] == accelerator
    assert template["execution_mode"] == "native"
    assert template["entrypoint"] == "libkapsl_backend_onnx.so"
    assert template["compatible_kapsl"] == "=1.2.3"
    assert template["files"]["libkapsl_backend_onnx.so"]
    assert template["licenses"]

    root = extracted / profile
    root.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root)
    payload = json.loads((root / "backend-pack.json").read_text())
    assert payload["profile"] == profile
    for relative, expected in template["files"].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("magic", ctypes.c_uint32),
            ("struct_size", ctypes.c_uint32),
            ("runtime_abi", ctypes.c_uint32),
            ("profile", ctypes.c_uint32),
        ]
    native = ctypes.CDLL(str(root / "libkapsl_backend_onnx.so"))
    native.kapsl_onnx_backend_pack_v1.restype = ctypes.POINTER(Descriptor)
    descriptor = native.kapsl_onnx_backend_pack_v1().contents
    assert descriptor.magic == 0x4B4F4E58
    assert descriptor.runtime_abi == 1
    assert descriptor.profile == {"cpu": 1, "cuda12": 2, "tensorrt10": 3}[profile]

cpu = extracted / "cpu"
assert not (cpu / "libonnxruntime_providers_cuda.so").exists()
assert not (cpu / "libcudnn.so.9").exists()
cuda = extracted / "cuda12"
assert (cuda / "kapsl-provider-cuda12.json").is_file()
assert (cuda / "libonnxruntime_providers_shared.so").read_text() == "ORT shared\n"
assert (cuda / "libonnxruntime_providers_cuda.so").read_text() == "ORT CUDA\n"
assert (cuda / "libcudnn.so.9").is_file()
assert (cuda / "libcudart.so.12").is_file()
assert not (cuda / "libonnxruntime_providers_tensorrt.so").exists()
tensorrt = extracted / "tensorrt10"
assert (tensorrt / "kapsl-provider-cuda12.json").is_file()
assert (tensorrt / "kapsl-provider-tensorrt10.json").is_file()
assert (tensorrt / "libnvinfer.so.10").is_file()
PY

if command -v nm >/dev/null 2>&1; then
  nm "$test_root/extracted/cpu/libkapsl_backend_onnx.so" \
    | grep -q 'kapsl_onnx_backend_pack_v1'
fi

echo "Linux ONNX backend packager tests passed."
