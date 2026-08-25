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
  "$test_root/repo/kapsl-runtime/include" \
  "$test_root/bin" \
  "$test_root/runner"

printf 'fixture NVIDIA redistribution terms\n' > "$test_root/NVIDIA-CONTAINER-LICENSE"

cp "$repo_root/.github/scripts/package-linux-llama-cpp-backend-packs.sh" \
  "$test_root/repo/.github/scripts/"
cp "$repo_root/.github/licenses/LLAMA-CPP-LICENSE" "$test_root/repo/.github/licenses/"
cp "$repo_root/kapsl-runtime/include/kapsl_llama_cpp_backend.h" \
  "$test_root/repo/kapsl-runtime/include/"
cp "$repo_root/LICENSE" "$repo_root/NOTICE" "$test_root/repo/"
cp "$repo_root/kapsl-runtime/Cargo.toml" "$test_root/repo/kapsl-runtime/Cargo.toml"

cat > "$test_root/fake-pack.c" <<'EOF'
#include "kapsl_llama_cpp_backend.h"

#ifndef TEST_CAPABILITY
#define TEST_CAPABILITY KAPSL_LLAMA_CAP_CPU
#endif
#ifndef TEST_KV_CAPABILITY
#define TEST_KV_CAPABILITY KAPSL_LLAMA_CAP_NATIVE_KV
#endif

#if TEST_KV_CAPABILITY == KAPSL_LLAMA_CAP_SHARED_POOL
__attribute__((used, visibility("default")))
const char KAPSL_LLAMA_CPP_KV_MODE_V1[] = "KAPSL_LLAMA_CPP_KV_MODE=shared_pool";
#else
__attribute__((used, visibility("default")))
const char KAPSL_LLAMA_CPP_KV_MODE_V1[] = "KAPSL_LLAMA_CPP_KV_MODE=native";
#endif

static const kapsl_llama_cpp_api_v1 API = {
    .magic = KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC,
    .abi_version = KAPSL_LLAMA_CPP_ABI_VERSION,
    .struct_size = sizeof(kapsl_llama_cpp_api_v1),
    .wire_format = KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1,
    .capabilities = TEST_CAPABILITY | TEST_KV_CAPABILITY,
};

const kapsl_llama_cpp_api_v1 *kapsl_llama_cpp_backend_v1(void) {
    return &API;
}
EOF
cc -shared -fPIC -O2 \
  -I "$test_root/repo/kapsl-runtime/include" \
  "$test_root/fake-pack.c" \
  -o "$test_root/libllama-cpu.so"
cat > "$test_root/fake-cudart.c" <<'EOF'
int kapsl_fixture_cuda_runtime(void) { return 12; }
EOF
cc -shared -fPIC -O2 \
  -Wl,-soname,libcudart.so.12 \
  "$test_root/fake-cudart.c" \
  -o "$test_root/libcudart.so.12"
cat >> "$test_root/fake-pack.c" <<'EOF'
#ifdef TEST_CUDA_LINK
extern int kapsl_fixture_cuda_runtime(void);
int kapsl_fixture_cuda_link(void) { return kapsl_fixture_cuda_runtime(); }
#endif
EOF
cc -shared -fPIC -O2 \
  -DTEST_CAPABILITY=KAPSL_LLAMA_CAP_CUDA \
  -DTEST_CUDA_LINK=1 \
  -I "$test_root/repo/kapsl-runtime/include" \
  "$test_root/fake-pack.c" \
  -L "$test_root" \
  -Wl,-rpath,"$test_root" \
  -Wl,--no-as-needed \
  -Wl,-l:libcudart.so.12 \
  -o "$test_root/libllama-cuda.so"

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
  KAPSL_NVIDIA_LICENSE_FILE="$test_root/NVIDIA-CONTAINER-LICENSE" \
  KAPSL_LLAMA_CPU_LIBRARY="$test_root/libllama-cpu.so" \
  KAPSL_LLAMA_CUDA_LIBRARY="$test_root/libllama-cuda.so" \
  .github/scripts/package-linux-llama-cpp-backend-packs.sh
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
import ctypes
import hashlib
import json
import pathlib
import sys
import tarfile

dist = pathlib.Path(sys.argv[1])
extracted = pathlib.Path(sys.argv[2])
index = json.loads((dist / "backend-index.json").read_text())
assert {(pack["backend"], pack["profile"]) for pack in index["packs"]} == {
    ("llama-cpp", "cpu"),
    ("llama-cpp", "cuda12"),
}

class ApiPrefix(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("wire_format", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint64),
    ]

for profile, accelerator, capability in (("cpu", "cpu", 1), ("cuda12", "cuda", 2)):
    archive = dist / f"kapsl-backend-llama-cpp-{profile}-1.2.3-linux-x86_64.tar.gz"
    template = json.loads(pathlib.Path(str(archive) + ".manifest.json").read_text())
    assert template["backend"] == "llama-cpp"
    assert template["profile"] == profile
    assert template["accelerator_profile"] == accelerator
    assert template["execution_mode"] == "native"
    assert template["kv_mode"] == "native"
    assert template["entrypoint"] == "lib/libkapsl_backend_llama_cpp.so"
    assert template["compatible_kapsl"] == "=1.2.3"
    assert len(template["licenses"]) == (3 if profile == "cpu" else 4)
    root = extracted / profile
    root.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(root)
    payload = json.loads((root / "backend-pack.json").read_text())
    assert payload["profile"] == profile
    assert payload["kv_mode"] == "native"
    assert (root / "include/kapsl_llama_cpp_backend.h").is_file()
    if profile == "cpu":
        assert not (root / "lib/libcudart.so.12").exists()
    else:
        assert (root / "lib/libcudart.so.12").is_file()
        assert (root / "licenses/NVIDIA-CONTAINER-LICENSE").is_file()
    for relative, expected in template["files"].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
    native = ctypes.CDLL(str(root / "lib/libkapsl_backend_llama_cpp.so"))
    native.kapsl_llama_cpp_backend_v1.restype = ctypes.POINTER(ApiPrefix)
    api = native.kapsl_llama_cpp_backend_v1().contents
    assert api.magic == 0x4B4C4C4D
    assert api.abi_version == 1
    assert api.wire_format == 1
    assert api.capabilities & capability
    assert api.capabilities & 4

assert "minimum_cuda" not in next(
    pack for pack in index["packs"] if pack["profile"] == "cpu"
)
assert next(pack for pack in index["packs"] if pack["profile"] == "cuda12")["minimum_cuda"] == "12.0"
PY

cc -shared -fPIC -O2 \
  -DTEST_CAPABILITY=KAPSL_LLAMA_CAP_CUDA \
  -DTEST_KV_CAPABILITY=KAPSL_LLAMA_CAP_SHARED_POOL \
  -DTEST_CUDA_LINK=1 \
  -I "$test_root/repo/kapsl-runtime/include" \
  "$test_root/fake-pack.c" \
  -L "$test_root" \
  -Wl,-rpath,"$test_root" \
  -Wl,--no-as-needed \
  -Wl,-l:libcudart.so.12 \
  -o "$test_root/libllama-cuda-shared.so"

mkdir -p \
  "$test_root/sdk/crates/kapsl-llm" \
  "$test_root/sdk/crates/kapsl-engine-api"
printf '[package]\nname = "fixture-kapsl-llm"\nversion = "0.0.0"\n' \
  > "$test_root/sdk/crates/kapsl-llm/Cargo.toml"
printf '[package]\nname = "fixture-kapsl-engine-api"\nversion = "0.0.0"\n' \
  > "$test_root/sdk/crates/kapsl-engine-api/Cargo.toml"
git -C "$test_root/sdk" init -q
git -C "$test_root/sdk" config user.name 'Kapsl Test'
git -C "$test_root/sdk" config user.email 'test@kapsl.invalid'
git -C "$test_root/sdk" add .
git -C "$test_root/sdk" commit -qm fixture
sdk_ref="$(git -C "$test_root/sdk" rev-parse HEAD)"

(
  cd "$test_root/repo"
  PATH="$test_root/bin:$PATH" \
  RUNNER_OS=Linux \
  RUNNER_ARCH=X64 \
  RUNNER_TEMP="$test_root/runner" \
  KAPSL_VERSION=1.2.3 \
  KAPSL_NVIDIA_LICENSE_FILE="$test_root/NVIDIA-CONTAINER-LICENSE" \
  KAPSL_LLAMA_CPU_LIBRARY="$test_root/libllama-cpu.so" \
  KAPSL_LLAMA_CUDA_LIBRARY="$test_root/libllama-cuda-shared.so" \
  KAPSL_LLAMA_SDK_DIR="$test_root/sdk" \
  KAPSL_LLAMA_SDK_REF="$sdk_ref" \
  .github/scripts/package-linux-llama-cpp-backend-packs.sh
)

python3 - "$test_root/repo/dist" "$test_root/extracted-shared" <<'PY'
import ctypes
import json
import pathlib
import sys
import tarfile

dist = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
archive = dist / "kapsl-backend-llama-cpp-cuda12-1.2.3-linux-x86_64.tar.gz"
template = json.loads(pathlib.Path(str(archive) + ".manifest.json").read_text())
assert template["kv_mode"] == "shared_pool"
root.mkdir(parents=True)
with tarfile.open(archive, "r:gz") as bundle:
    bundle.extractall(root)
payload = json.loads((root / "backend-pack.json").read_text())
assert payload["kv_mode"] == "shared_pool"

class ApiPrefix(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("wire_format", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint64),
    ]

native = ctypes.CDLL(str(root / "lib/libkapsl_backend_llama_cpp.so"))
native.kapsl_llama_cpp_backend_v1.restype = ctypes.POINTER(ApiPrefix)
api = native.kapsl_llama_cpp_backend_v1().contents
assert api.capabilities & 2
assert api.capabilities & 8
assert not api.capabilities & 4
PY

echo "Linux llama.cpp backend packager tests passed."
