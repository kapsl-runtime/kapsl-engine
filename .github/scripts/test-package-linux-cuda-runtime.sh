#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_root="$(mktemp -d)"
work_dir="${test_root}/work"
fake_bin="${test_root}/bin"
fake_lib="${test_root}/lib"
runner_temp="${test_root}/runner"
nvidia_license="${test_root}/NGC-DL-CONTAINER-LICENSE"

cleanup() {
  rm -rf "$test_root"
}
trap cleanup EXIT INT TERM

mkdir -p \
  "$work_dir/.github/scripts" \
  "$work_dir/kapsl-runtime/target/release" \
  "$work_dir/ort-core-libs" \
  "$work_dir/ort-cuda-libs" \
  "$fake_bin" \
  "$fake_lib" \
  "$runner_temp"
printf 'NVIDIA fixture license\n' > "$nvidia_license"

cp "$repo_root/.github/scripts/package-linux-cuda-runtime.sh" \
  "$work_dir/.github/scripts/package-linux-cuda-runtime.sh"

for file in \
  libonnxruntime.so.1 \
  libonnxruntime_providers_shared.so; do
  printf '%s\n' "$file" > "$work_dir/ort-core-libs/$file"
done
printf 'cuda provider\n' > "$work_dir/ort-cuda-libs/libonnxruntime_providers_cuda.so"

cat > "$work_dir/kapsl-runtime/target/release/kapsl" <<'EOF'
#!/bin/sh
echo kapsl
EOF
chmod +x "$work_dir/kapsl-runtime/target/release/kapsl"

for file in \
  libnccl.so.2 \
  libcublasLt.so.12 \
  libcublas.so.12 \
  libcurand.so.10 \
  libcufft.so.11 \
  libcudart.so.12 \
  libcudnn.so.9 \
  libcudnn_ops.so.9 \
  libnvrtc.so.12; do
  printf '%s\n' "$file" > "$fake_lib/$file"
done

cat > "$fake_bin/readelf" <<'EOF'
#!/bin/sh
cat <<'OUTPUT'
 0x0000000000000001 (NEEDED)             Shared library: [libcuda.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libnccl.so.2]
OUTPUT
EOF

cat > "$fake_bin/ldd" <<EOF
#!/bin/sh
case "\$(basename "\$1")" in
  kapsl)
    echo 'libcuda.so.1 => not found'
    echo 'libnccl.so.2 => $fake_lib/libnccl.so.2 (0x1)'
    ;;
  libonnxruntime_providers_cuda.so)
    echo 'libcublasLt.so.12 => $fake_lib/libcublasLt.so.12 (0x1)'
    echo 'libcublas.so.12 => $fake_lib/libcublas.so.12 (0x1)'
    echo 'libcurand.so.10 => $fake_lib/libcurand.so.10 (0x1)'
    echo 'libcufft.so.11 => $fake_lib/libcufft.so.11 (0x1)'
    echo 'libcudart.so.12 => $fake_lib/libcudart.so.12 (0x1)'
    echo 'libcudnn.so.9 => $fake_lib/libcudnn.so.9 (0x1)'
    ;;
  libcudnn*.so.9)
    echo 'libnvrtc.so.12 => $fake_lib/libnvrtc.so.12 (0x1)'
    ;;
esac
echo 'libc.so.6 => /lib/libc.so.6 (0x1)'
EOF

cat > "$fake_bin/find" <<EOF
#!/bin/sh
if [ "\${1:-}" = /usr ]; then
  echo '$fake_lib/libcudnn.so.9'
  echo '$fake_lib/libcudnn_ops.so.9'
  exit 0
fi
exec /usr/bin/find "\$@"
EOF

cat > "$fake_bin/patchelf" <<'EOF'
#!/bin/sh
case "$1" in
  --set-rpath) exit 0 ;;
  --print-rpath) echo '$ORIGIN' ;;
  *) exit 1 ;;
esac
EOF

chmod +x "$fake_bin/readelf" "$fake_bin/ldd" "$fake_bin/find" "$fake_bin/patchelf"

run_packager() {
  (
    cd "$work_dir"
    PATH="$fake_bin:$PATH" \
    RUNNER_OS=Linux \
    RUNNER_ARCH=X64 \
    RUNNER_TEMP="$runner_temp" \
    KAPSL_VERSION=9.9.9 \
    KAPSL_NVIDIA_LICENSE_FILE="$nvidia_license" \
      bash .github/scripts/package-linux-cuda-runtime.sh
  )
}

missing_shared_kv_log="${test_root}/missing-shared-kv.log"
if run_packager >"$missing_shared_kv_log" 2>&1; then
  echo "CUDA packager accepted a binary without the shared-KV feature marker." >&2
  exit 1
fi
grep -q 'does not include the shared-KV GGUF path' "$missing_shared_kv_log"

cat > "$work_dir/kapsl-runtime/target/release/kapsl" <<'EOF'
#!/bin/sh
# Shared-KV feature marker used by the packager's profile assertion:
# KAPSL_GGUF_DISABLE_SHARED_KV
echo kapsl
EOF
chmod +x "$work_dir/kapsl-runtime/target/release/kapsl"

run_packager

archive="$work_dir/dist/kapsl-9.9.9-linux-x86_64-cuda12.tar.gz"
test -f "$archive"
contents="$(tar -tzf "$archive")"

for required in \
  kapsl \
  kapsl-provider-cuda12.json \
  NVIDIA-CONTAINER-LICENSE \
  libonnxruntime_providers_cuda.so \
  libonnxruntime_providers_shared.so \
  libnccl.so.2 \
  libcublas.so.12 \
  libcudnn.so.9 \
  libcudnn_ops.so.9 \
  libnvrtc.so.12; do
  if ! printf '%s\n' "$contents" | grep -qx "./$required"; then
    echo "CUDA archive is missing $required" >&2
    exit 1
  fi
done

if printf '%s\n' "$contents" | grep -Eq 'libcuda\.so|libnvidia-.*\.so'; then
  echo "CUDA archive must not include the host driver library." >&2
  exit 1
fi

marker="$(tar -xOzf "$archive" ./kapsl-provider-cuda12.json)"
printf '%s\n' "$marker" | grep -q '"provider": "cuda"'
printf '%s\n' "$marker" | grep -q 'libonnxruntime_providers_cuda.so'

echo "Merged Linux CUDA package passed."
