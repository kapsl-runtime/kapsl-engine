#!/bin/sh
# Exercise the public installers against a local fixture release.
set -eu

version="9.9.9"
script_dir="$(CDPATH= cd "$(dirname "$0")" && pwd)"
repository_root="$(CDPATH= cd "${script_dir}/../.." && pwd)"
installer_dir="${repository_root}/installers"
test_root="$(mktemp -d)"
release_dir="${test_root}/release"
asset_dir="${release_dir}/runtime/v${version}"
beta_asset_dir="${release_dir}/runtime/beta/v${version}"
fake_bin="${test_root}/bin"
server_log="${test_root}/http-server.log"
server_pid=""
base_url="http://127.0.0.1:18081"
failures=0

cleanup() {
    if [ -n "${server_pid}" ]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    rm -rf "${test_root}"
}
trap cleanup EXIT INT TERM

mkdir -p "${asset_dir}" "${beta_asset_dir}" "${fake_bin}"

# Keep the test runnable on developer Macs while exercising the production
# Linux x86_64 selection path byte-for-byte.
cat > "${fake_bin}/uname" <<'EOF'
#!/bin/sh
case "${1:-}" in
    -s) echo Linux ;;
    -m) echo x86_64 ;;
    *) echo Linux ;;
esac
EOF
chmod +x "${fake_bin}/uname"
cat > "${fake_bin}/nvidia-smi" <<'EOF'
#!/bin/sh
exit 1
EOF
chmod +x "${fake_bin}/nvidia-smi"

# --- fixture payloads -------------------------------------------------------
make_bundle() {
    bundle_asset="$1"
    marker="$2"
    accelerator="$3"
    payload="${test_root}/payload-${marker}"
    rm -rf "${payload}"
    mkdir -p "${payload}"
    cat > "${payload}/kapsl" <<EOF
#!/bin/sh
echo "${marker}"
EOF
    chmod +x "${payload}/kapsl"
    echo "core sidecar" > "${payload}/libonnxruntime.so.1"
    if [ "${accelerator}" = "cuda" ]; then
        echo '# vllm-v1-packed-cuda-ipc/flash-attn' >> "${payload}/kapsl"
        echo '{}' > "${payload}/kapsl-provider-cuda12.json"
        echo "cuda provider" > "${payload}/libonnxruntime_providers_cuda.so"
        echo "cudnn" > "${payload}/libcudnn.so.9"
    fi
    tar -czf "${asset_dir}/${bundle_asset}" -C "${payload}" .
}

portable_asset="kapsl-${version}-linux-x86_64.tar.gz"
cuda_asset="kapsl-${version}-linux-x86_64-cuda12.tar.gz"
make_bundle "${portable_asset}" "portable" cpu
make_bundle "${cuda_asset}" "cuda12" cuda

make_vllm_pack() {
    destination="$1"
    backend_asset="kapsl-backend-vllm-${version}-linux-x86_64.tar.gz"
    backend_payload="${test_root}/payload-vllm"
    rm -rf "${backend_payload}"
    mkdir -p "${backend_payload}/backends/vllm-bootstrap"
    cat > "${backend_payload}/backends/vllm-bootstrap/bootstrap.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
bootstrap_root="$1"
target_root="$2"
mkdir -p "$target_root/bin"
cp "$bootstrap_root/fake-python" "$target_root/bin/python"
chmod 755 "$target_root/bin/python"
printf 'fixture managed backend\n' > "$target_root/kapsl-vllm-backend.json"
EOF
    cat > "${backend_payload}/backends/vllm-bootstrap/fake-python" <<'EOF'
#!/bin/sh
exit 0
EOF
    chmod 755 \
        "${backend_payload}/backends/vllm-bootstrap/bootstrap.sh" \
        "${backend_payload}/backends/vllm-bootstrap/fake-python"
    tar -czf "${destination}/${backend_asset}" -C "${backend_payload}" .
    (cd "${destination}" && sha256sum "${backend_asset}" > "${backend_asset}.sha256")
}

make_vllm_pack "${asset_dir}"

for provider in cuda12 tensorrt10; do
    provider_payload="${test_root}/payload-provider-${provider}"
    mkdir -p "${provider_payload}"
    echo '{}' > "${provider_payload}/kapsl-provider-${provider}.json"
    if [ "${provider}" = "cuda12" ]; then
        echo "legacy cuda provider" > "${provider_payload}/libonnxruntime_providers_cuda.so"
    fi
    tar -czf "${asset_dir}/kapsl-provider-${provider}-${version}-linux-x86_64.tar.gz" \
        -C "${provider_payload}" .
done

cat > "${asset_dir}/kapsl-${version}-linux-x86_64" <<'EOF'
#!/bin/sh
echo "single-exe"
EOF

# install-cuda.sh fetches the general installer from the same origin.
cp "${installer_dir}/install.sh" "${release_dir}/install.sh"
cp "${installer_dir}/install.sh" "${release_dir}/install-beta-base.sh"
cp "${asset_dir}/${portable_asset}" "${beta_asset_dir}/${portable_asset}"
cp "${asset_dir}/${cuda_asset}" "${beta_asset_dir}/${cuda_asset}"
cp "${asset_dir}/kapsl-backend-vllm-${version}-linux-x86_64.tar.gz" "${beta_asset_dir}/"
cp "${asset_dir}/kapsl-backend-vllm-${version}-linux-x86_64.tar.gz.sha256" "${beta_asset_dir}/"

# --- fixture server ---------------------------------------------------------
python3 -m http.server 18081 \
    --bind 127.0.0.1 \
    --directory "${release_dir}" \
    >"${server_log}" 2>&1 &
server_pid="$!"

attempt=1
while ! curl -fsSLI "${base_url}/runtime/v${version}/${portable_asset}" >/dev/null; do
    if [ "${attempt}" -ge 20 ]; then
        cat "${server_log}" >&2
        echo "Timed out waiting for the fixture HTTP server." >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 1
done

# --- harness ----------------------------------------------------------------
# run_install <name> <accelerator|->; sets install_dir/log_file/install_status.
run_install() {
    case_name="$1"
    accelerator="$2"
    prefetch="${3:-}"
    install_dir="${test_root}/install-${case_name}"
    log_file="${test_root}/log-${case_name}"
    rm -rf "${install_dir}"

    set -- --version "${version}" --base-url "${base_url}" --install-dir "${install_dir}"
    if [ "${accelerator}" != "-" ]; then
        set -- "$@" --accelerator "${accelerator}"
    fi
    if [ -n "${prefetch}" ]; then
        set -- "$@" --prefetch-backends "${prefetch}"
    fi

    set +e
    PATH="${fake_bin}:${PATH}" sh "${installer_dir}/install.sh" "$@" >"${log_file}" 2>&1
    install_status=$?
    set -e
}

run_cuda_wrapper() {
    case_name="$1"
    wrapper="$2"
    prefetch="${3:-}"
    install_dir="${test_root}/install-${case_name}"
    log_file="${test_root}/log-${case_name}"
    rm -rf "${install_dir}"

    set +e
    KAPSL_BASE_URL="${base_url}" \
    KAPSL_VERSION="${version}" \
    KAPSL_INSTALL_DIR="${install_dir}" \
    KAPSL_PREFETCH_BACKENDS="${prefetch}" \
    PATH="${fake_bin}:${PATH}" \
        sh "${wrapper}" >"${log_file}" 2>&1
    install_status=$?
    set -e
}

fail() {
    echo "  FAIL: $1" >&2
    if [ -f "${log_file}" ]; then
        sed 's/^/        /' "${log_file}" >&2
    fi
    failures=$((failures + 1))
}

expect_status() {
    expected="$1"
    if [ "${install_status}" -ne "${expected}" ]; then
        fail "installer exited ${install_status}; expected ${expected}"
    fi
}

expect_binary() {
    expected="$1"
    if [ ! -x "${install_dir}/kapsl" ]; then
        fail "no kapsl installed; expected '${expected}'"
        return
    fi
    actual="$("${install_dir}/kapsl" 2>/dev/null || echo "<unrunnable>")"
    if [ "${actual}" != "${expected}" ]; then
        fail "installed '${actual}', expected '${expected}'"
    fi
}

expect_log() {
    if ! grep -qF "$1" "${log_file}"; then
        fail "expected output containing '$1'"
    fi
}

reject_log() {
    if grep -qF "$1" "${log_file}"; then
        fail "output must not contain '$1'"
    fi
}

expect_file() {
    if [ ! -f "${install_dir}/$1" ]; then
        fail "expected '$1' to be installed"
    fi
}

# --- cases ------------------------------------------------------------------
echo "case: CUDA installs one merged GGUF + ONNX bundle"
run_install "cuda" cuda
expect_status 0
expect_binary "cuda12"
expect_file "kapsl-provider-cuda12.json"
expect_file "libonnxruntime_providers_cuda.so"
expect_file "libcudnn.so.9"
if [ -e "${install_dir}/backends/vllm/bin/python" ]; then
    fail "default CUDA install eagerly installed managed vLLM"
fi
expect_log "GGUF models: CUDA compiled"
expect_log "ONNX models: CUDA execution provider installed"
expect_log "installed on first eligible run"
reject_log "legacy split"

echo "case: direct CUDA wrapper selects the same merged bundle"
run_cuda_wrapper "cuda-wrapper" "${installer_dir}/install-cuda.sh"
expect_status 0
expect_binary "cuda12"
expect_file "kapsl-provider-cuda12.json"
expect_file "libonnxruntime_providers_cuda.so"
if [ -e "${install_dir}/backends/vllm/bin/python" ]; then
    fail "direct CUDA wrapper eagerly installed managed vLLM"
fi

echo "case: beta CUDA wrapper selects the merged beta bundle"
run_cuda_wrapper "beta-cuda-wrapper" "${installer_dir}/install-beta-cuda.sh"
expect_status 0
expect_binary "cuda12"
expect_file "kapsl-provider-cuda12.json"
expect_file "libonnxruntime_providers_cuda.so"
if [ -e "${install_dir}/backends/vllm/bin/python" ]; then
    fail "beta CUDA wrapper eagerly installed managed vLLM"
fi

echo "case: generic beta wrapper selects portable runtime without an NVIDIA driver"
run_cuda_wrapper "beta-wrapper-cpu" "${installer_dir}/install-beta.sh"
expect_status 0
expect_binary "portable"

echo "case: generic beta wrapper detects a working NVIDIA driver"
cat > "${fake_bin}/nvidia-smi" <<'EOF'
#!/bin/sh
case "${1:-}" in
    -L) echo 'GPU 0: fixture' ;;
esac
exit 0
EOF
chmod +x "${fake_bin}/nvidia-smi"
run_cuda_wrapper "beta-wrapper-cuda" "${installer_dir}/install-beta.sh"
expect_status 0
expect_binary "cuda12"
expect_file "kapsl-provider-cuda12.json"

echo "case: managed-vLLM eager compatibility prefetch remains explicit"
run_install "cuda-prefetch" cuda vllm
expect_status 0
expect_file "backends/vllm/bin/python"
expect_file "backends/vllm/kapsl-vllm-backend.json"
expect_log "managed vLLM backend installed"

echo "case: TensorRT adds only its provider to the merged CUDA bundle"
run_install "tensorrt" tensorrt
expect_status 0
expect_binary "cuda12"
expect_file "kapsl-provider-cuda12.json"
expect_file "libonnxruntime_providers_cuda.so"
expect_file "kapsl-provider-tensorrt10.json"
if [ -e "${install_dir}/backends/vllm/bin/python" ]; then
    fail "TensorRT install eagerly installed managed vLLM"
fi
reject_log "legacy split"

echo "case: CPU and the default install remain portable"
run_install "cpu" cpu
expect_status 0
expect_binary "portable"
reject_log "CUDA compiled"
run_install "default" -
expect_status 0
expect_binary "portable"
reject_log "CUDA compiled"

echo "case: an older split CUDA release remains installable"
cp "${asset_dir}/${cuda_asset}" "${test_root}/merged-cuda.tar.gz"
make_bundle "${cuda_asset}" "legacy-cuda12" cpu
run_install "legacy-cuda" cuda
expect_status 0
expect_binary "legacy-cuda12"
expect_file "kapsl-provider-cuda12.json"
expect_file "libonnxruntime_providers_cuda.so"
expect_log "legacy split ONNX CUDA provider pack"
cp "${test_root}/merged-cuda.tar.gz" "${asset_dir}/${cuda_asset}"

echo "case: missing managed-vLLM pack fails closed"
backend_asset="kapsl-backend-vllm-${version}-linux-x86_64.tar.gz"
mv "${asset_dir}/${backend_asset}" "${test_root}/held-vllm.tar.gz"
run_install "vllm-missing" cuda vllm
if [ "${install_status}" -eq 0 ]; then
    fail "missing managed-vLLM pack unexpectedly succeeded"
fi
if [ -e "${install_dir}/backends/vllm/bin/python" ]; then
    fail "missing managed-vLLM pack installed a backend"
fi
mv "${test_root}/held-vllm.tar.gz" "${asset_dir}/${backend_asset}"

echo "case: corrupt managed-vLLM pack fails checksum validation"
cp "${asset_dir}/${backend_asset}" "${test_root}/valid-vllm.tar.gz"
printf 'corrupt' > "${asset_dir}/${backend_asset}"
run_install "vllm-corrupt" cuda vllm
if [ "${install_status}" -eq 0 ]; then
    fail "corrupt managed-vLLM pack unexpectedly succeeded"
fi
expect_log "Managed-vLLM backend checksum mismatch"
cp "${test_root}/valid-vllm.tar.gz" "${asset_dir}/${backend_asset}"

echo "case: missing CUDA bundle fails instead of silently installing CPU"
mv "${asset_dir}/${cuda_asset}" "${test_root}/held-cuda.tar.gz"
run_install "cuda-missing" cuda
if [ "${install_status}" -eq 0 ]; then
    fail "missing CUDA bundle unexpectedly succeeded"
fi
expect_log "no CPU runtime was substituted"
if [ -e "${install_dir}/kapsl" ]; then
    fail "missing CUDA bundle installed a runtime"
fi
mv "${test_root}/held-cuda.tar.gz" "${asset_dir}/${cuda_asset}"

echo "case: corrupt CUDA bundle fails instead of silently installing CPU"
cp "${asset_dir}/${cuda_asset}" "${test_root}/valid-cuda.tar.gz"
printf 'corrupt' > "${asset_dir}/${cuda_asset}"
run_install "cuda-corrupt" cuda
if [ "${install_status}" -eq 0 ]; then
    fail "corrupt CUDA bundle unexpectedly succeeded"
fi
expect_log "no CPU runtime was substituted"
cp "${test_root}/valid-cuda.tar.gz" "${asset_dir}/${cuda_asset}"

echo "case: old CPU release without a bundle uses the bare executable"
mv "${asset_dir}/${portable_asset}" "${test_root}/held-portable.tar.gz"
run_install "cpu-no-bundle" cpu
expect_status 0
expect_binary "single-exe"
mv "${test_root}/held-portable.tar.gz" "${asset_dir}/${portable_asset}"

if [ "${failures}" -ne 0 ]; then
    echo "${failures} install.sh case(s) failed." >&2
    exit 1
fi

echo "All install.sh cases passed."
