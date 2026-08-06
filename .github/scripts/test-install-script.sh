#!/bin/sh
# Exercises install.sh against a local fixture release.
#
# The interesting behavior is all on the failure paths: which runtime actually
# lands, and whether the closing summary describes it honestly. Three separate
# bugs have shipped here — a stale "GPU-accelerated" message over a CPU-only
# install, a missing -cuda12 asset skipping the portable bundle, and a CUDA
# binary that installs cleanly and then cannot load — so each case below
# asserts the binary *and* the message, not just a zero exit.
#
# Runs unmodified on a Linux x86_64 runner, which is the only platform where
# install.sh reaches the CUDA path.
set -eu

version="9.9.9"
test_root="$(mktemp -d)"
release_dir="${test_root}/release"
asset_dir="${release_dir}/runtime/v${version}"
fake_bin_dir="${test_root}/fake-bin"
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

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "This test must run on Linux x86_64; install.sh gates the CUDA path on it." >&2
    exit 1
fi

mkdir -p "${asset_dir}" "${fake_bin_dir}"

# --- fixture payloads -------------------------------------------------------
make_bundle() {
    bundle_asset="$1"
    marker="$2"
    payload="${test_root}/payload-${marker}"
    mkdir -p "${payload}"
    cat > "${payload}/kapsl" <<EOF
#!/bin/sh
echo "${marker}"
EOF
    chmod +x "${payload}/kapsl"
    # Stands in for the ONNX Runtime core libraries the real bundles carry;
    # its presence is how we tell a bundle install from the bare executable.
    echo "sidecar" > "${payload}/libonnxruntime.so.1"
    tar -czf "${asset_dir}/${bundle_asset}" -C "${payload}" .
}

make_bundle "kapsl-${version}-linux-x86_64.tar.gz" "portable"
make_bundle "kapsl-${version}-linux-x86_64-cuda12.tar.gz" "cuda12"

# A CUDA bundle that installs cleanly and then refuses to load, which is what a
# host missing libnccl.so.2 actually looks like.
broken_payload="${test_root}/payload-broken"
mkdir -p "${broken_payload}"
cat > "${broken_payload}/kapsl" <<'EOF'
#!/bin/sh
echo "kapsl: error while loading shared libraries: libnccl.so.2: cannot open shared object file" >&2
exit 127
EOF
chmod +x "${broken_payload}/kapsl"
echo "sidecar" > "${broken_payload}/libonnxruntime.so.1"
tar -czf "${test_root}/cuda12-broken.tar.gz" -C "${broken_payload}" .

for provider in cuda12 tensorrt10; do
    # Distinct from the runtime bundle payloads: a provider pack that also
    # carried a kapsl binary would overwrite the one under test.
    provider_payload="${test_root}/payload-provider-${provider}"
    mkdir -p "${provider_payload}"
    echo '{}' > "${provider_payload}/kapsl-provider-${provider}.json"
    tar -czf "${asset_dir}/kapsl-provider-${provider}-${version}-linux-x86_64.tar.gz" \
        -C "${provider_payload}" .
done

cat > "${asset_dir}/kapsl-${version}-linux-x86_64" <<'EOF'
#!/bin/sh
echo "single-exe"
EOF

cat > "${fake_bin_dir}/nvidia-smi" <<'EOF'
#!/bin/sh
if [ "$1" = "-L" ]; then
    echo "GPU 0: Fixture GPU (UUID: GPU-00000000)"
fi
exit 0
EOF
chmod +x "${fake_bin_dir}/nvidia-smi"

# --- fixture server ---------------------------------------------------------
python3 -m http.server 18081 \
    --bind 127.0.0.1 \
    --directory "${release_dir}" \
    >"${server_log}" 2>&1 &
server_pid="$!"

attempt=1
while ! curl -fsSLI "${base_url}/runtime/v${version}/kapsl-${version}-linux-x86_64.tar.gz" >/dev/null; do
    if [ "${attempt}" -ge 20 ]; then
        cat "${server_log}" >&2
        echo "Timed out waiting for the fixture HTTP server." >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 1
done

# --- harness ----------------------------------------------------------------
# run_install <name> <driver:yes|no> <accelerator|-> ; sets install_dir/log
run_install() {
    case_name="$1"
    with_driver="$2"
    accelerator="$3"

    install_dir="${test_root}/install-${case_name}"
    log_file="${test_root}/log-${case_name}"
    rm -rf "${install_dir}"

    set -- --version "${version}" --base-url "${base_url}" --install-dir "${install_dir}"
    if [ "${accelerator}" != "-" ]; then
        set -- "$@" --accelerator "${accelerator}"
    fi

    if [ "${with_driver}" = "yes" ]; then
        PATH="${fake_bin_dir}:${PATH}" sh install.sh "$@" >"${log_file}" 2>&1 || true
    else
        sh install.sh "$@" >"${log_file}" 2>&1 || true
    fi
}

fail() {
    echo "  FAIL: $1" >&2
    if [ -f "${log_file}" ]; then
        sed 's/^/        /' "${log_file}" >&2
    fi
    failures=$((failures + 1))
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

gpu_claim="GGUF models: GPU-accelerated"
cpu_claim="GGUF models: CPU only"

# --- cases ------------------------------------------------------------------
echo "case: driver present installs the CUDA runtime"
run_install "cuda-driver" yes cuda
expect_binary "cuda12"
expect_file "kapsl-provider-cuda12.json"
expect_log "${gpu_claim}"

echo "case: no driver falls back and says so"
run_install "cuda-nodriver" no cuda
expect_binary "portable"
expect_file "kapsl-provider-cuda12.json"
expect_log "no working NVIDIA driver detected"
expect_log "${cpu_claim}"
reject_log "${gpu_claim}"

echo "case: tensorrt also gets the CUDA runtime and both packs"
run_install "tensorrt-driver" yes tensorrt
expect_binary "cuda12"
expect_file "kapsl-provider-cuda12.json"
expect_file "kapsl-provider-tensorrt10.json"
expect_log "${gpu_claim}"

echo "case: cpu accelerator is untouched by the CUDA path"
run_install "cpu" yes cpu
expect_binary "portable"
reject_log "${gpu_claim}"
reject_log "nvidia"

echo "case: default install is untouched by the CUDA path"
run_install "default" yes -
expect_binary "portable"
reject_log "${gpu_claim}"

# A CUDA binary that cannot load must not be left in place, and must not be
# reported as GPU-accelerated.
echo "case: unusable CUDA runtime falls back to the portable bundle"
cp "${asset_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz" "${test_root}/cuda12-good.tar.gz"
cp "${test_root}/cuda12-broken.tar.gz" "${asset_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz"
run_install "cuda-unusable" yes cuda
expect_binary "portable"
expect_log "libnccl.so.2"
expect_log "${cpu_claim}"
reject_log "${gpu_claim}"
cp "${test_root}/cuda12-good.tar.gz" "${asset_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz"

# Releases older than the -cuda12 artifact must still get the portable bundle,
# with its sidecars, rather than dropping to the bare executable.
echo "case: missing CUDA asset degrades to the portable bundle"
mv "${asset_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz" "${test_root}/held-cuda12.tar.gz"
run_install "cuda-missing" yes cuda
expect_binary "portable"
expect_file "libonnxruntime.so.1"
expect_log "${cpu_claim}"
reject_log "${gpu_claim}"

echo "case: no bundles at all still installs the bare executable"
mv "${asset_dir}/kapsl-${version}-linux-x86_64.tar.gz" "${test_root}/held-portable.tar.gz"
run_install "no-bundles" yes cuda
expect_binary "single-exe"
expect_log "${cpu_claim}"
reject_log "${gpu_claim}"
mv "${test_root}/held-cuda12.tar.gz" "${asset_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz"
mv "${test_root}/held-portable.tar.gz" "${asset_dir}/kapsl-${version}-linux-x86_64.tar.gz"

if [ "${failures}" -ne 0 ]; then
    echo "${failures} install.sh case(s) failed." >&2
    exit 1
fi

echo "All install.sh cases passed."
