#!/bin/sh
set -eu

python3 docker/test-wait-for-release-assets.py

version="9.9.9"
test_root="$(mktemp -d)"
release_dir="${test_root}/release"
portable_payload_dir="${test_root}/payload-portable"
cuda_payload_dir="${test_root}/payload-cuda"
cuda_provider_payload_dir="${test_root}/payload-provider-cuda"
tensorrt_provider_payload_dir="${test_root}/payload-provider-tensorrt"
server_log="${test_root}/http-server.log"
server_pid=""
base_url="http://127.0.0.1:18080"

cleanup() {
    if [ -n "${server_pid}" ]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    rm -rf "${test_root}"
}
trap cleanup EXIT INT TERM

mkdir -p \
    "${release_dir}" \
    "${portable_payload_dir}" \
    "${cuda_payload_dir}" \
    "${cuda_provider_payload_dir}" \
    "${tensorrt_provider_payload_dir}"
cat > "${portable_payload_dir}/kapsl" <<'EOF'
#!/bin/sh
echo "kapsl portable release-script test"
EOF
chmod +x "${portable_payload_dir}/kapsl"
cat > "${cuda_payload_dir}/kapsl" <<'EOF'
#!/bin/sh
echo "kapsl cuda release-script test"
EOF
chmod +x "${cuda_payload_dir}/kapsl"
echo '{}' > "${cuda_payload_dir}/kapsl-provider-cuda12.json"
echo 'cuda provider' > "${cuda_payload_dir}/libonnxruntime_providers_cuda.so"
echo '{}' > "${cuda_provider_payload_dir}/kapsl-provider-cuda12.json"
echo '{}' > "${tensorrt_provider_payload_dir}/kapsl-provider-tensorrt10.json"

assert_output() {
    expected_line="$1"
    output_file="$2"
    if ! grep -qx "${expected_line}" "${output_file}"; then
        echo "Missing expected resolver output '${expected_line}' in ${output_file}:" >&2
        cat "${output_file}" >&2
        exit 1
    fi
}

tag_output="${test_root}/tag-output"
GITHUB_OUTPUT="${tag_output}" \
GITHUB_REF_TYPE=tag \
GITHUB_REF_NAME=v1.2.3 \
GITHUB_SHA=1234567890abcdef \
    docker/resolve-release-version.sh
assert_output "version=1.2.3" "${tag_output}"
assert_output "channel=stable" "${tag_output}"
assert_output "update_channel_tags=true" "${tag_output}"
assert_output "wait_for_assets=true" "${tag_output}"

dispatch_output="${test_root}/dispatch-output"
GITHUB_OUTPUT="${dispatch_output}" \
GITHUB_REF_TYPE=branch \
GITHUB_REF_NAME=main \
GITHUB_SHA=1234567890abcdef \
DISPATCH_RELEASE_VERSION=v1.2.3-beta.1 \
DISPATCH_UPDATE_CHANNEL_TAGS=false \
    docker/resolve-release-version.sh
assert_output "version=1.2.3-beta.1" "${dispatch_output}"
assert_output "channel=beta" "${dispatch_output}"
assert_output "update_channel_tags=false" "${dispatch_output}"
assert_output "wait_for_assets=true" "${dispatch_output}"

development_output="${test_root}/development-output"
cargo_version="$(grep -m1 '^version = ' kapsl-runtime/crates/kapsl-cli/Cargo.toml | cut -d '"' -f2)"
GITHUB_OUTPUT="${development_output}" \
GITHUB_REF_TYPE=branch \
GITHUB_REF_NAME=main \
GITHUB_SHA=1234567890abcdef \
    docker/resolve-release-version.sh
assert_output "version=dev-${cargo_version}-12345678" "${development_output}"
assert_output "runtime_version=${cargo_version}" "${development_output}"
assert_output "channel=development" "${development_output}"
assert_output "wait_for_assets=false" "${development_output}"

if GITHUB_OUTPUT="${test_root}/invalid-output" \
    GITHUB_REF_TYPE=branch \
    GITHUB_REF_NAME=main \
    GITHUB_SHA=1234567890abcdef \
    DISPATCH_RELEASE_VERSION='1.2.3;invalid' \
    docker/resolve-release-version.sh; then
    echo "Version resolver unexpectedly accepted an unsafe release version." >&2
    exit 1
fi

create_asset() {
    asset="$1"
    source_dir="$2"
    tar -czf "${release_dir}/${asset}" -C "${source_dir}" .
    sha256sum "${release_dir}/${asset}" > "${release_dir}/${asset}.sha256"
}

create_asset "kapsl-${version}-linux-x86_64.tar.gz" "${portable_payload_dir}"
create_asset "kapsl-${version}-linux-aarch64.tar.gz" "${portable_payload_dir}"
create_asset "kapsl-${version}-linux-x86_64-cuda12.tar.gz" "${cuda_payload_dir}"
create_asset "kapsl-provider-cuda12-${version}-linux-x86_64.tar.gz" "${cuda_provider_payload_dir}"
create_asset "kapsl-provider-tensorrt10-${version}-linux-x86_64.tar.gz" "${tensorrt_provider_payload_dir}"

python3 -m http.server 18080 \
    --bind 127.0.0.1 \
    --directory "${release_dir}" \
    >"${server_log}" 2>&1 &
server_pid="$!"

attempt=1
while ! curl -fsSLI "${base_url}/kapsl-${version}-linux-x86_64.tar.gz" >/dev/null; do
    if [ "${attempt}" -ge 20 ]; then
        cat "${server_log}" >&2
        echo "Timed out waiting for test HTTP server." >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 1
done

KAPSL_VERSION="${version}" \
KAPSL_RELEASE_BASE_URL="${base_url}" \
KAPSL_WAIT_ATTEMPTS=1 \
KAPSL_WAIT_DELAY_SECONDS=0 \
    docker/wait-for-release-assets.sh

rm "${release_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz.sha256"
if KAPSL_VERSION="${version}" \
    KAPSL_RELEASE_BASE_URL="${base_url}" \
    KAPSL_WAIT_ATTEMPTS=1 \
    KAPSL_WAIT_DELAY_SECONDS=0 \
    docker/wait-for-release-assets.sh; then
    echo "Asset readiness check unexpectedly accepted an incomplete release." >&2
    exit 1
fi

sha256sum \
    "${release_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz" \
    >"${release_dir}/kapsl-${version}-linux-x86_64-cuda12.tar.gz.sha256"

# The normal CUDA/Docker path is one archive. Remove the legacy standalone
# provider pack after the readiness test to prove neither CUDA nor TensorRT
# installation downloads it.
rm \
    "${release_dir}/kapsl-provider-cuda12-${version}-linux-x86_64.tar.gz" \
    "${release_dir}/kapsl-provider-cuda12-${version}-linux-x86_64.tar.gz.sha256"

install_dir="${test_root}/install"
KAPSL_VERSION="${version}" \
KAPSL_RELEASE_BASE_URL="${base_url}" \
KAPSL_INSTALL_DIR="${install_dir}" \
    sh docker/install-release-assets.sh cpu
if [ "$("${install_dir}/kapsl")" != "kapsl portable release-script test" ]; then
    echo "CPU installer did not select the portable runtime asset." >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64 | amd64)
        cuda_install_dir="${test_root}/install-cuda"
        KAPSL_VERSION="${version}" \
        KAPSL_RELEASE_BASE_URL="${base_url}" \
        KAPSL_INSTALL_DIR="${cuda_install_dir}" \
            sh docker/install-release-assets.sh cuda
        if [ "$("${cuda_install_dir}/kapsl")" != "kapsl cuda release-script test" ]; then
            echo "CUDA installer did not select the CUDA-enabled runtime asset." >&2
            exit 1
        fi
        test -f "${cuda_install_dir}/kapsl-provider-cuda12.json"
        test -f "${cuda_install_dir}/libonnxruntime_providers_cuda.so"

        tensorrt_install_dir="${test_root}/install-tensorrt"
        KAPSL_VERSION="${version}" \
        KAPSL_RELEASE_BASE_URL="${base_url}" \
        KAPSL_INSTALL_DIR="${tensorrt_install_dir}" \
            sh docker/install-release-assets.sh tensorrt
        if [ "$("${tensorrt_install_dir}/kapsl")" != "kapsl cuda release-script test" ]; then
            echo "TensorRT installer did not select the CUDA-enabled runtime asset." >&2
            exit 1
        fi
        test -f "${tensorrt_install_dir}/kapsl-provider-cuda12.json"
        test -f "${tensorrt_install_dir}/libonnxruntime_providers_cuda.so"
        test -f "${tensorrt_install_dir}/kapsl-provider-tensorrt10.json"
        ;;
esac

case "$(uname -m)" in
    x86_64 | amd64) runtime_platform="linux-x86_64" ;;
    aarch64 | arm64) runtime_platform="linux-aarch64" ;;
    *)
        echo "Unsupported test architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

printf 'corrupt' >> "${release_dir}/kapsl-${version}-${runtime_platform}.tar.gz"
if KAPSL_VERSION="${version}" \
    KAPSL_RELEASE_BASE_URL="${base_url}" \
    KAPSL_INSTALL_DIR="${test_root}/corrupt-install" \
    sh docker/install-release-assets.sh cpu; then
    echo "Installer unexpectedly accepted a checksum mismatch." >&2
    exit 1
fi

echo "Release asset scripts passed."
