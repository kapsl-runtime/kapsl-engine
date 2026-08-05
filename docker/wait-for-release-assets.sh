#!/bin/sh
set -eu

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"

release_base="${KAPSL_RELEASE_BASE_URL:-https://github.com/kapsl-runtime/kapsl-engine/releases/download/v${KAPSL_VERSION}}"
max_attempts="${KAPSL_WAIT_ATTEMPTS:-180}"
retry_delay="${KAPSL_WAIT_DELAY_SECONDS:-10}"

case "${max_attempts}" in
    *[!0-9]* | 0 | "")
        echo "KAPSL_WAIT_ATTEMPTS must be a positive integer." >&2
        exit 1
        ;;
esac

case "${retry_delay}" in
    *[!0-9]* | "")
        echo "KAPSL_WAIT_DELAY_SECONDS must be a non-negative integer." >&2
        exit 1
        ;;
esac

assets="
kapsl-${KAPSL_VERSION}-linux-x86_64.tar.gz
kapsl-${KAPSL_VERSION}-linux-aarch64.tar.gz
kapsl-${KAPSL_VERSION}-linux-x86_64-cuda12.tar.gz
kapsl-provider-cuda12-${KAPSL_VERSION}-linux-x86_64.tar.gz
kapsl-provider-tensorrt10-${KAPSL_VERSION}-linux-x86_64.tar.gz
"

attempt=1
while [ "${attempt}" -le "${max_attempts}" ]; do
    missing_assets=""

    for asset in ${assets}; do
        for required_file in "${asset}" "${asset}.sha256"; do
            if ! curl -fsSLI --connect-timeout 30 "${release_base}/${required_file}" >/dev/null; then
                missing_assets="${missing_assets} ${required_file}"
            fi
        done
    done

    if [ -z "${missing_assets}" ]; then
        echo "All runtime and provider release assets are available for ${KAPSL_VERSION}."
        exit 0
    fi

    echo "Release assets are not complete (attempt ${attempt}/${max_attempts}):${missing_assets}"
    if [ "${attempt}" -lt "${max_attempts}" ]; then
        sleep "${retry_delay}"
    fi
    attempt=$((attempt + 1))
done

echo "Timed out waiting for complete release assets for ${KAPSL_VERSION}." >&2
exit 1
