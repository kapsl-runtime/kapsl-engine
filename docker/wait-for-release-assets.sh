#!/bin/sh
set -eu

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"

release_base="${KAPSL_RELEASE_BASE_URL:-https://github.com/kapsl-runtime/kapsl-engine/releases/download/v${KAPSL_VERSION}}"
# Allow the existing two-hour budget, polling no more often than every 30 seconds.
max_attempts="${KAPSL_WAIT_ATTEMPTS:-240}"
retry_delay="${KAPSL_WAIT_DELAY_SECONDS:-30}"

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

if [ -n "${KAPSL_INSTALLER_RUN_SHA:-}" ]; then
    : "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required to check the installer run}"
fi

assets="
kapsl-${KAPSL_VERSION}-linux-x86_64.tar.gz
kapsl-${KAPSL_VERSION}-linux-aarch64.tar.gz
kapsl-${KAPSL_VERSION}-linux-x86_64-cuda12.tar.gz
kapsl-provider-cuda12-${KAPSL_VERSION}-linux-x86_64.tar.gz
kapsl-provider-tensorrt10-${KAPSL_VERSION}-linux-x86_64.tar.gz
"

attempt=1
while [ "${attempt}" -le "${max_attempts}" ]; do
    installer_pending=false
    if [ -n "${KAPSL_INSTALLER_RUN_SHA:-}" ]; then
        # Match both the exact tag and commit, never an unrelated branch run.
        # Pick the newest matching run so a later attempt supersedes old failures.
        if ! installer_run="$(gh api --method GET \
            "repos/${GITHUB_REPOSITORY}/actions/workflows/release-runtime-installers.yml/runs" \
            -f event=push -f head_sha="${KAPSL_INSTALLER_RUN_SHA}" -f per_page=100 \
            --jq '.workflow_runs
                | map(select(.head_branch == ("v" + env.KAPSL_VERSION)
                    and .head_sha == env.KAPSL_INSTALLER_RUN_SHA))
                | sort_by(.id) | last
                | if . == null then "not_started"
                  else [.status, (.conclusion // "pending"), .html_url] | join(" ")
                  end')"; then
            echo "Unable to check the installer workflow; refusing to publish Docker images." >&2
            exit 1
        fi
        case "${installer_run}" in
            'completed success '*) ;;
            completed\ *)
                echo "Installer workflow did not succeed: ${installer_run}" >&2
                echo "Stopping: release assets will not be produced by this run." >&2
                exit 1
                ;;
            *)
                installer_pending=true
                echo "Waiting for installer workflow (attempt ${attempt}/${max_attempts}): ${installer_run}"
                ;;
        esac
    fi

    missing_assets=""

    if [ "${installer_pending}" = false ]; then
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
    fi
    if [ "${attempt}" -lt "${max_attempts}" ]; then
        sleep "${retry_delay}"
    fi
    attempt=$((attempt + 1))
done

echo "Timed out waiting for complete release assets for ${KAPSL_VERSION}." >&2
exit 1
