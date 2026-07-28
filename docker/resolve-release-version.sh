#!/bin/sh
set -eu

: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"

cargo_version="$(grep -m1 '^version = ' kapsl-runtime/crates/kapsl-cli/Cargo.toml | cut -d '"' -f2)"
ref_type="${GITHUB_REF_TYPE:-}"
ref_name="${GITHUB_REF_NAME:-}"
dispatch_release_version="${DISPATCH_RELEASE_VERSION:-}"
dispatch_update_channel_tags="${DISPATCH_UPDATE_CHANNEL_TAGS:-false}"

if [ "${ref_type}" = "tag" ]; then
    release_version="${ref_name#v}"
    update_channel_tags=true
elif [ -n "${dispatch_release_version}" ]; then
    release_version="${dispatch_release_version#v}"
    update_channel_tags="${dispatch_update_channel_tags}"
else
    : "${GITHUB_SHA:?GITHUB_SHA is required for development images}"
    short_sha="$(printf '%.8s' "${GITHUB_SHA}")"
    echo "version=dev-${cargo_version}-${short_sha}" >> "${GITHUB_OUTPUT}"
    echo "runtime_version=${cargo_version}" >> "${GITHUB_OUTPUT}"
    echo "channel=development" >> "${GITHUB_OUTPUT}"
    echo "update_channel_tags=false" >> "${GITHUB_OUTPUT}"
    echo "wait_for_assets=false" >> "${GITHUB_OUTPUT}"
    exit 0
fi

case "${release_version}" in
    [0-9]*) ;;
    *)
        echo "Release version must start with a number: ${release_version}" >&2
        exit 1
        ;;
esac
case "${release_version}" in
    *[!0-9A-Za-z._-]*)
        echo "Release version contains unsupported characters: ${release_version}" >&2
        exit 1
        ;;
esac

case "${update_channel_tags}" in
    true | false) ;;
    *)
        echo "Channel tag option must be true or false: ${update_channel_tags}" >&2
        exit 1
        ;;
esac

channel=stable
case "${release_version}" in
    *-beta*) channel=beta ;;
    *-*) channel=prerelease ;;
esac

echo "version=${release_version}" >> "${GITHUB_OUTPUT}"
echo "runtime_version=${release_version}" >> "${GITHUB_OUTPUT}"
echo "channel=${channel}" >> "${GITHUB_OUTPUT}"
echo "update_channel_tags=${update_channel_tags}" >> "${GITHUB_OUTPUT}"
echo "wait_for_assets=true" >> "${GITHUB_OUTPUT}"
