#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ] || { [ "$#" -eq 3 ] && [ "$3" != "--artifacts-only" ]; }; then
  echo "usage: verify-public-backend-release.sh LOCAL_INDEX PUBLIC_RELEASE_ROOT [--artifacts-only]" >&2
  exit 2
fi

local_index="$1"
public_root="${2%/}"
mode="${3:-}"
local_signature="${local_index}.sig"
if [ ! -f "$local_index" ] || [ ! -f "$local_signature" ]; then
  echo "Local backend index and signature are required." >&2
  exit 1
fi

scratch="$(mktemp -d)"
cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT INT TERM

wait_for_public_object() {
  local url="$1"
  shift
  local deadline=$((SECONDS + 600))
  local http_status curl_exit remaining request_timeout
  echo "Waiting for public release object: $url" >&2
  while true; do
    remaining=$((deadline - SECONDS))
    if [ "$remaining" -le 0 ]; then
      echo "Timed out waiting for public release object: $url" >&2
      return 1
    fi
    request_timeout=30
    if [ "$remaining" -lt "$request_timeout" ]; then
      request_timeout="$remaining"
    fi
    : > "$scratch/http-headers"
    if http_status="$(curl --fail --silent --show-error --location \
      --connect-timeout 15 --max-time "$request_timeout" \
      --dump-header "$scratch/http-headers" --write-out '%{http_code}' \
      "$@" "$url")"; then
      curl_exit=0
    else
      curl_exit=$?
    fi

    if grep -Eiq '^cf-mitigated:[[:space:]]*challenge[[:space:]]*$' "$scratch/http-headers"; then
      echo "Cloudflare browser challenge blocked public release object: $url" >&2
      grep -Ei '^(HTTP/|cf-ray:|cf-mitigated:|server:|content-type:)' "$scratch/http-headers" >&2 || true
      echo "Check the matching Ray ID in Cloudflare Security Events and exempt intended public downloads from the triggering challenge rule. Automated release clients cannot complete browser challenges." >&2
      return 1
    fi
    if [ "$curl_exit" -eq 0 ] && [ "$http_status" = 200 ]; then
      return 0
    fi

    echo "Public release object failed: $url (HTTP $http_status, curl exit $curl_exit)" >&2
    grep -Ei '^(HTTP/|cf-ray:|cf-mitigated:|server:|content-type:)' "$scratch/http-headers" >&2 || true
    case "$http_status" in
      000|404|408|429|5??) ;;
      *) return 1 ;;
    esac
    if [ "$((deadline - SECONDS))" -le 30 ]; then
      echo "Public release object did not become available within the retry budget: $url" >&2
      return 1
    fi
    sleep 30
  done
}

if [ "$mode" != "--artifacts-only" ]; then
  for name in backend-index.json backend-index.json.sig; do
    wait_for_public_object \
      "$public_root/$name" \
      --output "$scratch/$name"
  done
  cmp "$local_index" "$scratch/backend-index.json"
  cmp "$local_signature" "$scratch/backend-index.json.sig"
fi

python3 - "$local_index" "$public_root" <<'PY' > "$scratch/artifact-urls"
import json
import pathlib
import posixpath
import sys
from urllib.parse import unquote, urlsplit

index = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
root = urlsplit(sys.argv[2].rstrip("/") + "/")
for pack in index.get("packs", []):
    artifact = pack.get("artifact", "")
    parsed = urlsplit(artifact)
    normalized = posixpath.normpath(unquote(parsed.path))
    release_prefix = root.path.rstrip("/") + "/"
    if (
        parsed.scheme != root.scheme
        or parsed.netloc != root.netloc
        or parsed.query
        or parsed.fragment
        or not normalized.startswith(release_prefix)
        or normalized == release_prefix.rstrip("/")
    ):
        raise SystemExit(f"backend artifact escapes public release root: {artifact}")
    print(artifact)
PY

if [ ! -s "$scratch/artifact-urls" ]; then
  echo "Published backend index contains no artifacts." >&2
  exit 1
fi
while IFS= read -r artifact; do
  wait_for_public_object "$artifact" --head --output /dev/null
done < "$scratch/artifact-urls"

if [ "$mode" = "--artifacts-only" ]; then
  echo "Verified backend artifacts through $public_root"
else
  echo "Verified signed backend release through $public_root"
fi
