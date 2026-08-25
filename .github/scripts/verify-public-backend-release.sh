#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: verify-public-backend-release.sh LOCAL_INDEX PUBLIC_RELEASE_ROOT" >&2
  exit 2
fi

local_index="$1"
public_root="${2%/}"
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

for name in backend-index.json backend-index.json.sig; do
  curl --fail --silent --show-error --location \
    --retry 8 --retry-delay 5 --retry-all-errors \
    --output "$scratch/$name" "$public_root/$name"
done
cmp "$local_index" "$scratch/backend-index.json"
cmp "$local_signature" "$scratch/backend-index.json.sig"

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
  curl --fail --silent --show-error --location --head \
    --retry 8 --retry-delay 5 --retry-all-errors \
    "$artifact" >/dev/null
done < "$scratch/artifact-urls"

echo "Verified signed backend release through $public_root"
