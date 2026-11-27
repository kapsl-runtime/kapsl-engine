#!/usr/bin/env bash
# Embedded llama.cpp versus signed llama-cpp/cpu pack correctness smoke.
#
# The default portable Linux build serves CPU GGUF models through the signed
# llama-cpp/cpu pack. This script builds that default engine once, installs the
# locally packaged and test-signed pack from an offline bundle, and checks that
# greedy chat completions through the pack match the embedded llama.cpp route
# (KAPSL_LAZY_LLAMA_CPP_PACKS=0) exactly. Both routes construct the same
# kapsl_llm::GgufBackend, so the comparison isolates the pack boundary.
set -Eeuo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"
: "${KAPSL_LLAMA_PARITY_MODEL:?KAPSL_LLAMA_PARITY_MODEL is required}"
: "${KAPSL_LLAMA_PARITY_MODEL_SHA256:?KAPSL_LLAMA_PARITY_MODEL_SHA256 is required}"
: "${KAPSL_LLAMA_PARITY_PUBLIC_KEY:?KAPSL_LLAMA_PARITY_PUBLIC_KEY is required}"

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "llama.cpp CPU parity runs only on Linux x86_64, where the pack is published." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
script_dir="$repo_root/.github/scripts"
parity_helper="$script_dir/llama_cpp_cpu_parity.py"
artifacts_dir="${KAPSL_LLAMA_PARITY_ARTIFACTS_DIR:-$repo_root/dist}"
evidence_dir="${KAPSL_LLAMA_PARITY_EVIDENCE_DIR:-$artifacts_dir/llama-cpp-cpu-smoke}"
index_path="$artifacts_dir/backend-index.json"
archive_path="$artifacts_dir/kapsl-backend-llama-cpp-cpu-${KAPSL_VERSION}-linux-x86_64.tar.gz"
engine_binary="$repo_root/kapsl-runtime/target/release/kapsl"
max_tokens="${KAPSL_LLAMA_PARITY_MAX_TOKENS:-32}"
load_timeout="${KAPSL_LLAMA_PARITY_LOAD_TIMEOUT_SECONDS:-300}"
reference_port="${KAPSL_LLAMA_PARITY_REFERENCE_PORT:-19115}"
candidate_port="${KAPSL_LLAMA_PARITY_CANDIDATE_PORT:-19116}"
runtime_pid=""
work_root="$(mktemp -d)"

stop_runtime() {
  if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
    kill "$runtime_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$runtime_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$runtime_pid" 2>/dev/null || true
    wait "$runtime_pid" 2>/dev/null || true
  fi
  runtime_pid=""
}
cleanup() {
  stop_runtime
  rm -rf "$work_root"
}
trap cleanup EXIT INT TERM

die() {
  echo "FAIL: $*" >&2
  exit 1
}

for command_name in cargo curl python3 sha256sum; do
  command -v "$command_name" >/dev/null 2>&1 \
    || die "$command_name is required for llama.cpp CPU parity"
done
for required in "$index_path" "${index_path}.sig" "$archive_path" "$KAPSL_LLAMA_PARITY_MODEL" "$parity_helper"; do
  [ -s "$required" ] || die "llama.cpp CPU parity input is missing or empty: $required"
done
[ ! -e "$evidence_dir" ] || die "refusing to reuse evidence directory: $evidence_dir"
for value in "$max_tokens" "$load_timeout" "$reference_port" "$candidate_port"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "expected a positive integer, got: $value"
done

actual_model_sha256="$(sha256sum "$KAPSL_LLAMA_PARITY_MODEL" | awk '{ print $1 }')"
[ "$actual_model_sha256" = "$KAPSL_LLAMA_PARITY_MODEL_SHA256" ] \
  || die "parity model digest is $actual_model_sha256, expected $KAPSL_LLAMA_PARITY_MODEL_SHA256"

# The default feature set is what portable Linux installers ship: no eager
# GGUF backend, so the pack is the default CPU route and the embedded
# kapsl_llm::GgufBackend is the explicit KAPSL_LAZY_LLAMA_CPP_PACKS=0 route.
KAPSL_VERSION="$KAPSL_VERSION" cargo build \
  --manifest-path "$repo_root/kapsl-runtime/Cargo.toml" \
  --package kapsl \
  --release \
  --locked

mkdir -p "$evidence_dir"
model_input_dir="$work_root/model-input"
mkdir -p "$model_input_dir"
# Hosted runners have limited disk; link the verified model when possible.
ln "$KAPSL_LLAMA_PARITY_MODEL" "$model_input_dir/model.gguf" 2>/dev/null \
  || cp "$KAPSL_LLAMA_PARITY_MODEL" "$model_input_dir/model.gguf"
model_package="$work_root/llama-cpu-parity.aimod"
"$engine_binary" build "$model_input_dir/model.gguf" \
  --output "$model_package" \
  --project-name llama-cpu-parity \
  --format gguf \
  --model-type causal-lm \
  --task generate \
  --version 1.0.0

bundle_path="$work_root/llama-cpu-parity.kapsl-bundle"
KAPSL_BACKEND_CACHE_DIR="$work_root/bundle-builder-backends" \
KAPSL_BACKEND_INDEX_PATH="$index_path" \
KAPSL_BACKEND_PUBLIC_KEYS="$KAPSL_LLAMA_PARITY_PUBLIC_KEY" \
KAPSL_LAZY_LLAMA_CPP_PACKS=1 \
  "$engine_binary" bundle "$model_package" \
    --output "$bundle_path" \
    --target linux-x86_64-cpu \
    --backend-artifacts-dir "$artifacts_dir"

candidate_backend_cache="$work_root/candidate-backends"
candidate_bundle_cache="$work_root/candidate-bundles"
KAPSL_BACKEND_CACHE_DIR="$candidate_backend_cache" \
KAPSL_BUNDLE_CACHE_DIR="$candidate_bundle_cache" \
KAPSL_BACKEND_PUBLIC_KEYS="$KAPSL_LLAMA_PARITY_PUBLIC_KEY" \
KAPSL_LAZY_LLAMA_CPP_PACKS=1 \
  "$engine_binary" backend ensure "$bundle_path" --offline \
    >"$evidence_dir/backend-ensure.log" 2>&1 \
  || { cat "$evidence_dir/backend-ensure.log" >&2; die "offline bundle did not install its backend packs"; }
KAPSL_BACKEND_CACHE_DIR="$candidate_backend_cache" \
KAPSL_BACKEND_PUBLIC_KEYS="$KAPSL_LLAMA_PARITY_PUBLIC_KEY" \
  "$engine_binary" backend list --json >"$evidence_dir/backend-list.json"
python3 - "$evidence_dir/backend-list.json" <<'PY'
import json
import pathlib
import sys

entries = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
matches = [
    entry
    for entry in entries
    if entry["backend"] == "llama-cpp" and entry["profile"] == "cpu" and entry["valid"]
]
if len(matches) != 1:
    raise SystemExit("offline bundle did not install exactly one valid llama-cpp/cpu pack")
PY

wait_for_model() {
  local base_url="$1"
  local log_file="$2"
  local deadline=$((SECONDS + load_timeout))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$runtime_pid" 2>/dev/null; then
      tail -n 200 "$log_file" >&2 || true
      die "runtime exited before its model became ready"
    fi
    if curl -fsS "$base_url/v1/models" 2>/dev/null | python3 -c '
import json, sys
body = json.load(sys.stdin)
entries = body.get("data") if isinstance(body, dict) else body
sys.exit(0 if entries else 1)
' 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  tail -n 200 "$log_file" >&2 || true
  die "runtime model did not become ready within ${load_timeout}s"
}

# run_case LABEL PORT LAZY_PACKS MODEL_OR_BUNDLE
run_case() {
  local label="$1"
  local port="$2"
  local lazy="$3"
  local input="$4"
  local log_file="$evidence_dir/$label-runtime.log"
  local backend_cache="$work_root/$label-backends"
  local bundle_cache="$work_root/$label-bundles"
  if [ "$label" = pack ]; then
    backend_cache="$candidate_backend_cache"
    bundle_cache="$candidate_bundle_cache"
  fi
  mkdir -p "$work_root/state-$label" "$work_root/model-cache-$label" "$backend_cache"

  env \
    "KAPSL_LAZY_LLAMA_CPP_PACKS=$lazy" \
    "KAPSL_BACKEND_CACHE_DIR=$backend_cache" \
    "KAPSL_BUNDLE_CACHE_DIR=$bundle_cache" \
    "KAPSL_BACKEND_PUBLIC_KEYS=$KAPSL_LLAMA_PARITY_PUBLIC_KEY" \
    "KAPSL_MODEL_CACHE_DIR=$work_root/model-cache-$label" \
    "KAPSL_API_TOKEN_READER=" \
    "KAPSL_API_TOKEN_WRITER=" \
    "KAPSL_API_TOKEN_ADMIN=" \
    "RUST_LOG=info" \
    "$engine_binary" run "$input" \
      --offline \
      --transport socket \
      --socket "$work_root/$label.sock" \
      --metrics-port "$port" \
      --http-bind 127.0.0.1 \
      --state-dir "$work_root/state-$label" \
      >"$log_file" 2>&1 &
  runtime_pid="$!"

  local base_url="http://127.0.0.1:$port"
  wait_for_model "$base_url" "$log_file"
  python3 "$parity_helper" run \
    --base-url "$base_url" \
    --label "$label" \
    --max-tokens "$max_tokens" \
    --output "$evidence_dir/$label-results.json"
  stop_runtime

  if [ "$lazy" = 1 ]; then
    grep -Fq 'Activated signed llama.cpp backend pack llama-cpp/cpu' "$log_file" \
      || die "the pack run did not activate the signed llama-cpp/cpu pack"
    ! grep -Fq 'Using GgufBackend (llama.cpp)' "$log_file" \
      || die "the pack run fell back to the embedded llama.cpp backend"
    ! grep -Fq 'Downloading Kapsl backend' "$log_file" \
      || die "the offline pack run tried to download a backend"
  else
    grep -Fq 'Using GgufBackend (llama.cpp)' "$log_file" \
      || die "the embedded run did not use the in-process llama.cpp backend"
    ! grep -Fq 'Activated signed llama.cpp backend pack' "$log_file" \
      || die "the embedded run activated a signed llama.cpp pack"
  fi
}

run_case embedded "$reference_port" 0 "$model_package"
run_case pack "$candidate_port" 1 "$bundle_path"

python3 "$parity_helper" compare \
  --reference "$evidence_dir/embedded-results.json" \
  --candidate "$evidence_dir/pack-results.json" \
  --output "$evidence_dir/comparison.json"
