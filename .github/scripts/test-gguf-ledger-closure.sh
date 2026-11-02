#!/usr/bin/env bash
# Opt-in, real-GPU check that the memory ledger matches physical usage when two
# GGUF models are resident on one CUDA device.
#
# It starts the runtime with one GGUF model, then starts a second distinct one,
# and after each step reads /api/system/stats and /metrics and requires, for the
# CUDA domain under test, that
#
#   max(domain reserved, domain committed)   (the model leases in that domain)
#   + runtime-overhead row                   (owner_kind "runtime")
#   + free bytes of the device pool backing  (physical, but in no row)
#
# is within KAPSL_GPU_TEST_LEDGER_TOLERANCE_PCT (default 5) of BOTH
#   * the process's own GPU memory as reported by the driver (nvidia-smi), which
#     is independent of the ledger, and
#   * the domain's `observed_bytes` in the stats response.
#
# The nvidia-smi comparison is the one that cannot pass by construction: the
# domain's observed figure is max(sampled, attributed), so an over-attributed
# ledger would otherwise agree with itself. The runtime row is measured once
# after CUDA init, before any model, so it cannot absorb per-model
# under-reporting either. The `models[]` sum in the stats response spans every
# domain (host rows included), so it is printed as a diagnostic only.
#
# Per-model vs fixed overhead: the residual (process - ledger) after one model
# and after two is printed. A residual that grows with the second model is
# per-model overhead and belongs in that model's lease.
#
# Required when KAPSL_GPU_INTEGRATION=1:
#   KAPSL_GPU_TEST_BINARY        CUDA kapsl binary (production `cuda` profile)
#   KAPSL_GPU_TEST_GGUF_MODEL    first GGUF model (.gguf or .aimod)
#   KAPSL_GPU_TEST_GGUF_MODEL_B  second GGUF model; must be a different file so
#                                the two models do not share cached weights
#
# Optional:
#   KAPSL_GPU_TEST_CUDA_VISIBLE_DEVICES  one physical GPU/UUID (default: 0)
#   KAPSL_GPU_TEST_DOMAIN                stats domain name (default: cuda:0)
#   KAPSL_GPU_TEST_LEDGER_TOLERANCE_PCT  allowed relative gap (default: 5)
#   KAPSL_GPU_TEST_GPU_MEMORY_LIMIT_MB   passed as KAPSL_GPU_MEMORY_LIMIT_MB
#   KAPSL_GPU_TEST_LOAD_TIMEOUT_SECONDS  per-model readiness timeout (default: 300)
#   KAPSL_GPU_TEST_PORT                  HTTP port (default: 19096)
#   KAPSL_GPU_TEST_OUTPUT_DIR            retained logs and snapshots
#
# `--self-test` runs on any machine (no GPU, no binary) and checks the closure
# rule against fixtures, including the figures from the incident that motivated
# this test (two 0.5B models: 0.39 GB leases against 2.43 GB observed).
set -Eeuo pipefail

script_name="$(basename "$0")"
runtime_pid=""
api_admin_token=""
created_output_dir=0

usage() {
  cat <<EOF
Usage:
  $script_name --self-test
  KAPSL_GPU_INTEGRATION=1 \\
    KAPSL_GPU_TEST_BINARY=/path/to/kapsl \\
    KAPSL_GPU_TEST_GGUF_MODEL=/models/a.gguf \\
    KAPSL_GPU_TEST_GGUF_MODEL_B=/models/b.gguf \\
    $script_name

Without KAPSL_GPU_INTEGRATION=1 the script exits successfully with SKIP.
EOF
}

die() {
  echo "FAIL: $*" >&2
  exit 1
}

note() {
  echo "gguf-ledger-closure: $*"
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

# ledger_closure_report STATS_JSON DOMAIN PROCESS_BYTES TOLERANCE_PCT POOL_FREE_BYTES
#
# Prints one JSON object of figures and exits 0 on success, nonzero on
# malformed input. `vs_observed_ok` / `vs_process_ok` carry the verdicts.
# PROCESS_BYTES is the driver-reported GPU memory of the runtime process, or
# the word "unavailable" to skip that comparison. POOL_FREE_BYTES is the
# device pool backing that is allocated but unused (0 when there is no pool).
ledger_closure_report() {
  local stats="$1" domain="$2" process_bytes="$3" tolerance_pct="$4" pool_free="$5"
  jq -e --arg domain "$domain" --arg process "$process_bytes" \
    --argjson tol "$tolerance_pct" --argjson pool_free "$pool_free" '
    def within($a; $b): if $b == 0 then $a == 0
      else (($a - $b) | fabs) * 100 / $b <= $tol end;
    (.memory_authority.models // []) as $models
    | ([.memory_authority.domains[]? | select(.domain == $domain)] | first) as $d
    | if $d == null then error("domain \($domain) missing from stats") else . end
    | ([$d.reserved_bytes, $d.committed_bytes] | max) as $leases
    | ([$models[] | select(.owner_kind == "runtime") | .used_bytes] | add // 0) as $runtime
    | ($leases + $runtime + $pool_free) as $ledger
    | ($d.observed_bytes) as $observed
    | ($process | if . == "unavailable" then null else tonumber end) as $proc
    | {
        leases: $leases, runtime: $runtime, pool_free: $pool_free, ledger: $ledger,
        observed: $observed, process: $proc,
        residual: (if $proc == null then null else ($proc - $ledger) end),
        models_sum_all_domains: ([$models[] | select(.owner_kind == "model") | .used_bytes] | add // 0),
        vs_observed_ok: within($ledger; $observed),
        vs_process_ok: (if $proc == null then true else within($ledger; $proc) end)
      }' "$stats"
}

assert_ledger_closes() {
  local stats="$1" domain="$2" process_bytes="$3" tolerance_pct="$4" pool_free="${5:-0}" report
  report="$(ledger_closure_report "$stats" "$domain" "$process_bytes" "$tolerance_pct" "$pool_free")" \
    || die "stats at $stats are not parseable or lack domain $domain"
  note "ledger figures: $(jq -c . <<<"$report")"
  jq -e '.vs_observed_ok' <<<"$report" >/dev/null \
    || die "ledger ($(jq -r .ledger <<<"$report")) is not within ${tolerance_pct}% of domain observed ($(jq -r .observed <<<"$report"))"
  jq -e '.vs_process_ok' <<<"$report" >/dev/null \
    || die "ledger ($(jq -r .ledger <<<"$report")) is not within ${tolerance_pct}% of driver-reported process memory ($(jq -r .process <<<"$report"))"
}

# fixture NAME LEASES_A LEASES_B RUNTIME OBSERVED [HOST_EXTRA]
# HOST_EXTRA adds host-domain bytes to the per-model aggregates only: the
# `models[]` entries span every domain, the cuda domain row does not.
fixture() {
  local name="$1" leases_a="$2" leases_b="$3" runtime="$4" observed="$5" host_extra="${6:-0}"
  local file="$self_test_dir/$name.json"
  jq -n --argjson a "$leases_a" --argjson b "$leases_b" --argjson r "$runtime" \
    --argjson o "$observed" --argjson h "$host_extra" '
    {memory_authority: {
      models: ([
        {model_id: 0, owner_kind: "model", used_bytes: ($a + $h)},
        {model_id: 1, owner_kind: "model", used_bytes: ($b + $h)}
      ] + (if $r > 0 then [{model_id: 1073741824, owner_kind: "runtime", used_bytes: $r}] else [] end)),
      domains: [{domain: "cuda:0", observed_bytes: $o,
                 reserved_bytes: ($a + $b), committed_bytes: ($a + $b)}]}}' >"$file"
  printf '%s\n' "$file"
}

run_self_test() {
  require_command jq
  self_test_dir="$(mktemp -d)"
  trap 'rm -rf "$self_test_dir"' EXIT
  local mib=$((1024 * 1024))

  # The incident: two 0.5B models, leases of ~0.39 GB each, observed 2.43 GB.
  local before
  before="$(fixture before_fix $((410 * mib)) $((410 * mib)) 0 $((2490 * mib)))"
  ( assert_ledger_closes "$before" cuda:0 unavailable 5 ) >/dev/null 2>&1 \
    && die "self-test accepted the undercounted ledger from the incident"

  # After the fix: ~1.04 GB per lease (weights + compute buffer) plus the
  # runtime row, closing against both observed and the driver's process figure.
  local after
  after="$(fixture after_fix $((1065 * mib)) $((1065 * mib)) $((370 * mib)) $((2500 * mib)))"
  ( assert_ledger_closes "$after" cuda:0 $((2495 * mib)) 5 ) >/dev/null 2>&1 \
    || die "self-test rejected a ledger that closes"

  # Host-domain rows inflate the per-model aggregates but not the cuda domain;
  # the closure must be computed from the domain, not from models[].
  local host_heavy
  host_heavy="$(fixture host_heavy $((1065 * mib)) $((1065 * mib)) $((370 * mib)) $((2500 * mib)) $((900 * mib)))"
  ( assert_ledger_closes "$host_heavy" cuda:0 $((2495 * mib)) 5 ) >/dev/null 2>&1 \
    || die "self-test let host-domain rows break the cuda closure"

  # An auto pool's unused backing is in the driver's process figure but in no
  # row: it must be added from the pool metrics, and omitting it must fail.
  local pooled
  pooled="$(fixture pooled $((1065 * mib)) $((1065 * mib)) $((370 * mib)) $((4000 * mib)))"
  ( assert_ledger_closes "$pooled" cuda:0 $((2495 * mib + 1500 * mib)) 5 $((1500 * mib)) ) >/dev/null 2>&1 \
    || die "self-test rejected a ledger that closes once pool backing is added"
  ( assert_ledger_closes "$pooled" cuda:0 $((2495 * mib + 1500 * mib)) 5 0 ) >/dev/null 2>&1 \
    && die "self-test ignored unused pool backing"

  # Over-attribution must not pass: observed follows the ledger (it is
  # max(sampled, attributed)), but the driver's process figure does not.
  local over
  over="$(fixture over_attributed $((1700 * mib)) $((1700 * mib)) $((370 * mib)) $((3770 * mib)))"
  ( assert_ledger_closes "$over" cuda:0 $((2495 * mib)) 5 ) >/dev/null 2>&1 \
    && die "self-test accepted an over-attributed ledger"

  # A missing runtime row leaves a gap larger than the tolerance.
  local no_runtime
  no_runtime="$(fixture no_runtime $((1065 * mib)) $((1065 * mib)) 0 $((2500 * mib)))"
  ( assert_ledger_closes "$no_runtime" cuda:0 $((2495 * mib)) 5 ) >/dev/null 2>&1 \
    && die "self-test accepted a ledger missing its runtime row"

  # Boundary: exactly 5% passes, just over 5% fails.
  local edge_ok edge_bad
  edge_ok="$(fixture edge_ok 1000 0 0 1050)"
  ( assert_ledger_closes "$edge_ok" cuda:0 unavailable 5 ) >/dev/null 2>&1 \
    || die "self-test rejected a ledger exactly at the tolerance"
  edge_bad="$(fixture edge_bad 1000 0 0 1053)"
  ( assert_ledger_closes "$edge_bad" cuda:0 unavailable 5 ) >/dev/null 2>&1 \
    && die "self-test accepted a ledger just outside the tolerance"

  # The residual is reported so per-model overhead is visible.
  local residual
  residual="$(ledger_closure_report "$after" cuda:0 $((2600 * mib)) 5 0 | jq -r .residual)"
  [[ "$residual" == "$((2600 * mib - 2500 * mib))" ]] \
    || die "self-test mis-reported the residual: $residual"

  # A stats document without the domain is an error, not a pass.
  ( ledger_closure_report "$before" cuda:7 unavailable 5 0 ) >/dev/null 2>&1 \
    && die "self-test accepted a missing domain"

  note "ledger-closure self-test passed"
}

cleanup() {
  local status="$?"
  if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
    kill -TERM "$runtime_pid" 2>/dev/null || true
    wait "$runtime_pid" 2>/dev/null || true
  fi
  if [[ -n "${output_dir:-}" ]]; then
    if (( status == 0 )) && (( created_output_dir == 1 )); then
      rm -rf "$output_dir"
    else
      echo "ledger-closure artifacts: $output_dir" >&2
    fi
  fi
  return "$status"
}

api_get() {
  curl -fsS --max-time 15 -H "Authorization: Bearer $api_admin_token" "$base_url$1"
}

api_post() {
  local path="$1"
  shift
  curl -fsS --max-time 30 -H "Authorization: Bearer $api_admin_token" -X POST "$base_url$path" "$@"
}

# pool_free_bytes METRICS_FILE DEVICE_INDEX: unallocated bytes of the runtime
# device pool backing, or 0 when there is no pool or the metric is absent.
pool_free_bytes() {
  awk -v device="device=\"$2\"" '
    $1 ~ /^kapsl_gpu_device_pool_free_bytes\{/ && index($1, device) { print $2; found = 1; exit }
    END { if (!found) print 0 }
  ' "$1" | awk '{ printf "%.0f\n", $1 }'
}

# check_ledger_step LABEL: snapshot stats and metrics, assert the ledger closes.
check_ledger_step() {
  local label="$1" process_bytes device_index pool_free
  api_get /api/system/stats >"$output_dir/stats-$label.json"
  api_get /metrics >"$output_dir/metrics-$label.txt"
  process_bytes="$(process_vram_bytes || true)"
  process_bytes="${process_bytes:-unavailable}"
  [[ "$process_bytes" != unavailable ]] \
    || note "driver did not report per-process memory; checking against observed only"
  device_index="${domain##*:}"
  pool_free="$(pool_free_bytes "$output_dir/metrics-$label.txt" "$device_index")"
  note "step $label: pool backing free=$pool_free"
  assert_ledger_closes "$output_dir/stats-$label.json" "$domain" "$process_bytes" \
    "$tolerance_pct" "$pool_free"
}

runtime_is_alive() {
  [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null
}

wait_for_models_active() {
  local want="$1" deadline=$((SECONDS + load_timeout_seconds)) active
  while (( SECONDS < deadline )); do
    active="$(api_get /api/models 2>/dev/null | jq '[.[] | select(.status == "active")] | length' || echo 0)"
    [[ "$active" == "$want" ]] && return 0
    runtime_is_alive || return 1
    sleep 1
  done
  return 1
}

process_vram_bytes() {
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v wanted="$runtime_pid" '
        {
          pid=$1; memory=$2
          gsub(/[[:space:]]/, "", pid)
          gsub(/[[:space:]]/, "", memory)
          if (pid == wanted && memory ~ /^[0-9]+$/) { total += memory; found = 1 }
        }
        END { if (found) printf "%.0f\n", total * 1024 * 1024 }
      '
}

main() {
  if ! is_true "${KAPSL_GPU_INTEGRATION:-0}"; then
    echo "SKIP: set KAPSL_GPU_INTEGRATION=1 and provide the binary plus two distinct GGUF models"
    return 0
  fi

  require_command curl
  require_command jq
  require_command nvidia-smi
  require_command python3

  local binary="${KAPSL_GPU_TEST_BINARY:-}"
  local model_a="${KAPSL_GPU_TEST_GGUF_MODEL:-}"
  local model_b="${KAPSL_GPU_TEST_GGUF_MODEL_B:-}"
  [[ -x "$binary" ]] || die "KAPSL_GPU_TEST_BINARY is missing or not executable: $binary"
  [[ -s "$model_a" ]] || die "KAPSL_GPU_TEST_GGUF_MODEL is missing or empty: $model_a"
  [[ -s "$model_b" ]] || die "KAPSL_GPU_TEST_GGUF_MODEL_B is missing or empty: $model_b"
  [[ "$(cd "$(dirname "$model_a")" && pwd)/$(basename "$model_a")" \
    != "$(cd "$(dirname "$model_b")" && pwd)/$(basename "$model_b")" ]] \
    || die "the two GGUF models must be different files (identical files share cached weights)"

  tolerance_pct="${KAPSL_GPU_TEST_LEDGER_TOLERANCE_PCT:-5}"
  [[ "$tolerance_pct" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "tolerance must be a non-negative number"
  load_timeout_seconds="${KAPSL_GPU_TEST_LOAD_TIMEOUT_SECONDS:-300}"
  [[ "$load_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || die "load timeout must be a positive integer"
  domain="${KAPSL_GPU_TEST_DOMAIN:-cuda:0}"
  settle_seconds="${KAPSL_GPU_TEST_SETTLE_SECONDS:-5}"
  [[ "$settle_seconds" =~ ^[0-9]+$ ]] || die "settle seconds must be a non-negative integer"
  local cuda_visible_devices="${KAPSL_GPU_TEST_CUDA_VISIBLE_DEVICES:-0}"
  local port="${KAPSL_GPU_TEST_PORT:-19096}"
  [[ "$port" =~ ^[0-9]+$ ]] && (( port > 0 && port < 65536 )) || die "invalid KAPSL_GPU_TEST_PORT: $port"
  base_url="http://127.0.0.1:$port"

  if [[ -n "${KAPSL_GPU_TEST_OUTPUT_DIR:-}" ]]; then
    output_dir="$KAPSL_GPU_TEST_OUTPUT_DIR"
    mkdir -p "$output_dir"
  else
    output_dir="$(mktemp -d)"
    created_output_dir=1
  fi
  mkdir -p "$output_dir/state"
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  nvidia-smi -L >"$output_dir/nvidia-smi.txt"
  api_admin_token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  local env_args=(
    "CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
    "KAPSL_API_TOKEN_READER="
    "KAPSL_API_TOKEN_WRITER="
    "KAPSL_API_TOKEN_ADMIN=$api_admin_token"
    "KAPSL_ALLOW_INSECURE_HTTP=1"
    "RUST_LOG=${RUST_LOG:-info}"
  )
  if [[ -n "${KAPSL_GPU_TEST_GPU_MEMORY_LIMIT_MB:-}" ]]; then
    env_args+=("KAPSL_GPU_MEMORY_LIMIT_MB=$KAPSL_GPU_TEST_GPU_MEMORY_LIMIT_MB")
  fi

  note "starting GGUF model A on CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
  env "${env_args[@]}" "$binary" run \
    --model "$model_a" \
    --transport socket --socket "$output_dir/kapsl.sock" \
    --metrics-port "$port" --http-bind 127.0.0.1 \
    --state-dir "$output_dir/state" \
    >"$output_dir/runtime.log" 2>&1 &
  runtime_pid="$!"

  wait_for_models_active 1 || {
    tail -n 200 "$output_dir/runtime.log" >&2 || true
    die "GGUF model A did not become active"
  }
  # Let post-load reconciliation and allocator settling finish.
  sleep "$settle_seconds"
  check_ledger_step one-model

  note "starting GGUF model B"
  local start_payload
  start_payload="$(jq -cn --arg path "$model_b" \
    '{model_path:$path,topology:"data-parallel",tp_degree:1}')"
  api_post /api/models/start -H 'Content-Type: application/json' \
    --data-binary "$start_payload" >"$output_dir/start-model-b.json" || {
    tail -n 200 "$output_dir/runtime.log" >&2 || true
    die "starting GGUF model B was rejected (see the runtime log)"
  }
  wait_for_models_active 2 || {
    tail -n 200 "$output_dir/runtime.log" >&2 || true
    die "GGUF model B did not become active"
  }
  sleep "$settle_seconds"
  check_ledger_step two-models

  # The residual (process - ledger) is printed in each step's figures. One that
  # grows with the second model is per-model overhead and belongs in that
  # model's lease; a constant one is runtime overhead the row missed.
  note "PASS: ledger closed against ${domain} within ${tolerance_pct}% after one and after two models"
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --self-test)
    run_self_test
    exit 0
    ;;
  "")
    main
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
