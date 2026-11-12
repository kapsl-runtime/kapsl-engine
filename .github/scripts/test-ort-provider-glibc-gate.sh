#!/usr/bin/env bash
# Tests for fetch-microsoft-ort-providers.sh and check-linux-glibc-floor.sh using synthetic fixtures
# and a fake readelf (no network, no real ELF files).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scripts="$repo_root/.github/scripts"
test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT INT TERM

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

expect_fail() { # description, expected-message-fragment, log, command...
  local description="$1" fragment="$2" log="$3"
  shift 3
  if "$@" >"$log" 2>&1; then
    echo "Expected failure but succeeded: $description" >&2
    cat "$log" >&2
    exit 1
  fi
  if ! grep -qF -- "$fragment" "$log"; then
    echo "Failure for '$description' did not mention: $fragment" >&2
    cat "$log" >&2
    exit 1
  fi
}

# ---- fake readelf: reads a "GLIBC=x GLIBCXX=y ISOC23=0|1" marker out of the fixture file ----
fake_bin="$test_root/bin"
mkdir -p "$fake_bin"
cat > "$fake_bin/readelf" <<'FAKE'
#!/bin/sh
mode="$1"; file="$3"
marker="$(tr -c '[:print:]\n' ' ' < "$file" | grep -o 'GLIBC=[^ ]* GLIBCXX=[^ ]* ISOC23=[01]' | head -n 1)"
glibc="$(printf '%s' "$marker" | sed 's/GLIBC=\([^ ]*\) .*/\1/')"
glibcxx="$(printf '%s' "$marker" | sed 's/.*GLIBCXX=\([^ ]*\) .*/\1/')"
isoc="$(printf '%s' "$marker" | sed 's/.*ISOC23=//')"
case "$mode" in
  --version-info)
    [ "$glibc" = "none" ] || echo "  0x0010:   Name: GLIBC_2.2.5  Flags: none  Version: 3"
    [ "$glibc" = "none" ] || echo "  0x0020:   Name: GLIBC_$glibc  Flags: none  Version: 4"
    [ "$glibcxx" = "none" ] || echo "  0x0030:   Name: GLIBCXX_$glibcxx  Flags: none  Version: 5"
    ;;
  --dyn-syms)
    echo "     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND malloc@GLIBC_2.2.5 (3)"
    [ "$isoc" != 1 ] || echo "     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __isoc23_strtoll@GLIBC_2.38 (4)"
    ;;
esac
FAKE
chmod +x "$fake_bin/readelf"

make_elf() { # path, glibc, glibcxx, isoc23
  mkdir -p "$(dirname "$1")"
  printf '\177ELF fixture GLIBC=%s GLIBCXX=%s ISOC23=%s\n' "$2" "$3" "$4" > "$1"
}

gate() { PATH="$fake_bin:$PATH" "$scripts/check-linux-glibc-floor.sh" "$@"; }

# ---- gate: passing tree (Microsoft-like and engine-like floors) ----
ok="$test_root/ok"
make_elf "$ok/kapsl" 2.34 3.4.30 0
make_elf "$ok/libonnxruntime_providers_cuda.so" 2.27 3.4.21 0
make_elf "$ok/libonnxruntime_providers_shared.so" 2.2.5 none 0
printf 'not an elf\n' > "$ok/kapsl-provider-cuda12.json"
gate "$ok" > "$test_root/ok.log"
grep -q 'gate passed (3 ELF object(s))' "$test_root/ok.log"

# ---- gate: each violation class fails and names the file ----
bad_glibc="$test_root/bad-glibc"; make_elf "$bad_glibc/libonnxruntime_providers_cuda.so" 2.38 3.4.29 0
expect_fail "glibc 2.38" "needs GLIBC_2.38" "$test_root/l1" gate "$bad_glibc"
bad_cxx="$test_root/bad-cxx"; make_elf "$bad_cxx/libx.so" 2.31 3.4.33 0
expect_fail "libstdc++ too new" "needs GLIBCXX_3.4.33" "$test_root/l2" gate "$bad_cxx"
bad_c23="$test_root/bad-c23"; make_elf "$bad_c23/libx.so" 2.31 3.4.20 1
expect_fail "__isoc23" "imports C23 glibc symbols: __isoc23_strtoll" "$test_root/l3" gate "$bad_c23"
expect_fail "no ELF" "no ELF objects found" "$test_root/l4" gate "$test_root/ok/kapsl-provider-cuda12.json"
expect_fail "missing path" "No such file or directory" "$test_root/l5" gate "$test_root/does-not-exist"

# ---- gate: ceilings are configurable, archive input works ----
KAPSL_MAX_GLIBC=2.39 gate "$bad_glibc" > /dev/null
expect_fail "tight glibc ceiling" "allows at most GLIBC_2.30" "$test_root/l6" env KAPSL_MAX_GLIBC=2.30 PATH="$fake_bin:$PATH" "$scripts/check-linux-glibc-floor.sh" "$ok"
tar -czf "$test_root/ok.tar.gz" -C "$ok" .
gate --archive "$test_root/ok.tar.gz" > "$test_root/arch.log"
grep -q 'gate passed (3 ELF object(s))' "$test_root/arch.log"

# ---- fetch: synthetic Microsoft archive with pins ----
prefix="onnxruntime-linux-x64-gpu-9.9.9/lib"
src="$test_root/ms"
mkdir -p "$src/$prefix"
printf 'ms cuda provider\n' > "$src/$prefix/libonnxruntime_providers_cuda.so"
printf 'ms tensorrt provider\n' > "$src/$prefix/libonnxruntime_providers_tensorrt.so"
tar -czf "$test_root/ms.tgz" -C "$src" onnxruntime-linux-x64-gpu-9.9.9

write_pins() { # file, archive-sha, cuda-sha, trt-sha
  cat > "$1" <<PINS
MS_ORT_VERSION="9.9.9"
MS_ORT_URL="file://$test_root/ms.tgz"
MS_ORT_ARCHIVE_SHA256="$2"
MS_ORT_ARCHIVE_BYTES="$(wc -c < "$test_root/ms.tgz" | tr -d ' ')"
MS_ORT_MEMBER_PREFIX="$prefix"
MS_ORT_CUDA_SHA256="$3"
MS_ORT_TENSORRT_SHA256="$4"
PINS
}
good_archive_sha="$(sha256_of "$test_root/ms.tgz")"
good_cuda_sha="$(sha256_of "$src/$prefix/libonnxruntime_providers_cuda.so")"
good_trt_sha="$(sha256_of "$src/$prefix/libonnxruntime_providers_tensorrt.so")"
zero="0000000000000000000000000000000000000000000000000000000000000000"

cuda_dir="$test_root/cuda"; trt_dir="$test_root/trt"
mkdir -p "$cuda_dir" "$trt_dir"
printf 'pyke cuda provider\n' > "$cuda_dir/libonnxruntime_providers_cuda.so"
printf 'pyke tensorrt provider\n' > "$trt_dir/libonnxruntime_providers_tensorrt.so"

write_pins "$test_root/good.env" "$good_archive_sha" "$good_cuda_sha" "$good_trt_sha"
KAPSL_MS_ORT_PINS_FILE="$test_root/good.env" "$scripts/fetch-microsoft-ort-providers.sh" \
  cuda "$cuda_dir" tensorrt "$trt_dir" > "$test_root/fetch.log"
cmp "$cuda_dir/libonnxruntime_providers_cuda.so" "$src/$prefix/libonnxruntime_providers_cuda.so"
cmp "$trt_dir/libonnxruntime_providers_tensorrt.so" "$src/$prefix/libonnxruntime_providers_tensorrt.so"
test -x "$cuda_dir/libonnxruntime_providers_cuda.so"
test -z "$(find "$cuda_dir" -name '.*' -type f -print -quit)"

# A pin mismatch must fail and leave the staged file untouched.
printf 'pyke cuda provider\n' > "$cuda_dir/libonnxruntime_providers_cuda.so"
write_pins "$test_root/bad-archive.env" "$zero" "$good_cuda_sha" "$good_trt_sha"
expect_fail "archive sha mismatch" "archive SHA-256" "$test_root/f1" \
  env KAPSL_MS_ORT_PINS_FILE="$test_root/bad-archive.env" "$scripts/fetch-microsoft-ort-providers.sh" cuda "$cuda_dir"
write_pins "$test_root/bad-member.env" "$good_archive_sha" "$zero" "$good_trt_sha"
expect_fail "member sha mismatch" "expected $zero" "$test_root/f2" \
  env KAPSL_MS_ORT_PINS_FILE="$test_root/bad-member.env" "$scripts/fetch-microsoft-ort-providers.sh" cuda "$cuda_dir"
grep -q 'pyke cuda provider' "$cuda_dir/libonnxruntime_providers_cuda.so"
expect_fail "unknown provider" "Unknown provider" "$test_root/f3" \
  env KAPSL_MS_ORT_PINS_FILE="$test_root/good.env" "$scripts/fetch-microsoft-ort-providers.sh" rocm "$cuda_dir"
expect_fail "odd arguments" "usage:" "$test_root/f4" \
  env KAPSL_MS_ORT_PINS_FILE="$test_root/good.env" "$scripts/fetch-microsoft-ort-providers.sh" cuda

# The committed pins file must define everything the script needs.
for var in MS_ORT_VERSION MS_ORT_URL MS_ORT_ARCHIVE_SHA256 MS_ORT_ARCHIVE_BYTES MS_ORT_MEMBER_PREFIX MS_ORT_CUDA_SHA256 MS_ORT_TENSORRT_SHA256; do
  grep -q "^${var}=\"[^\"]\+\"" "$repo_root/.github/ort-microsoft-providers.env"
done
grep -Eq '^MS_ORT_ARCHIVE_SHA256="[0-9a-f]{64}"' "$repo_root/.github/ort-microsoft-providers.env"

echo "ORT provider fetch and glibc floor gate tests passed."
