#!/usr/bin/env bash
# Release gate: every ELF object that will ship must run on Ubuntu 22.04. Fails if any needs a glibc
# newer than KAPSL_MAX_GLIBC, a libstdc++ newer than KAPSL_MAX_GLIBCXX, or imports any __isoc23_*
# symbol (glibc >= 2.38 headers redirect strtol/strtoll/strtoull to them, which pins the object to 2.38).
#
# usage: check-linux-glibc-floor.sh [--archive FILE.tar.gz]... PATH...   (PATH is a file or a directory,
#        searched recursively; non-ELF files are ignored; finding no ELF at all is an error)
# env:   KAPSL_MAX_GLIBC   highest allowed GLIBC_x.y   (default 2.35, Ubuntu 22.04)
#        KAPSL_MAX_GLIBCXX highest allowed GLIBCXX_x   (default 3.4.30, libstdc++ of GCC 12 on Ubuntu 22.04)
set -euo pipefail

max_glibc="${KAPSL_MAX_GLIBC:-2.35}"
max_glibcxx="${KAPSL_MAX_GLIBCXX:-3.4.30}"

if ! command -v readelf >/dev/null 2>&1; then
  echo "readelf is required for the glibc floor gate." >&2
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT INT TERM

targets=()
archive_index=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --archive)
      [ "$#" -ge 2 ] || { echo "--archive needs a file" >&2; exit 2; }
      archive_index=$((archive_index + 1))
      mkdir -p "$work/archive-$archive_index"
      tar -xzf "$2" -C "$work/archive-$archive_index"
      targets+=("$work/archive-$archive_index")
      shift 2
      ;;
    *)
      targets+=("$1")
      shift
      ;;
  esac
done
if [ "${#targets[@]}" -eq 0 ]; then
  echo "usage: $0 [--archive FILE.tar.gz]... PATH..." >&2
  exit 2
fi

# True when $1 is strictly greater than $2 in version order.
version_gt() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1)" = "$1" ]
}

is_elf() {
  [ -f "$1" ] && [ "$(head -c 4 "$1" 2>/dev/null | od -An -tx1 | tr -d ' \n')" = "7f454c46" ]
}

checked=0
failures=0
check_file() {
  local file="$1" label="$2" info newest newest_cxx isoc23
  info="$(readelf --version-info --wide "$file" 2>/dev/null || true)"
  newest="$(printf '%s\n' "$info" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -uV | tail -n 1 || true)"
  newest_cxx="$(printf '%s\n' "$info" | grep -o 'GLIBCXX_[0-9][0-9.]*' | sed 's/^GLIBCXX_//' | sort -uV | tail -n 1 || true)"
  isoc23="$(readelf --dyn-syms --wide "$file" 2>/dev/null | grep -o '__isoc23_[A-Za-z0-9_]*' | sort -u | tr '\n' ' ' || true)"
  checked=$((checked + 1))
  printf '  %-60s GLIBC %-7s GLIBCXX %-8s\n' "$label" "${newest:--}" "${newest_cxx:--}"
  if [ -n "$newest" ] && version_gt "$newest" "$max_glibc"; then
    echo "FAIL: $label needs GLIBC_$newest; the gate allows at most GLIBC_$max_glibc" >&2
    failures=$((failures + 1))
  fi
  if [ -n "$newest_cxx" ] && version_gt "$newest_cxx" "$max_glibcxx"; then
    echo "FAIL: $label needs GLIBCXX_$newest_cxx; the gate allows at most GLIBCXX_$max_glibcxx" >&2
    failures=$((failures + 1))
  fi
  if [ -n "$isoc23" ]; then
    echo "FAIL: $label imports C23 glibc symbols: $isoc23" >&2
    failures=$((failures + 1))
  fi
}

echo "glibc floor gate: GLIBC <= $max_glibc, GLIBCXX <= $max_glibcxx, no __isoc23_*"
for target in "${targets[@]}"; do
  if [ -d "$target" ]; then
    while IFS= read -r file; do
      if is_elf "$file"; then
        check_file "$file" "${file#"$target"/}"
      fi
    done < <(find "$target" -type f | sort)
  elif [ -f "$target" ]; then
    if is_elf "$target"; then
      check_file "$target" "$(basename "$target")"
    fi
  else
    echo "No such file or directory: $target" >&2
    exit 1
  fi
done

if [ "$checked" -eq 0 ]; then
  echo "FAIL: no ELF objects found under: ${targets[*]}" >&2
  exit 1
fi
if [ "$failures" -ne 0 ]; then
  echo "glibc floor gate FAILED ($failures violation(s) across $checked ELF object(s))." >&2
  exit 1
fi
echo "glibc floor gate passed ($checked ELF object(s))."
