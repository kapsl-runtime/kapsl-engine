#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"
: "${KAPSL_ORT_INTEGRATIONS_REF:?KAPSL_ORT_INTEGRATIONS_REF is required}"

if [ "${RUNNER_OS:-Linux}" != "Linux" ] || [ "${RUNNER_ARCH:-X64}" != "X64" ]; then
  echo "The ORT CPU adapter is currently packaged only for Linux x86_64." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
integrations_dir="${KAPSL_ORT_INTEGRATIONS_DIR:-$repo_root/kapsl-integrations-ort}"
verifier="$repo_root/.github/scripts/verify-ort-integration-checkout.sh"
build_script="$integrations_dir/integrations/ort/packaging/build_cpu_pack.sh"
output_dir="${KAPSL_ORT_RELEASE_OUTPUT_DIR:-$repo_root/dist}"
archive_name="kapsl-backend-onnx-cpu-${KAPSL_VERSION}-linux-x86_64.tar.gz"
archive="$output_dir/$archive_name"
manifest="${archive}.manifest.json"
checksum="${archive}.sha256"
signature="${archive}.sig"

for command_name in git python3 sha256sum tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to package the ORT CPU adapter." >&2
    exit 1
  fi
done
"$verifier" "$integrations_dir" "$KAPSL_ORT_INTEGRATIONS_REF"
if [ ! -x "$build_script" ]; then
  echo "Certified ORT packaging entrypoint is missing or not executable: $build_script" >&2
  exit 1
fi
for output in "$archive" "$manifest" "$checksum" "$signature"; do
  if [ -e "$output" ]; then
    echo "Refusing to overwrite an existing ORT CPU release output: $output" >&2
    exit 1
  fi
done

mkdir -p "$output_dir"
KAPSL_ORT_PACK_OUTPUT_DIR="$output_dir" \
KAPSL_ORT_PACK_BUILD_DIR="${KAPSL_ORT_PACK_BUILD_DIR:-${RUNNER_TEMP:-/tmp}/kapsl-ort-cpu-pack}" \
KAPSL_VERSION="$KAPSL_VERSION" \
  "$build_script"
"$verifier" "$integrations_dir" "$KAPSL_ORT_INTEGRATIONS_REF"

for required in "$archive" "$manifest" "$checksum"; do
  if [ ! -s "$required" ]; then
    echo "Certified ORT packaging did not produce $required" >&2
    exit 1
  fi
done
if [ -e "$signature" ]; then
  echo "The integrations handoff must not sign release artifacts; the engine index publisher owns the release key." >&2
  exit 1
fi
(cd "$output_dir" && sha256sum --check "${archive_name}.sha256")

python3 - "$archive" "$manifest" "$KAPSL_VERSION" "$KAPSL_ORT_INTEGRATIONS_REF" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
import tarfile

archive_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
version = sys.argv[3]
source_ref = sys.argv[4]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
runtime_soname = "libonnxruntime.so.1"
runtime_version = "1.23.2"
runtime_distribution_url = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    "v1.23.2/onnxruntime-linux-x64-1.23.2.tgz"
)
runtime_distribution_sha256 = (
    "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b"
)
maximum_glibc = (2, 35)
expected = {
    "schema_version": 1,
    "backend": "onnx",
    "profile": "cpu",
    "runtime_abi": 1,
    "adapter_abi": "kapsl-backend-v1",
    "compatible_kapsl": f"={version}",
    "platform": "linux-x86_64",
    "architecture": "x86_64",
    "accelerator_profile": "cpu",
    "execution_mode": "native",
    "entrypoint": "libkapsl_backend_ort.so",
    "priority": 200,
}
for field, value in expected.items():
    if manifest.get(field) != value:
        raise SystemExit(
            f"{manifest_path}: {field}={manifest.get(field)!r}, expected {value!r}"
        )
for forbidden in ("artifact", "download_bytes", "sha256", "signature"):
    if forbidden in manifest:
        raise SystemExit(f"{manifest_path}: handoff template must omit {forbidden}")

members = {}
with tarfile.open(archive_path, "r:gz") as archive:
    for member in archive.getmembers():
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"{archive_path}: unsafe member {member.name}")
        normalized = path.as_posix().removeprefix("./")
        if member.isfile():
            if normalized in members:
                raise SystemExit(f"{archive_path}: duplicate member {normalized}")
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit(f"{archive_path}: cannot read {normalized}")
            members[normalized] = stream.read()

for required in (
    "backend-pack.json",
    "provenance.json",
    expected["entrypoint"],
    runtime_soname,
):
    if required not in members:
        raise SystemExit(f"{archive_path}: missing {required}")
payload = json.loads(members["backend-pack.json"])
for field in (
    "schema_version",
    "backend",
    "profile",
    "pack_version",
    "runtime_abi",
    "adapter_abi",
    "platform",
    "execution_mode",
    "entrypoint",
):
    if payload.get(field) != manifest.get(field):
        raise SystemExit(f"{archive_path}: payload/template mismatch for {field}")

signed_files = manifest.get("files")
if not isinstance(signed_files, dict) or set(signed_files) != set(members):
    raise SystemExit(f"{manifest_path}: signed file set does not exactly match the archive")
for name, payload_bytes in members.items():
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if signed_files.get(name) != digest:
        raise SystemExit(f"{manifest_path}: digest mismatch for {name}")

provenance = json.loads(members["provenance.json"])
if provenance.get("source_repository") != "https://github.com/kapsl-runtime/kapsl-integrations":
    raise SystemExit(f"{archive_path}: unexpected source repository")
if provenance.get("source_commit") != source_ref:
    raise SystemExit(f"{archive_path}: source commit does not match the certified checkout")
if provenance.get("adapter", {}).get("adapter_abi") != "kapsl-backend-v1":
    raise SystemExit(f"{archive_path}: provenance does not identify the standard adapter ABI")

onnx_runtime = provenance.get("onnx_runtime")
if not isinstance(onnx_runtime, dict):
    raise SystemExit(f"{archive_path}: provenance is missing ONNX Runtime metadata")
runtime_expected = {
    "version": runtime_version,
    "distribution_url": runtime_distribution_url,
    "distribution_sha256": runtime_distribution_sha256,
}
for field, value in runtime_expected.items():
    if onnx_runtime.get(field) != value:
        raise SystemExit(
            f"{archive_path}: onnx_runtime.{field}={onnx_runtime.get(field)!r}, "
            f"expected {value!r}"
        )

runtime_library = onnx_runtime.get("library")
if not isinstance(runtime_library, dict):
    raise SystemExit(f"{archive_path}: provenance is missing ONNX Runtime library metadata")
for field in ("path", "soname"):
    if runtime_library.get(field) != runtime_soname:
        raise SystemExit(
            f"{archive_path}: onnx_runtime.library.{field} must be {runtime_soname}"
        )
if runtime_library.get("sha256") != signed_files[runtime_soname]:
    raise SystemExit(
        f"{archive_path}: ONNX Runtime provenance digest does not match the signed file"
    )

entrypoint = provenance.get("entrypoint")
if not isinstance(entrypoint, dict) or entrypoint.get("path") != expected["entrypoint"]:
    raise SystemExit(f"{archive_path}: provenance entrypoint path is invalid")
if entrypoint.get("sha256") != signed_files[expected["entrypoint"]]:
    raise SystemExit(f"{archive_path}: entrypoint provenance digest does not match the signed file")
needed_libraries = entrypoint.get("needed_libraries")
if not isinstance(needed_libraries, list) or runtime_soname not in needed_libraries:
    raise SystemExit(
        f"{archive_path}: entrypoint does not declare the pack-local ONNX Runtime dependency"
    )

def parse_glibc(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value):
        raise SystemExit(f"{archive_path}: {label} is not a GLIBC version")
    return tuple(int(component) for component in value.split("."))


build = provenance.get("build")
if not isinstance(build, dict) or build.get("maximum_permitted_glibc") != "2.35":
    raise SystemExit(f"{archive_path}: provenance does not enforce the GLIBC 2.35 policy")
for metadata, label in (
    (runtime_library, "ONNX Runtime library"),
    (entrypoint, "ORT adapter entrypoint"),
):
    required_glibc = parse_glibc(metadata.get("maximum_required_glibc"), label)
    if required_glibc > maximum_glibc:
        raise SystemExit(
            f"{archive_path}: {label} requires GLIBC {metadata['maximum_required_glibc']}, "
            "which exceeds 2.35"
        )
PY

echo "Accepted certified ORT CPU adapter from $KAPSL_ORT_INTEGRATIONS_REF"
