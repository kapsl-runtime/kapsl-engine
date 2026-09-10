#!/usr/bin/env bash
set -euo pipefail

: "${KAPSL_VERSION:?KAPSL_VERSION is required}"
: "${KAPSL_ORT_INTEGRATIONS_REF:?KAPSL_ORT_INTEGRATIONS_REF is required}"

if [ "${RUNNER_OS:-Linux}" != "Linux" ] || [ "${RUNNER_ARCH:-X64}" != "X64" ]; then
  echo "The ORT accelerator adapters are currently packaged only for Linux x86_64." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
integrations_dir="${KAPSL_ORT_INTEGRATIONS_DIR:-$repo_root/kapsl-integrations-ort}"
verifier="$repo_root/.github/scripts/verify-ort-integration-checkout.sh"
build_script="$integrations_dir/integrations/ort/packaging/build_accelerator_packs.sh"
output_dir="${KAPSL_ORT_RELEASE_OUTPUT_DIR:-$repo_root/dist}"
cuda_runtime_root="${KAPSL_CUDA_RUNTIME_ROOT:-${RUNNER_TEMP:?RUNNER_TEMP is required}/kapsl-${KAPSL_VERSION}-linux-x86_64-cuda12}"
tensorrt_runtime_dir="${KAPSL_TENSORRT_RUNTIME_DIR:-$repo_root/tensorrt-runtime-libs}"
tensorrt_license_dir="${KAPSL_TENSORRT_LICENSE_DIR:-$repo_root/tensorrt-license-files}"
nvidia_license="${KAPSL_NVIDIA_LICENSE_FILE:-$cuda_runtime_root/NVIDIA-CONTAINER-LICENSE}"

for command_name in git python3 sha256sum tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to package ORT accelerator adapters." >&2
    exit 1
  fi
done
"$verifier" "$integrations_dir" "$KAPSL_ORT_INTEGRATIONS_REF"
if [ ! -x "$build_script" ]; then
  echo "Certified ORT accelerator packaging entrypoint is missing or not executable: $build_script" >&2
  exit 1
fi
for required in \
  "$cuda_runtime_root" \
  "$tensorrt_runtime_dir" \
  "$tensorrt_license_dir" \
  "$nvidia_license"; do
  if [ ! -e "$required" ]; then
    echo "Missing ORT accelerator packaging input: $required" >&2
    exit 1
  fi
done

outputs=()
for profile in cuda12 tensorrt10; do
  archive_name="kapsl-backend-onnx-${profile}-${KAPSL_VERSION}-linux-x86_64.tar.gz"
  for suffix in "" .manifest.json .sha256 .sig; do
    output="$output_dir/${archive_name}${suffix}"
    outputs+=("$output")
    if [ -e "$output" ]; then
      echo "Refusing to overwrite an existing ORT accelerator release output: $output" >&2
      exit 1
    fi
  done
done

mkdir -p "$output_dir"
KAPSL_ORT_PACK_OUTPUT_DIR="$output_dir" \
KAPSL_ORT_PACK_BUILD_DIR="${KAPSL_ORT_PACK_BUILD_DIR:-${RUNNER_TEMP:-/tmp}/kapsl-ort-accelerator-pack}" \
KAPSL_CUDA_RUNTIME_ROOT="$cuda_runtime_root" \
KAPSL_TENSORRT_RUNTIME_DIR="$tensorrt_runtime_dir" \
KAPSL_TENSORRT_LICENSE_DIR="$tensorrt_license_dir" \
KAPSL_NVIDIA_LICENSE_FILE="$nvidia_license" \
KAPSL_VERSION="$KAPSL_VERSION" \
  "$build_script"
"$verifier" "$integrations_dir" "$KAPSL_ORT_INTEGRATIONS_REF"

for profile in cuda12 tensorrt10; do
  archive_name="kapsl-backend-onnx-${profile}-${KAPSL_VERSION}-linux-x86_64.tar.gz"
  archive="$output_dir/$archive_name"
  manifest="${archive}.manifest.json"
  checksum="${archive}.sha256"
  signature="${archive}.sig"
  for required in "$archive" "$manifest" "$checksum"; do
    if [ ! -s "$required" ]; then
      echo "Certified ORT accelerator packaging did not produce $required" >&2
      exit 1
    fi
  done
  if [ -e "$signature" ]; then
    echo "The integrations handoff must not sign release artifacts; the engine index publisher owns the release key." >&2
    exit 1
  fi
  (cd "$output_dir" && sha256sum --check "${archive_name}.sha256")
done

python3 - \
  "$output_dir" \
  "$KAPSL_VERSION" \
  "$KAPSL_ORT_INTEGRATIONS_REF" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
import tarfile

output_dir = pathlib.Path(sys.argv[1])
version = sys.argv[2]
source_ref = sys.argv[3]
profiles = {
    "cuda12": {
        "accelerator": "cuda",
        "provider": "libonnxruntime_providers_cuda.so",
        "forbidden": "libonnxruntime_providers_tensorrt.so",
        "extra": set(),
    },
    "tensorrt10": {
        "accelerator": "tensorrt",
        "provider": "libonnxruntime_providers_tensorrt.so",
        "forbidden": None,
        "extra": {"libnvinfer.so.10", "libnvonnxparser.so.10"},
    },
}
runtime_distribution_url = (
    "https://github.com/microsoft/onnxruntime/releases/download/"
    "v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz"
)
runtime_distribution_sha256 = (
    "2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53"
)
common_libraries = {
    "libkapsl_backend_ort.so",
    "libonnxruntime.so.1",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
    "libcublas.so.12",
    "libcudart.so.12",
    "libcudnn.so.9",
}
maximum_glibc = (2, 35)


def parse_glibc(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value):
        raise SystemExit(f"{label} is not a GLIBC version")
    return tuple(int(component) for component in value.split("."))


for profile, contract in profiles.items():
    name = f"kapsl-backend-onnx-{profile}-{version}-linux-x86_64.tar.gz"
    archive_path = output_dir / name
    manifest_path = output_dir / f"{name}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "backend": "onnx",
        "profile": profile,
        "runtime_abi": 1,
        "adapter_abi": "kapsl-backend-v1",
        "compatible_kapsl": f"={version}",
        "platform": "linux-x86_64",
        "architecture": "x86_64",
        "accelerator_profile": contract["accelerator"],
        "minimum_cuda": "12.0",
        "minimum_driver": "560.28.03",
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
            raise SystemExit(f"{manifest_path}: handoff template contains {forbidden}")

    member_digests = {}
    metadata_payloads = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"{archive_path}: unsafe member {member.name}")
            if not member.isfile():
                continue
            if member.name in member_digests:
                raise SystemExit(f"{archive_path}: duplicate member {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit(f"{archive_path}: cannot read {member.name}")
            digest = hashlib.sha256()
            metadata = bytearray()
            retain_metadata = member.name in {"backend-pack.json", "provenance.json"}
            if retain_metadata and member.size > 16 * 1024 * 1024:
                raise SystemExit(f"{archive_path}: oversized metadata {member.name}")
            observed_size = 0
            while block := stream.read(1024 * 1024):
                observed_size += len(block)
                digest.update(block)
                if retain_metadata:
                    metadata.extend(block)
            if observed_size != member.size:
                raise SystemExit(f"{archive_path}: truncated member {member.name}")
            member_digests[member.name] = digest.hexdigest()
            if retain_metadata:
                metadata_payloads[member.name] = bytes(metadata)

    required = (
        common_libraries
        | contract["extra"]
        | {contract["provider"], "backend-pack.json", "provenance.json"}
    )
    missing = sorted(required - set(member_digests))
    if missing:
        raise SystemExit(f"{archive_path}: missing {', '.join(missing)}")
    if contract["forbidden"] in member_digests:
        raise SystemExit(f"{archive_path}: CUDA profile contains TensorRT provider")
    driver_libraries = sorted(
        name
        for name in member_digests
        if re.fullmatch(r"libcuda\.so(?:\..*)?|libnvidia-[^/]+\.so(?:\..*)?", name)
    )
    if driver_libraries:
        raise SystemExit(
            f"{archive_path}: pack contains host driver libraries {driver_libraries}"
        )

    payload = json.loads(metadata_payloads["backend-pack.json"])
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
    if not isinstance(signed_files, dict) or set(signed_files) != set(member_digests):
        raise SystemExit(f"{manifest_path}: signed files do not exactly match archive")
    for filename, digest in member_digests.items():
        if signed_files[filename] != digest:
            raise SystemExit(f"{manifest_path}: digest mismatch for {filename}")

    provenance = json.loads(metadata_payloads["provenance.json"])
    if provenance.get("source_repository") != "https://github.com/kapsl-runtime/kapsl-integrations":
        raise SystemExit(f"{archive_path}: unexpected source repository")
    if provenance.get("source_commit") != source_ref:
        raise SystemExit(f"{archive_path}: source commit does not match checkout")
    if provenance.get("adapter", {}).get("adapter_abi") != "kapsl-backend-v1":
        raise SystemExit(f"{archive_path}: provenance adapter ABI is invalid")
    if provenance.get("profile", {}).get("pack") != profile:
        raise SystemExit(f"{archive_path}: provenance profile is invalid")
    runtime = provenance.get("onnx_runtime", {})
    if runtime.get("version") != "1.23.2":
        raise SystemExit(f"{archive_path}: provenance runtime version is invalid")
    if runtime.get("distribution_url") != runtime_distribution_url:
        raise SystemExit(f"{archive_path}: provenance runtime URL is not approved")
    if runtime.get("distribution_sha256") != runtime_distribution_sha256:
        raise SystemExit(f"{archive_path}: provenance runtime digest is not approved")
    libraries = provenance.get("libraries")
    if not isinstance(libraries, dict) or set(libraries) != {
        name for name in member_digests if ".so" in name
    }:
        raise SystemExit(f"{archive_path}: provenance library closure is incomplete")
    for filename, metadata in libraries.items():
        if metadata.get("sha256") != signed_files[filename]:
            raise SystemExit(f"{archive_path}: provenance digest mismatch for {filename}")
        if metadata.get("runpath") != "$ORIGIN":
            raise SystemExit(f"{archive_path}: {filename} is not pack-local")
        if parse_glibc(metadata.get("maximum_required_glibc"), filename) > maximum_glibc:
            raise SystemExit(f"{archive_path}: {filename} exceeds GLIBC 2.35")
        for dependency in metadata.get("needed_libraries", []):
            host_library = dependency in {
                "ld-linux-x86-64.so.2",
                "libatomic.so.1",
                "libc.so.6",
                "libdl.so.2",
                "libgcc_s.so.1",
                "libgomp.so.1",
                "libm.so.6",
                "libpthread.so.0",
                "libresolv.so.2",
                "librt.so.1",
                "libstdc++.so.6",
                "libutil.so.1",
            } or re.fullmatch(
                r"libcuda\.so(?:\..*)?|libnvidia-[^/]+\.so(?:\..*)?",
                dependency,
            )
            if dependency not in libraries and not host_library:
                raise SystemExit(
                    f"{archive_path}: {filename} requires unpackaged {dependency}"
                )

print(f"Accepted certified ORT accelerator adapters from {source_ref}")
PY
