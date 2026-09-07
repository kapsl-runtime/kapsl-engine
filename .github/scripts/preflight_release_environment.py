#!/usr/bin/env python3
"""Fail early on release configuration errors without provisioning resources."""

import argparse
import base64
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile


class PreflightError(RuntimeError):
    pass


def require(environ, names):
    for name in names:
        if not environ.get(name, "").strip():
            raise PreflightError(f"Required release setting {name} is empty.")


def validate_gpu(environ):
    require(environ, (
        "GCP_GPU_PROJECT_ID", "GCP_GPU_ZONE", "GCP_GPU_RUNNER_IMAGE",
        "GCP_WORKLOAD_IDENTITY_PROVIDER", "GCP_GPU_PROVISIONER_SERVICE_ACCOUNT",
        "GPU_RUNNER_GITHUB_APP_ID", "KAPSL_VLLM_SDK_REF",
    ))
    if not environ.get("GPU_RUNNER_GITHUB_APP_PRIVATE_KEY", "").strip():
        raise PreflightError(
            "GPU_RUNNER_GITHUB_APP_PRIVATE_KEY is empty in gcp-gpu-conformance; "
            "check the environment secret and the caller's secrets: inherit."
        )
    for name, default in (("GPU_RUNNER_GITHUB_APP_ID", ""), ("GCP_GPU_RUNNER_GROUP_ID", "1")):
        if not re.fullmatch(r"[1-9][0-9]*", environ.get(name) or default):
            raise PreflightError(f"{name} must be a positive integer.")
    if (environ.get("GCP_GPU_EXTERNAL_IP") or "true") not in ("true", "false"):
        raise PreflightError("GCP_GPU_EXTERNAL_IP must be true or false.")
    if not re.fullmatch(r"[0-9a-f]{40}", environ["KAPSL_VLLM_SDK_REF"]):
        raise PreflightError("KAPSL_VLLM_SDK_REF must be an exact lowercase 40-hex commit.")
    image = environ["GCP_GPU_RUNNER_IMAGE"].removeprefix(
        "https://www.googleapis.com/compute/v1/"
    )
    if not re.fullmatch(r"projects/[a-z][a-z0-9-]+/global/images/[a-z][a-z0-9-]+", image):
        raise PreflightError("GCP_GPU_RUNNER_IMAGE must identify an immutable image, not a family.")


def validate_signing(environ):
    require(environ, (
        "KAPSL_BACKEND_SIGNING_KEY_B64", "KAPSL_BACKEND_PUBLIC_KEYS",
        "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
    ))
    spec = importlib.util.spec_from_file_location(
        "release_index", Path(__file__).with_name("generate-backend-index.py")
    )
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    try:
        private_key = base64.b64decode(
            "".join(environ["KAPSL_BACKEND_SIGNING_KEY_B64"].split()), validate=True
        )
        trusted = generator.expected_public_keys([environ["KAPSL_BACKEND_PUBLIC_KEYS"]])
        with tempfile.TemporaryDirectory(prefix="kapsl-signing-preflight-") as directory:
            key_file = Path(directory) / "signing.pem"
            key_file.write_bytes(private_key)
            key_file.chmod(0o600)
            if generator.signing_public_key(key_file) not in trusted:
                raise PreflightError("Backend signing key does not match runtime trust.")
            generator.sign(key_file, b"kapsl-release-preflight-v1\0")
    except (ValueError, subprocess.CalledProcessError, SystemExit) as error:
        # Never echo OpenSSL output or decoded private-key material.
        raise PreflightError("Backend signing configuration is invalid.") from error


def validate_sdk(environ):
    require(environ, ("KAPSL_VLLM_SDK_REF", "GH_TOKEN"))
    ref = environ["KAPSL_VLLM_SDK_REF"]
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise PreflightError("KAPSL_VLLM_SDK_REF must be an exact lowercase 40-hex commit.")
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/kapsl-runtime/kapsl-sdk/commits/{ref}", "--jq", ".sha"],
            env=dict(environ), check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreflightError("Could not verify the certified SDK commit before builds.") from error
    if result.stdout.strip() != ref:
        raise PreflightError("Certified SDK commit did not resolve to the exact requested revision.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--signing", action="store_true")
    parser.add_argument("--sdk", action="store_true")
    args = parser.parse_args()
    if not args.gpu and not args.signing and not args.sdk:
        parser.error("select --gpu, --signing or --sdk")
    try:
        if args.gpu:
            validate_gpu(os.environ)
        if args.signing:
            validate_signing(os.environ)
        if args.sdk:
            validate_sdk(os.environ)
    except PreflightError as error:
        raise SystemExit(str(error)) from None
    print("Release environment configuration passed (no resources provisioned).")


if __name__ == "__main__":
    main()
