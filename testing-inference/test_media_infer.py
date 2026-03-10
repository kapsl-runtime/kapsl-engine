#!/usr/bin/env python3
"""
End-to-end media inference test for /api/models/:id/infer.

Covers:
- image media payload (base64)
- optional video media payload (base64)
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests

DEFAULT_BASE_URL = "http://localhost:9095"

# 1x1 red PNG
RED_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEklEQVR4nGP4z8AARMAgYQAAAwAB"
    "9HFkzgAAAABJRU5ErkJggg=="
)


def get_auth_headers() -> Dict[str, str]:
    token = (
        os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def http_get(url: str, timeout: int = 30) -> requests.Response:
    return requests.get(url, headers=get_auth_headers(), timeout=timeout)


def http_post(
    url: str, payload: Dict[str, Any], timeout: int = 60
) -> requests.Response:
    return requests.post(url, json=payload, headers=get_auth_headers(), timeout=timeout)


def fail(msg: str) -> None:
    print(msg)
    sys.exit(1)


def ensure_ok(resp: requests.Response, step: str) -> Dict[str, Any]:
    if 200 <= resp.status_code < 300:
        return resp.json()
    body = resp.text.strip()
    fail(f"{step} failed ({resp.status_code}): {body}")
    return {}


def detect_model(base_url: str, explicit_model_id: Optional[int]) -> int:
    resp = http_get(f"{base_url}/api/models")
    models = ensure_ok(resp, "List models")
    if not isinstance(models, list) or not models:
        fail("No models are running. Start an ONNX model first.")

    if explicit_model_id is not None:
        for model in models:
            if model.get("id") == explicit_model_id:
                framework = str(model.get("framework", "")).lower()
                if framework != "onnx":
                    fail(
                        f"Model {explicit_model_id} framework is '{framework}', expected 'onnx' for media tensor tests."
                    )
                return explicit_model_id
        fail(f"Model {explicit_model_id} not found.")

    for model in models:
        if str(model.get("framework", "")).lower() == "onnx":
            return int(model["id"])

    fail("No ONNX model found. Media test currently targets ONNX tensor models.")
    return 0


def print_tensor_summary(tag: str, result: Dict[str, Any]) -> None:
    shape = result.get("shape")
    dtype = result.get("dtype")
    data = result.get("data", [])
    size = len(data) if isinstance(data, list) else -1
    print(f"{tag}: dtype={dtype}, shape={shape}, data_len={size}")


def run_image_test(base_url: str, model_id: int, print_response: bool) -> None:
    payload = {
        "media": {
            "kind": "image",
            "base64": RED_PNG_BASE64,
        },
        "tensor_options": {
            "target_width": 224,
            "target_height": 224,
            "layout": "nchw",
            "channels": "rgb",
            "dtype": "float32",
            "normalize": "zero_to_one",
        },
    }
    resp = http_post(f"{base_url}/api/models/{model_id}/infer", payload, timeout=90)
    if resp.status_code >= 400:
        msg = resp.text.strip()
        if "Inline media preprocessing is disabled" in msg:
            fail(
                "Image test failed because inline media preprocessing is disabled in runtime. "
                "Unset KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS (legacy KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS also works) or send preprocessed tensors."
            )
        fail(f"Image infer failed ({resp.status_code}): {msg}")
    result = resp.json()
    print_tensor_summary("Image infer", result)
    if print_response:
        print(json.dumps(result, indent=2))


def generate_test_video_bytes(width: int = 224, height: int = 224) -> Optional[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    with tempfile.TemporaryDirectory(prefix="kapsl-media-test-") as temp_dir:
        video_path = Path(temp_dir) / "sample.mp4"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s={width}x{height}:d=1",
            "-r",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not video_path.exists():
            stderr = proc.stderr.strip()
            fail(f"Failed to generate local test video with ffmpeg: {stderr}")
        return video_path.read_bytes()


def run_video_test(base_url: str, model_id: int, print_response: bool) -> None:
    video_bytes = generate_test_video_bytes()
    if video_bytes is None:
        print("Video infer: skipped (ffmpeg not found in PATH).")
        return

    payload = {
        "media": {
            "kind": "video",
            "base64": base64.b64encode(video_bytes).decode("ascii"),
        },
        "tensor_options": {
            "target_width": 224,
            "target_height": 224,
            "layout": "nchw",
            "channels": "rgb",
            "dtype": "float32",
            "normalize": "zero_to_one",
            "frame_count": 1,
            "frame_stride": 1,
        },
    }
    resp = http_post(f"{base_url}/api/models/{model_id}/infer", payload, timeout=120)
    if resp.status_code >= 400:
        msg = resp.text.strip()
        if "ffmpeg is required for video infer payloads" in msg:
            fail(
                "Video infer failed: runtime ffmpeg dependency is missing on server host."
            )
        fail(f"Video infer failed ({resp.status_code}): {msg}")
    result = resp.json()
    print_tensor_summary("Video infer", result)
    if print_response:
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Media inference E2E test")
    parser.add_argument(
        "--base-url",
        default=os.getenv("KAPSL_BASE_URL")
        or os.getenv("KAPSL_BASE_URL")
        or DEFAULT_BASE_URL,
        help="Runtime base URL (default: http://localhost:9095)",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=None,
        help="ONNX model ID to test. If omitted, first ONNX model is used.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video payload test",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print full infer response JSON",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model_id = detect_model(base_url, args.model_id)
    print(f"Using model_id={model_id} for media inference tests.")

    run_image_test(base_url, model_id, args.print_response)
    if not args.skip_video:
        run_video_test(base_url, model_id, args.print_response)

    print("Media inference tests completed.")


if __name__ == "__main__":
    main()
