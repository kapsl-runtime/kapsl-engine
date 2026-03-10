#!/usr/bin/env python3
"""
Test Script: HTTP Inference Validation

Tests the deployed kapsl-runtime server using the new HTTP API endpoint.
"""

import base64
import json
import os
import sys
import time

import numpy as np
import requests


def get_auth_headers():
    token = (
        os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
        or os.getenv("KAPSL_DESKTOP_API_TOKEN")
        or os.getenv("KAPSL_API_TOKEN")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def test_http_inference(model_id=0):
    """Test inference via POST /api/models/:id/infer"""
    print(f"🧪 Testing HTTP Inference for Model {model_id}")
    print("-" * 50)

    # Create test input (MNIST-like: 1x1x28x28)
    # Note: Use small data because JSON encoding of binary data is inefficient
    input_shape = [1, 1, 28, 28]
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Prepare InferenceRequest JSON
    # BinaryTensorPacket { shape: Vec<i64>, dtype: String, data: Vec<u8> }
    # InferenceRequest { input: BinaryTensorPacket, session_id: Option<String> }
    request_payload = {
        "input": {
            "shape": input_shape,
            "dtype": "float32",
            "data_base64": base64.b64encode(input_data.tobytes()).decode("ascii"),
        },
        "session_id": None,
    }

    url = f"http://localhost:9095/api/models/{model_id}/infer"

    try:
        start = time.time()
        response = requests.post(
            url,
            json=request_payload,
            headers=get_auth_headers(),
            timeout=10,
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Received response in {elapsed * 1000:.2f}ms")
            print(f"📊 Output Info:")
            print(f"   Shape: {result.get('shape')}")
            print(f"   Dtype: {result.get('dtype')}")
            print(f"   Data size: {len(result.get('data', []))} bytes")
            return True
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_api_status():
    """Verify basic API connectivity"""
    print("🧪 Verifying API Status")
    print("-" * 50)
    try:
        resp = requests.get(
            "http://localhost:9095/api/models",
            headers=get_auth_headers(),
        )
        if resp.status_code == 200:
            models = resp.json()
            print(f"✅ Found {len(models)} models")
            for m in models:
                print(f"   - ID {m['id']}: {m['name']} ({m['status']})")
            return True
        else:
            print(f"❌ Failed to list models: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ API unreachable: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  kapsl-runtime HTTP Inference Test")
    print("=" * 60 + "\n")

    if not test_api_status():
        print("\n💡 Make sure the server is running with the new HTTP infer endpoint!")
        sys.exit(1)

    if test_http_inference(0):
        print("\n🎉 HTTP Inference Test Passed!")
    else:
        print("\n⚠️  HTTP Inference Test Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
