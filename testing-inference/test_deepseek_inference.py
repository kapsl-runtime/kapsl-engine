#!/usr/bin/env python3
"""
Test Script: Deepseek Inference Validation

Tests the deepseek language model deployed on kapsl-runtime using the HTTP API.
Based on the deepseek_client.rs example.
"""

import base64
import json
import os
import struct
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


def test_deepseek_inference(model_id=0):
    """Test deepseek inference via POST /api/models/:id/infer"""
    print(f"🧪 Testing Deepseek Inference for Model {model_id}")
    print("-" * 50)

    #  Example prompt: "Hello world" tokens
    # Shape: [batch_size, sequence_length] = [1, 2]
    input_ids = [15496, 1234]  # Example token IDs
    batch_size = 1
    seq_length = len(input_ids)

    print(f"📝 Input tokens: {input_ids}")
    print(f"📝 Shape: [{batch_size}, {seq_length}]")

    # Convert to bytes (int64, little-endian)
    data_bytes = bytearray()
    for token_id in input_ids:
        data_bytes.extend(struct.pack("<q", token_id))  # '<q' = little-endian int64

    # Create BinaryTensorPacket
    request_payload = {
        "input": {
            "shape": [batch_size, seq_length],
            "dtype": "int64",
            "data_base64": base64.b64encode(bytes(data_bytes)).decode("ascii"),
        },
        "session_id": None,
    }

    # Print payload info
    payload_json = json.dumps(request_payload)
    print(f"📦 Payload size: {len(payload_json) / 1024:.2f} KB")

    url = f"http://localhost:9095/api/models/{model_id}/infer"

    try:
        start = time.time()
        response = requests.post(
            url,
            json=request_payload,
            headers=get_auth_headers(),
            timeout=30,
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Received response in {elapsed * 1000:.2f}ms")
            print(f"📊 Output Info:")
            print(f"   Shape: {result.get('shape')}")
            print(f"   Dtype: {result.get('dtype')}")
            print(f"   Data size: {len(result.get('data', []))} bytes")

            # Parse output logits if dtype is float32
            if result.get("dtype") == "float32":
                shape = result["shape"]
                data_bytes = bytes(result["data"])

                # Calculate number of floats
                num_floats = len(data_bytes) // 4
                print(f"   Number of output values: {num_floats}")

                # Parse first few floats
                floats = []
                for i in range(min(10, num_floats)):
                    float_bytes = data_bytes[i * 4 : (i + 1) * 4]
                    float_val = struct.unpack("<f", float_bytes)[0]
                    floats.append(float_val)

                print(f"   First 10 outputs: {floats[:10]}")

                # If output shape is [batch, seq, vocab_size], find predicted token
                if len(shape) == 3:
                    vocab_size = shape[-1]
                    # Get logits for last token
                    last_token_start = (seq_length - 1) * vocab_size
                    last_token_logits = []
                    for i in range(vocab_size):
                        idx = last_token_start + i
                        if idx < num_floats:
                            float_bytes = data_bytes[idx * 4 : (idx + 1) * 4]
                            last_token_logits.append(
                                struct.unpack("<f", float_bytes)[0]
                            )

                    if last_token_logits:
                        predicted_token = np.argmax(last_token_logits)
                        print(f"   Predicted next token ID: {predicted_token}")

            return True
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
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
            return True, models
        else:
            print(f"❌ Failed to list models: {resp.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ API unreachable: {e}")
        return False, []


def main():
    print("\n" + "=" * 60)
    print("  Deepseek Inference Test")
    print("=" * 60 + "\n")

    success, models = test_api_status()
    if not success:
        print("\n💡 Make sure kapsl-runtime is running!")
        sys.exit(1)

    # Find deepseek model
    deepseek_id = None
    for model in models:
        if "deepseek" in model["name"].lower():
            deepseek_id = model["id"]
            break

    if deepseek_id is None:
        print("❌ No deepseek model found!")
        print("Available models:", [m["name"] for m in models])
        sys.exit(1)

    print(f"\n🎯 Using model ID: {deepseek_id}\n")

    if test_deepseek_inference(deepseek_id):
        print("\n🎉 Deepseek Inference Test Passed!")
    else:
        print("\n⚠️  Deepseek Inference Test Failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
