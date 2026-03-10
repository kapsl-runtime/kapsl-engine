#!/usr/bin/env python3
"""Quick socket test"""
import socket
import struct
import json
import numpy as np

socket_path = "/tmp/kapsl.sock"

# Create dummy input
dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

# Prepare request
request = {
    "model_id": 0,
    "input": {
        "shape": list(dummy_input.shape),
        "dtype": "float32",
        "data": dummy_input.tobytes().hex()
    }
}

print("Connecting to", socket_path)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(socket_path)

request_json = json.dumps(request)
request_bytes = request_json.encode('utf-8')

print(f"Sending {len(request_bytes)} bytes...")
# Send length prefix + request
sock.sendall(struct.pack('<I', len(request_bytes)))
sock.sendall(request_bytes)

print("Reading response...")
# Read response
length_bytes = sock.recv(4)
if len(length_bytes) == 4:
    response_length = struct.unpack('<I', length_bytes)[0]
    print(f"Response length: {response_length}")
    
    response_data = b''
    while len(response_data) < response_length:
        chunk = sock.recv(min(4096, response_length - len(response_data)))
        if not chunk:
            break
        response_data += chunk
    
    print(f"Received {len(response_data)} bytes")
    response = json.loads(response_data)
    print("Response:", response.keys() if isinstance(response, dict) else "not a dict")
else:
    print("Failed to read length")

sock.close()
print("✓ Success!")
