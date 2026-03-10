#!/usr/bin/env python3
"""
Test with actual valid tensor data using kapsl_runtime module
"""
from kapsl_runtime import KapslClient
import numpy as np
import time

client = KapslClient("/tmp/kapsl.sock")
dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

shape = [1, 1, 28, 28]
dtype = "float32"
data = list(dummy_input.tobytes())

print(f"Sending inference request...")
print(f"Shape: {shape}")
print(f"Dtype: {dtype}")
print(f"Data size: {len(data)} bytes")

start = time.time()
try:
    result = client.infer(shape, dtype, data)
    elapsed = (time.time() - start) * 1000
    print(f"✓ Success! Latency: {elapsed:.2f}ms")
    print(f"Result size: {len(result)} bytes")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
