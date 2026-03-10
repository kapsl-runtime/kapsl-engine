#!/usr/bin/env python3
"""Test SHM client with dynamic name"""
from kapsl_runtime import KapslShmClient
import numpy as np
import time
import subprocess

# Get the PID of kapsl
result = subprocess.run(
    ["pgrep", "-f", "kapsl.*shm"],
    capture_output=True,
    text=True
)

if not result.stdout.strip():
    print("❌ kapsl not running with shm transport")
    exit(1)

pid = result.stdout.strip().split()[0]
shm_name = f"/kapsl_shm_{pid}"

print(f"Connecting to shared memory: {shm_name}")

try:
    client = KapslShmClient(shm_name)
    print("✓ Connected to shared memory")
    
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    shape = [1, 1, 28, 28]
    dtype = "float32"
    data = list(dummy_input.tobytes())
    
    print(f"Sending inference request...")
    start = time.time()
    result = client.infer(shape, dtype, data)
    elapsed = (time.time() - start) * 1000
    
    print(f"✓ Success! Latency: {elapsed:.2f}ms")
    print(f"Result size: {len(result)} bytes")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
