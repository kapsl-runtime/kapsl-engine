#!/usr/bin/env python3
"""Quick 3-iteration benchmark using HTTP instead of sockets"""
import time
import json
import numpy as np
import onnxruntime as ort
import requests

def benchmark_onnx(model_path, iterations=3):
    """Benchmark ONNX Runtime"""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    # Warmup
    session.run(None, {input_name: dummy_input})
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    end = time.time()
    
    avg_ms = ((end - start) / iterations) * 1000
    return avg_ms

def benchmark_kapsl_http(iterations=3):
    """Benchmark kapsl-runtime via HTTP"""
    url = "http://localhost:8080/api/models/0/infer"
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    payload = {
        "shape": [1, 1, 28, 28],
        "dtype": "float32",
        "data": dummy_input.tobytes().hex()
    }
    
    # Warmup
    requests.post(url, json=payload)
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.time()
        response = requests.post(url, json=payload)
        end = time.time()
        latencies.append((end - start) * 1000)
        
    avg_ms = np.mean(latencies)
    return avg_ms, latencies

if __name__ == "__main__":
    print("🚀 Quick 3-Iteration Benchmark\\n")
    
    # Traditional ONNX
    print("1️⃣  Traditional ONNX Runtime...", end=" ", flush=True)
    onnx_ms = benchmark_onnx("scripts/mnist.onnx")
    print(f"{onnx_ms:.3f}ms")
    
    # kapsl-runtime HTTP
    print("2️⃣  kapsl-runtime (HTTP)...", end=" ", flush=True)
    kapsl_ms, latencies = benchmark_kapsl_http()
    print(f"{kapsl_ms:.3f}ms")
    
    # Results
    print(f"\\n{'='*50}")
    print(f"Traditional ONNX:  {onnx_ms:.3f}ms avg")
    print(f"kapsl-runtime:     {kapsl_ms:.3f}ms avg")
    print(f"  Latencies:       {[f'{l:.1f}ms' for l in latencies]}")
    print(f"\\nOverhead:          +{(kapsl_ms - onnx_ms):.3f}ms")
    print(f"Speedup:           {(kapsl_ms / onnx_ms):.2f}x slower (expected due to HTTP)")
    print(f"{'='*50}")
