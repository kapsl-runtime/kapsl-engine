#!/usr/bin/env python3
"""
Direct Python bindings benchmark via Unix socket
Uses the Rust PyO3 bindings for zero-overhead FFI
"""
import time
import numpy as np
import onnxruntime as ort
from kapsl_runtime import KapslClient

NUM_REQUESTS = 100

def benchmark_traditional():
    """Traditional ONNX Runtime"""
    session = ort.InferenceSession("scripts/mnist.onnx")
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    start = time.time()
    for _ in range(NUM_REQUESTS):
        session.run(None, {input_name: dummy_input})
    end = time.time()
    
    total = end - start
    return {
        "total_sec": total,
        "avg_ms": (total / NUM_REQUESTS) * 1000,
        "throughput": NUM_REQUESTS / total,
    }

def benchmark_kapsl_socket():
    """kapsl-runtime via Unix socket (PyO3 bindings)"""
    client = KapslClient("/tmp/kapsl.sock")
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    shape = [1, 1, 28, 28]
    dtype = "float32"
    data = dummy_input.tobytes()
    
    # Warmup
    try:
        client.infer(shape, dtype, list(data))
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return None
    
    start = time.time()
    for _ in range(NUM_REQUESTS):
        client.infer(shape, dtype, list(data))
    end = time.time()
    
    total = end - start
    return {
        "total_sec": total,
        "avg_ms": (total / NUM_REQUESTS) * 1000,
        "throughput": NUM_REQUESTS / total,
    }

if __name__ == "__main__":
    print(f"🚀 PyO3 Direct Binding Benchmark ({NUM_REQUESTS} requests)\\n")
    
    print("1️⃣  Traditional ONNX Runtime...")
    trad = benchmark_traditional()
    print(f"   ✓ {trad['total_sec']:.3f}s total, {trad['throughput']:.1f} req/sec\\n")
    
    print("2️⃣  kapsl-runtime (PyO3 + Unix Socket)...")
    kapsl = benchmark_kapsl_socket()
    
    if kapsl:
        print(f"   ✓ {kapsl['total_sec']:.3f}s total, {kapsl['throughput']:.1f} req/sec\\n")
        
        print("="*60)
        print(f"{'Metric':<30} {'Traditional':<15} {'kapsl-runtime':<15}")
        print("-"*60)
        print(f"{'Total Time (sec)':<30} {trad['total_sec']:<15.3f} {kapsl['total_sec']:<15.3f}")
        print(f"{'Avg Latency (ms)':<30} {trad['avg_ms']:<15.3f} {kapsl['avg_ms']:<15.3f}")
        print(f"{'Throughput (req/sec)':<30} {trad['throughput']:<15.1f} {kapsl['throughput']:<15.1f}")
        
        overhead_ms = kapsl['avg_ms'] - trad['avg_ms']
        slowdown = kapsl['avg_ms'] / trad['avg_ms']
        
        print("="*60)
        print(f"\\n💡 Socket IPC overhead: +{overhead_ms:.3f}ms per request")
        print(f"   Slowdown factor: {slowdown:.1f}x")
        print("\\n✅ This is the fastest Python can call kapsl-runtime!")
        print("   (Unix socket + Rust FFI, no HTTP/JSON overhead)")
        print("="*60)
    else:
        print("   ✗ Failed - check if kapsl-runtime is running")
