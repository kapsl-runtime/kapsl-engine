#!/usr/bin/env python3
"""
Hybrid Architecture Benchmark
Compares Direct ONNX Runtime vs kapsl-runtime Hybrid Client (Socket + SHM)
"""
import time
import numpy as np
import onnxruntime as ort
from kapsl_runtime import KapslHybridClient
import subprocess
import os

NUM_REQUESTS = 100

# Get the PID of kapsl to determine SHM name
# We assume kapsl is running in hybrid mode
result = subprocess.run(
    ["pgrep", "-f", "kapsl.*hybrid"],
    capture_output=True,
    text=True
)

if not result.stdout.strip():
    print("❌ kapsl not running with hybrid transport")
    print("Please run: ./target/release/kapsl --model scripts/mnist_opt_basic.aimod --transport hybrid")
    exit(1)

pid = result.stdout.strip().split()[0]
shm_name = f"/kapsl_shm_{pid}"
socket_path = "/tmp/kapsl.sock"
shm_size = 1024 * 1024 * 1024  # 1GB

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

def benchmark_kapsl_hybrid():
    """kapsl-runtime via Hybrid Client"""
    try:
        # Note: shm_size is not needed for connect in the updated client, 
        # but the constructor might still accept it if we didn't update the python signature?
        # Let's check the rust code... we updated it to `new(socket_path, shm_name)`.
        client = KapslHybridClient(socket_path, shm_name)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return None

    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    shape = [1, 1, 28, 28]
    dtype = "float32"
    data = list(dummy_input.tobytes())
    
    # Warmup
    try:
        client.infer(0, shape, dtype, data)
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return None
    
    # Collect individual latencies
    latencies = []
    start_total = time.perf_counter()
    for _ in range(NUM_REQUESTS):
        req_start = time.perf_counter()
        client.infer(0, shape, dtype, data)
        latencies.append((time.perf_counter() - req_start) * 1000)
    end_total = time.perf_counter()
    
    total = end_total - start_total
    
    return {
        "total_sec": total,
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "throughput": NUM_REQUESTS / total,
    }

if __name__ == "__main__":
    print(f"🚀 Hybrid Architecture Benchmark ({NUM_REQUESTS} requests)\n")
    print(f"Configuration:")
    print(f"  Socket: {socket_path}")
    print(f"  SHM:    {shm_name}")
    print("-" * 40)
    
    print("1️⃣  Traditional ONNX Runtime...")
    trad = benchmark_traditional()
    print(f"   ✓ {trad['total_sec']:.3f}s total, {trad['throughput']:.1f} req/sec\n")
    
    print("2️⃣  kapsl-runtime (Hybrid)...")
    kapsl = benchmark_kapsl_hybrid()
    
    if kapsl:
        print(f"   ✓ {kapsl['total_sec']:.3f}s total, {kapsl['throughput']:.1f} req/sec\n")
        
        print("="*60)
        print(f"{'Metric':<30} {'Traditional':<15} {'kapsl (Hybrid)':<15}")
        print("-"*60)
        print(f"{'Total Time (sec)':<30} {trad['total_sec']:<15.3f} {kapsl['total_sec']:<15.3f}")
        print(f"{'Avg Latency (ms)':<30} {trad['avg_ms']:<15.3f} {kapsl['avg_ms']:<15.3f}")
        print(f"{'Min Latency (ms)':<30} {'-':<15} {kapsl['min_ms']:<15.3f}")
        print(f"{'Max Latency (ms)':<30} {'-':<15} {kapsl['max_ms']:<15.3f}")
        print(f"{'Throughput (req/sec)':<30} {trad['throughput']:<15.1f} {kapsl['throughput']:<15.1f}")
        
        overhead_ms = kapsl['avg_ms'] - trad['avg_ms']
        slowdown = kapsl['avg_ms'] / trad['avg_ms']
        
        print("="*60)
        print(f"\n💡 Hybrid IPC overhead: +{overhead_ms:.3f}ms per request")
        print(f"   Slowdown factor: {slowdown:.1f}x")
        print("="*60)
    else:
        print("   ✗ Failed - check if kapsl-runtime is running in hybrid mode")
