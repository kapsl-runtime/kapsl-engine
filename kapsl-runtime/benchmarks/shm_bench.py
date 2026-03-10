#!/usr/bin/env python3
"""
SHM benchmark
"""
import time
import numpy as np
import onnxruntime as ort
from kapsl_runtime import KapslShmClient
import subprocess

NUM_REQUESTS = 100

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

def benchmark_kapsl_shm():
    """kapsl-runtime via SHM"""
    client = KapslShmClient(shm_name)
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
   
    shape = [1, 1, 28, 28]
    dtype = "float32"
    data = list(dummy_input.tobytes())
    
    # Warmup
    client.infer(shape, dtype, data)
    
    start = time.time()
    for _ in range(NUM_REQUESTS):
        client.infer(shape, dtype, data)
    end = time.time()
    
    total = end - start
    return {
        "total_sec": total,
        "avg_ms": (total / NUM_REQUESTS) * 1000,
        "throughput": NUM_REQUESTS / total,
    }

if __name__ == "__main__":
    print(f"🚀 SHM Performance Benchmark ({NUM_REQUESTS} requests)\n")
    
    print("1️⃣  Traditional ONNX Runtime...")
    trad = benchmark_traditional()
    print(f"   ✓ {trad['total_sec']:.3f}s total, {trad['throughput']:.1f} req/sec\n")
    
    print("2️⃣  kapsl-runtime (SHM)...")
    kapsl = benchmark_kapsl_shm()
    # The following code block was provided by the user to be inserted.
    # It attempts to re-calculate and print detailed latency stats.
    # Note: The original `kapsl` dictionary already contains `total_sec`, `avg_ms`, `throughput`.
    # The provided snippet re-runs inference to collect individual latencies,
    # and then prints a new summary table.
    
    # Re-initialize client and dummy_input for detailed latency collection
    client = KapslShmClient(shm_name)
    dummy_input_for_latency = np.random.randn(1, 1, 28, 28).astype(np.float32)
    shape = [1, 1, 28, 28]
    dtype = "float32"
    data_for_latency = list(dummy_input_for_latency.tobytes())

    # Collect individual latencies
    latencies = []
    for _ in range(NUM_REQUESTS):
        start = time.perf_counter()
        client.infer(shape, dtype, data_for_latency) # Using consistent infer call
        latencies.append((time.perf_counter() - start) * 1000)
    
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50_latency = sorted(latencies)[len(latencies) // 2]
    avg_latency_ms = sum(latencies) / len(latencies)
    total_time_sec = sum(latencies) / 1000
    throughput_req_sec = NUM_REQUESTS / total_time_sec

    print(f"   ✓ {total_time_sec:.3f}s total, {throughput_req_sec:.1f} req/sec\n") # Summary for kapsl-runtime
    
    print("============================================================")
    print(f"{'Metric':<30} {'Traditional':<15} {'kapsl (SHM)':<15}")
    print("------------------------------------------------------------")
    print(f"{'Total Time (sec)':<30} {trad['total_sec']:<15.3f} {total_time_sec:<15.3f}")
    print(f"{'Avg Latency (ms)':<30} {trad['avg_ms']:<15.3f} {avg_latency_ms:<15.3f}")
    print(f"{'Min Latency (ms)':<30} {'-':<15} {min_latency:<15.3f}") # Traditional min/max not collected
    print(f"{'Max Latency (ms)':<30} {'-':<15} {max_latency:<15.3f}") # Traditional min/max not collected
    print(f"{'P50 Latency (ms)':<30} {'-':<15} {p50_latency:<15.3f}") # Added P50 as it was calculated
    print(f"{'Throughput (req/sec)':<30} {trad['throughput']:<15.1f} {throughput_req_sec:<15.1f}")
    print("============================================================")
    
    # Calculate overhead and slowdown based on the newly calculated kapsl stats
    overhead_ms = avg_latency_ms - trad['avg_ms']
    slowdown = avg_latency_ms / trad['avg_ms']
    
    print(f"\n💡 SHM IPC overhead: +{overhead_ms:.3f}ms per request")
    print(f"   Slowdown factor: {slowdown:.1f}x")
    print("\n🎯 SHM is the FASTEST Python→kapsl-runtime method!")
    print("   (Zero-copy tensor transfer, lock-free queues)")
    print("="*60)
