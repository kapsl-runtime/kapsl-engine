#!/usr/bin/env python3
"""
Simplified concurrent benchmark
Since HTTP API doesn't have /infer endpoint, we'll compare:
1. Traditional ONNX - 100 sequential requests
2. Traditional ONNX - 100 threaded requests (simulating concurrent load)
"""
import time
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_REQUESTS = 100

# Global session for thread-safe concurrent access
session = None

def init_session():
    global session
    session = ort.InferenceSession("scripts/mnist.onnx")

def single_inference(request_id):
    """Single inference call"""
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    start = time.time()
    session.run(None, {input_name: dummy_input})
    latency = (time.time() - start) * 1000
    
    return latency

def benchmark_sequential():
    """Sequential processing - one at a time"""
    print("📊 Processing 100 requests sequentially...")
    
    init_session()
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    
    start = time.time()
    for _ in range(NUM_REQUESTS):
        session.run(None, {input_name: dummy_input})
    end = time.time()
    
    total_time = end - start
    throughput = NUM_REQUESTS / total_time
    avg_latency = (total_time / NUM_REQUESTS) * 1000
    
    return {
        "method": "Sequential",
        "total_time_sec": total_time,
        "avg_latency_ms": avg_latency,
        "throughput_rps": throughput,
    }

def benchmark_concurrent():
    """Concurrent processing with thread pool"""
    print(f"📊 Processing 100 requests concurrently (20 workers)...")
    
    init_session()
    
    start = time.time()
    
    latencies = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(single_inference, i) for i in range(NUM_REQUESTS)]
        for future in as_completed(futures):
            latencies.append(future.result())
    
    end = time.time()
    
    total_time = end - start
    throughput = NUM_REQUESTS / total_time
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    return {
        "method": "Concurrent (ThreadPool)",
        "total_time_sec": total_time,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "throughput_rps": throughput,
    }

def print_results(seq_results, conc_results):
    """Print comparison"""
    print("\n" + "="*70)
    print(f"📊 CONCURRENCY IMPACT ({NUM_REQUESTS} requests)")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Sequential':<20} {'Concurrent':<20}")
    print("-"*70)
    print(f"{'Total Time (sec)':<30} {seq_results['total_time_sec']:<20.3f} {conc_results['total_time_sec']:<20.3f}")
    print(f"{'Avg Latency (ms)':<30} {seq_results['avg_latency_ms']:<20.3f} {conc_results['avg_latency_ms']:<20.3f}")
    print(f"{'P50 Latency (ms)':<30} {'N/A':<20} {conc_results['p50_latency_ms']:<20.3f}")
    print(f"{'P95 Latency (ms)':<30} {'N/A':<20} {conc_results['p95_latency_ms']:<20.3f}")
    print(f"{'P99 Latency (ms)':<30} {'N/A':<20} {conc_results['p99_latency_ms']:<20.3f}")
    print(f"{'Throughput (req/sec)':<30} {seq_results['throughput_rps']:<20.1f} {conc_results['throughput_rps']:<20.1f}")
    
    speedup = seq_results['total_time_sec'] / conc_results['total_time_sec']
    throughput_gain = conc_results['throughput_rps'] / seq_results['throughput_rps']
    
    print("\n" + "="*70)
    print("🎯 KEY INSIGHTS:")
    print("="*70)
    print(f"\n✅ Concurrent processing: {speedup:.2f}x faster")
    print(f"✅ Throughput improvement: {throughput_gain:.2f}x")
    print(f"\n💡 But latency increased: {conc_results['avg_latency_ms']/seq_results['avg_latency_ms']:.2f}x")
    print("   This is the concurrency overhead - threads compete for resources")
    print("\n📌 NOTE: kapsl-runtime would queue these requests and batch them")
    print("   for better GPU utilization than simple threading!")
    print("="*70)

if __name__ == "__main__":
    print("🚀 Concurrency Overhead Benchmark\n")
    print(f"Testing with {NUM_REQUESTS} ONNX Runtime requests\n")
    print("NOTE: kapsl-runtime HTTP API doesn't expose /infer endpoint")
    print("This shows threading overhead in Python as a baseline\n")
    
    # Sequential
    print("1️⃣  Sequential Processing")
    seq_results = benchmark_sequential()
    print(f"   ✓ {seq_results['total_time_sec']:.3f}s total, {seq_results['throughput_rps']:.1f} req/sec\n")
    
    # Concurrent
    print("2️⃣  Concurrent Processing (Thread Pool)")
    conc_results = benchmark_concurrent()
    print(f"   ✓ {conc_results['total_time_sec']:.3f}s total, {conc_results['throughput_rps']:.1f} req/sec\n")
    
    print_results(seq_results, conc_results)
