#!/usr/bin/env python3
"""
Python Client Benchmark using kapsl_runtime module

Compares Python vs Rust client performance.
"""

import time
import statistics
import array

try:
    from kapsl_runtime import KapslClient
except ImportError:
    print("❌ kapsl_runtime module not found!")
    print("   Build it with: cd crates/kapsl-pyo3 && maturin develop --release")
    exit(1)

def create_mnist_input():
    """Create dummy 28x28 MNIST image"""
    # 1x1x28x28 float32 tensor
    data = array.array('f', [0.5] * (1 * 1 * 28 * 28))
    return list(data)  # Convert to list of floats

def benchmark():
    """Run benchmark with Python client"""
    print("🐍 Python Client Benchmark (via kapsl_runtime module)")
    print("=" * 60)
    print()
    
    socket_path = "/tmp/kapsl.sock"
    num_warmup = 5
    num_requests = 20
    
    print("Configuration:")
    print(f"  Socket: {socket_path}")
    print(f"  Warmup requests: {num_warmup}")
    print(f" Benchmark requests: {num_requests}")
    print()
    
    try:
        client = KapslClient(socket_path)
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return None
    
    shape = [1, 1, 28, 28]
    dtype = "float32"
    
    # Warmup
    print("🔥 Warming up...")
    for i in range(num_warmup):
        try:
            data = create_mnist_input()
            start = time.perf_counter()
            result = client.infer(shape, dtype, data)
            latency = (time.perf_counter() - start) * 1000
            print(f"  Warmup {i+1}/{num_warmup}: {latency:.2f}ms ✅")
        except Exception as e:
            print(f"  Warmup {i+1}/{num_warmup}: Failed - {e}")
    
    print()
    print("📊 Running benchmark...")
    
    latencies = []
    successes = 0
    
    for i in range(num_requests):
        try:
            data = create_mnist_input()
            start = time.perf_counter()
            result = client.infer(shape, dtype, data)
            latency = (time.perf_counter() - start) * 1000
            
            latencies.append(latency)
            successes += 1
            print(f"  Request {i+1:2}/{num_requests}: {latency:.2f}ms ✅")
        except Exception as e:
            print(f"  Request {i+1:2}/{num_requests}: Failed - {e}")
    
    print()
    print("=" * 60)
    print("Results:")
    print("-" * 60)
    
    if latencies:
        avg = statistics.mean(latencies)
        median = statistics.median(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        print(f"Total requests: {num_requests}")
        print(f"Successful: {successes}")
        print(f"Failed: {num_requests - successes}")
        print()
        print("Latency Statistics:")
        print(f"  Average: {avg:.2f}ms")
        print(f"  Median:  {median:.2f}ms")
        print(f"  Min:     {min_lat:.2f}ms")
        print(f"  Max:     {max_lat:.2f}ms")
        print(f"  P95:     {p95:.2f}ms")
        print(f"  P99:     {p99:.2f}ms")
        print()
        
        # Throughput
        total_time = sum(latencies) / 1000
        throughput = num_requests / total_time
        print(f"Throughput: {throughput:.1f} inferences/sec")
        print()
        
        # Performance rating
        if avg < 5.0:
            print("✅ Excellent performance (<5ms)")
        elif avg < 10.0:
            print("✅ Very good performance (<10ms)")
        elif avg < 50.0:
            print("✅ Good performance (<50ms)")
        elif avg < 100.0:
            print("⚠️  Acceptable performance (<100ms)")
        else:
            print("❌ Poor performance (>100ms)")
        
        print("=" * 60)
        print()
        
        return {
            'language': 'Python',
            'avg': avg,
            'median': median,
            'min': min_lat,
            'max': max_lat,
            'p95': p95,
            'p99': p99,
            'throughput': throughput,
            'success_rate': successes / num_requests
        }
    else:
        print("❌ No successful requests")
        print("=" * 60)
        return None

if __name__ == "__main__":
    results = benchmark()
    
    if results:
        print()
        print("💡 Comparison Notes:")
        print("   - This uses PyO3 bindings (Rust code called from Python)")
        print("   - Performance should be close to pure Rust")
        print("   - Main overhead: Python->Rust FFI boundary")
        print("   - Compare with pure Rust benchmark for language overhead")
        exit(0)
    else:
        exit(1)
