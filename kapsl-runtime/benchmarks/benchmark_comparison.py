import time
import numpy as np
import onnxruntime as ort
import os
import sys
from kapsl_runtime import KapslHybridClient

# Configuration
MODEL_PATH = "squeezenet.onnx"
NUM_REQUESTS = 100
WARMUP_REQUESTS = 10
SHM_NAME = f"/kapsl_shm_{os.getpid()}" # This will be different for the client if server is separate process
SOCKET_PATH = "/tmp/kapsl.sock"

def benchmark_onnx_direct(model_path, input_data):
    print(f"Loading model {model_path} with ONNX Runtime...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    print("Warming up...")
    for _ in range(WARMUP_REQUESTS):
        session.run(None, {input_name: input_data})
        
    print(f"Running {NUM_REQUESTS} requests...")
    latencies = []
    start_total = time.time()
    
    for _ in range(NUM_REQUESTS):
        start = time.time()
        session.run(None, {input_name: input_data})
        latencies.append((time.time() - start) * 1000) # ms
        
    total_time = time.time() - start_total
    return latencies, total_time

def benchmark_hybrid_client(shm_name, socket_path, input_data, input_shape):
    print(f"Connecting to kapsl-runtime via Hybrid Client...")
    # Note: shm_name here should match what the server created.
    # We'll assume the server is running and we need to find its SHM name.
    # For this script, we'll pass it as an argument or find it.
    
    try:
        client = KapslHybridClient(shm_name, socket_path)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return [], 0

    print("Warming up...")
    dtype = "float32"
    input_bytes = input_data.tobytes()
    
    for _ in range(WARMUP_REQUESTS):
        try:
            client.infer(input_shape, dtype, input_bytes)
        except Exception as e:
            print(f"Warmup failed: {e}")
            
    print(f"Running {NUM_REQUESTS} requests...")
    latencies = []
    start_total = time.time()
    
    for _ in range(NUM_REQUESTS):
        start = time.time()
        try:
            client.infer(input_shape, dtype, input_bytes)
            latencies.append((time.time() - start) * 1000) # ms
        except Exception as e:
            print(f"Request failed: {e}")
            
    total_time = time.time() - start_total
    return latencies, total_time

def print_stats(name, latencies, total_time):
    if not latencies:
        print(f"No results for {name}")
        return
        
    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = len(latencies) / total_time
    
    print(f"\n--- {name} Results ---")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Latency Avg: {avg:.4f} ms")
    print(f"Latency p50: {p50:.4f} ms")
    print(f"Latency p95: {p95:.4f} ms")
    print(f"Latency p99: {p99:.4f} ms")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    # Prepare input data (SqueezeNet takes 1x3x224x224)
    input_shape = [1, 3, 224, 224]
    input_data = np.random.rand(*input_shape).astype(np.float32)
    
    # 1. Benchmark Direct ONNX
    onnx_latencies, onnx_time = benchmark_onnx_direct(MODEL_PATH, input_data)
    print_stats("Direct ONNX", onnx_latencies, onnx_time)
    
    # 2. Benchmark Hybrid Client
    # We need to know the SHM name of the running server.
    # The server creates /kapsl_shm_<PID>.
    # We will look for the most recent kapsl-runtime process.
    
    import subprocess
    try:
        # Find pid of kapsl
        pid = subprocess.check_output(["pgrep", "-n", "kapsl"]).decode().strip()
        shm_name = f"/kapsl_shm_{pid}"
        print(f"\nFound kapsl-runtime PID: {pid}, using SHM: {shm_name}")
        
        hybrid_latencies, hybrid_time = benchmark_hybrid_client(shm_name, SOCKET_PATH, input_data, input_shape)
        print_stats("Hybrid Client", hybrid_latencies, hybrid_time)
        
        # Comparison
        if onnx_latencies and hybrid_latencies:
            speedup = np.mean(onnx_latencies) / np.mean(hybrid_latencies)
            print(f"\nSpeedup (Direct / Hybrid): {speedup:.2f}x")
            if speedup < 1:
                print(f"Overhead: {(1/speedup - 1)*100:.2f}% slower")
            else:
                print(f"Improvement: {(speedup - 1)*100:.2f}% faster")
                
    except subprocess.CalledProcessError:
        print("\nCould not find running kapsl-runtime process. Skipping hybrid benchmark.")
        print("Please run: cargo run --release -- --model squeezenet.onnx --transport hybrid")

if __name__ == "__main__":
    main()
