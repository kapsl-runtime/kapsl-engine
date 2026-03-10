#!/usr/bin/env python3
"""
Real Inference Latency Benchmark

Measures actual ML model inference latency, not just API response time.
"""

import socket
import struct
import time
import sys

def create_dummy_mnist_input():
    """Create a dummy 28x28 MNIST image (random data)"""
    import array
    # MNIST input: 1x1x28x28 float32
    size = 1 * 1 * 28 * 28
    data = array.array('f', [0.5] * size)  # Fill with 0.5
    return data.tobytes()

def send_inference_request(sock):
    """Send an actual inference request and measure latency"""
    # Prepare request
    model_id = 0
    input_data = create_dummy_mnist_input()
    
    # Simple protocol: [magic][model_id][data_size][data]
    magic = 0xABCD
    data_size = len(input_data)
    
    header = struct.pack('HHI', magic, model_id, data_size)
    
    try:
        # Send request
        start = time.time()
        sock.sendall(header)
        sock.sendall(input_data)
        
        # Receive response (simplified - just read some bytes)
        response = sock.recv(4096)
        latency = (time.time() - start) * 1000
        
        return latency, len(response) > 0
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None, False

def benchmark_inference(num_requests=10):
    """Run inference benchmark"""
    print("🔬 Real Inference Latency Benchmark")
    print("=" * 60)
    print(f"Testing with {num_requests} inference requests...\n")
    
    latencies = []
    success_count = 0
    
    for i in range(num_requests):
        try:
            # Create new connection for each request
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect("/tmp/kapsl.sock")
            
            latency, success = send_inference_request(sock)
            
            if latency is not None:
                latencies.append(latency)
                if success:
                    success_count += 1
                    
                print(f"Request {i+1}/{num_requests}: {latency:.2f}ms {'✅' if success else '❌'}")
            
            sock.close()
            
        except Exception as e:
            print(f"Request {i+1}/{num_requests}: Failed - {e}")
        
        # Small delay between requests
        time.sleep(0.1)
    
    # Results
    print("\n" + "=" * 60)
    print("Results:")
    print("-" * 60)
    
    if latencies:
        avg = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        print(f"Total requests: {num_requests}")
        print(f"Successful: {success_count}")
        print(f"Failed: {num_requests - success_count}")
        print()
        print(f"Average latency: {avg:.2f}ms")
        print(f"Min latency: {min_lat:.2f}ms")
        print(f"Max latency: {max_lat:.2f}ms")
        print(f"Median latency: {sorted(latencies)[len(latencies)//2]:.2f}ms")
        print()
        
        if avg < 10:
            print("✅ Excellent latency (<10ms)")
        elif avg < 50:
            print("✅ Good latency (<50ms)")
        elif avg < 100:
            print("⚠️  Acceptable latency (<100ms)")
        else:
            print("❌ High latency (>100ms)")
    else:
        print("❌ No successful requests")
    
    print("=" * 60)
    print()
    print("📝 Note: This measures socket communication + inference time")
    print("   For GPU inference, expect 5-50ms depending on model size")
    print()

if __name__ == "__main__":
    try:
        benchmark_inference(10)
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
