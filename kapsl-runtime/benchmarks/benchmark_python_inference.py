#!/usr/bin/env python3
"""
Python Inference Benchmark

Compares Python client performance with Rust benchmark for calling
the kapsl-runtime inference server.
"""

import socket
import struct
import time
import statistics

OP_INFER = 1
STATUS_OK = 0

def create_mnist_input():
    """Create dummy 28x28 MNIST image (float32)"""
    import array
    # Create 1x1x28x28 float32 tensor filled with 0.5
    data = array.array('f', [0.5] * (1 * 1 * 28 * 28))
    return data.tobytes()

def serialize_request(input_data):
    """Serialize InferenceRequest using simplified bincode format"""
    import pickle
    
    # Create BinaryTensorPacket structure
    packet = {
        'shape': [1, 1, 28, 28],
        'dtype': 'float32',
        'data': input_data
    }
    
    # Create InferenceRequest
    request = {
        'input': packet,
        'session_id': None
    }
    
    # Use pickle as a simple serialization (not exact bincode, but close enough)
    return pickle.dumps(request)

def send_inference_request(sock, model_id=0):
    """Send inference request and measure latency"""
    # Prepare request
    input_data = create_mnist_input()
    
    try:
        # Use bincode library if available, otherwise pickle
        try:
            import bincode
            # Proper bincode serialization
            packet = {
                'shape': [1, 1, 28, 28],
                'dtype': 'float32',
                'data': list(input_data)
            }
            request = {
                'input': packet,
                'session_id': None
            }
            payload = bincode.encode(request)
        except ImportError:
            payload = serialize_request(input_data)
        
        # Build header
        model_id_bytes = struct.pack('<I', model_id)  # Little-endian u32
        op_code_bytes = struct.pack('<I', OP_INFER)
        payload_size_bytes = struct.pack('<I', len(payload))
        
        # Start timing
        start = time.perf_counter()
        
        # Send request
        sock.sendall(model_id_bytes)
        sock.sendall(op_code_bytes)
        sock.sendall(payload_size_bytes)
        sock.sendall(payload)
        
        # Read response header
        status_bytes = sock.recv(4)
        if len(status_bytes) < 4:
            return None, False
        status = struct.unpack('<I', status_bytes)[0]
        
        resp_size_bytes = sock.recv(4)
        if len(resp_size_bytes) < 4:
            return None, False
        resp_size = struct.unpack('<I', resp_size_bytes)[0]
        
        # Read response payload
        resp_payload = b''
        while len(resp_payload) < resp_size:
            chunk = sock.recv(min(4096, resp_size - len(resp_payload)))
            if not chunk:
                break
            resp_payload += chunk
        
        latency = (time.perf_counter() - start) * 1000  # Convert to ms
        
        success = (status == STATUS_OK)
        return latency, success
        
    except Exception as e:
        print(f"    Error: {e}")
        return None, False

def benchmark_inference(num_warmup=5, num_requests=20):
    """Run inference benchmark"""
    print("🔬 Python Inference Latency Benchmark")
    print("=" * 60)
    print()
    
    socket_path = "/tmp/kapsl.sock"
    model_id = 0
    
    print("Configuration:")
    print(f"  Socket: {socket_path}")
    print(f"  Model ID: {model_id}")
    print(f"  Warmup requests: {num_warmup}")
    print(f"  Benchmark requests: {num_requests}")
    print()
    
    # Warmup phase
    print("🔥 Warming up...")
    for i in range(num_warmup):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)
            latency, success = send_inference_request(sock, model_id)
            sock.close()
            
            if latency is not None:
                status = "✅" if success else "❌"
                print(f"  Warmup {i+1}/{num_warmup}: {latency:.2f}ms {status}")
            else:
                print(f"  Warmup {i+1}/{num_warmup}: Failed")
        except Exception as e:
            print(f"  Warmup {i+1}/{num_warmup}: Connection failed - {e}")
    
    print()
    print("📊 Running benchmark...")
    
    latencies = []
    successes = 0
    
    for i in range(num_requests):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)
            latency, success = send_inference_request(sock, model_id)
            sock.close()
            
            if latency is not None:
                if success:
                    latencies.append(latency)
                    successes += 1
                status = "✅" if success else "❌"
                print(f"  Request {i+1:2}/{num_requests}: {latency:.2f}ms {status}")
            else:
                print(f"  Request {i+1:2}/{num_requests}: Failed")
        except Exception as e:
            print(f"  Request {i+1:2}/{num_requests}: Connection failed - {e}")
    
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
        total_time = sum(latencies) / 1000  # Convert to seconds
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
        
        print()
        print("-" * 60)
        print("📝 Note: This is Python client -> IPC -> inference server")
        print("   Compare with Rust benchmark to see language overhead")
        
        return {
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
        return None
    
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    try:
        results = benchmark_inference(num_warmup=5, num_requests=20)
        if results:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
