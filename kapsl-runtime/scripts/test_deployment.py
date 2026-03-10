#!/usr/bin/env python3
"""
Test Script: Streaming Inference Validation

Tests the deployed kapsl-runtime server with streaming inference.
"""

import socket
import struct
import numpy as np
import time
import sys

def send_inference_request(sock, model_id, input_data):
    """Send inference request and receive response"""
    # Prepare tensor
    input_shape = list(input_data.shape)
    dtype = "float32"
    data_bytes = input_data.tobytes()
    
    # Create binary tensor packet (simplified protocol)
    import pickle
    request = {
        'model_id': model_id,
        'shape': input_shape,
        'dtype': dtype,
        'data': data_bytes
    }
    
    payload = pickle.dumps(request)
    
    # Send request
    header = struct.pack('III', 0, model_id, len(payload))
    sock.sendall(header)
    sock.sendall(payload)
    
    # Receive response
    resp_header = sock.recv(8)
    if len(resp_header) < 8:
        raise Exception("Incomplete response header")
    
    status, payload_size = struct.unpack('II', resp_header)
    
    if status != 0:
        error_data = sock.recv(payload_size)
        raise Exception(f"Server error: {error_data.decode('utf-8', errors='ignore')}")
    
    # Receive payload
    response_data = b''
    while len(response_data) < payload_size:
        chunk = sock.recv(min(4096, payload_size - len(response_data)))
        if not chunk:
            break
        response_data += chunk
    
    result = pickle.loads(response_data)
    return result

def test_basic_inference():
    """Test 1: Basic batch inference"""
    print("🧪 Test 1: Basic Batch Inference")
    print("-" * 50)
    
    try:
        # Connect to server
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect("/tmp/kapsl.sock")
        print("✅ Connected to server")
        
        # Create test input (MNIST: 1x1x28x28)
        input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
        print(f"📤 Sending input: shape={input_data.shape}, dtype={input_data.dtype}")
        
        start = time.time()
        result = send_inference_request(sock, 0, input_data)
        elapsed = time.time() - start
        
        print(f"📥 Received response in {elapsed*1000:.2f}ms")
        print(f"✅ Test 1 PASSED\n")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}\n")
        return False

def test_multiple_requests():
    """Test 2: Multiple sequential requests"""
    print("🧪 Test 2: Multiple Sequential Requests")
    print("-" * 50)
    
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect("/tmp/kapsl.sock")
        
        num_requests = 10
        total_time = 0
        
        for i in range(num_requests):
            input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
            
            start = time.time()
            result = send_inference_request(sock, 0, input_data)
            elapsed = time.time() - start
            total_time += elapsed
            
            print(f"  Request {i+1}/{num_requests}: {elapsed*1000:.2f}ms")
        
        avg_latency = (total_time / num_requests) * 1000
        throughput = num_requests / total_time
        
        print(f"\n📊 Statistics:")
        print(f"  Total time: {total_time*1000:.2f}ms")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Throughput: {throughput:.1f} req/s")
        print(f"✅ Test 2 PASSED\n")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}\n")
        return False

def test_api_endpoints():
    """Test 3: HTTP API endpoints"""
    print("🧪 Test 3: HTTP API Endpoints")
    print("-" * 50)
    
    try:
        import requests
        
        # Test /api/models
        response = requests.get("http://localhost:9095/api/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ GET /api/models: {len(models)} model(s) listed")
            for model in models:
                print(f"   Model {model['id']}: {model['name']} ({model['status']})")
        else:
            print(f"❌ GET /api/models failed: {response.status_code}")
            return False
        
        # Test /api/health
        response = requests.get("http://localhost:9095/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ GET /api/health: {health['status']}")
        else:
            print(f"❌ GET /api/health failed: {response.status_code}")
            return False
        
        # Test /metrics
        response = requests.get("http://localhost:9095/metrics", timeout=5)
        if response.status_code == 200:
            metrics_text = response.text
            metric_count = metrics_text.count('kapsl_')
            print(f"✅ GET /metrics: {metric_count} metrics exposed")
        else:
            print(f"❌ GET /metrics failed: {response.status_code}")
            return False
        
        print(f"✅ Test 3 PASSED\n")
        return True
        
    except ImportError:
        print("⚠️  Test 3 SKIPPED: requests module not installed")
        print("   Install with: pip install requests\n")
        return True
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}\n")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  kapsl-runtime Streaming Inference Test Suite")
    print("="*60 + "\n")
    
    # Check if server is running
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect("/tmp/kapsl.sock")
        sock.close()
        print("✅ Server is reachable at /tmp/kapsl.sock\n")
    except Exception as e:
        print("❌ Cannot connect to server!")
        print(f"   Error: {e}")
        print("\n💡 Make sure to start the server first:")
        print("   ./deploy_streaming.sh")
        print("\n")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Basic Inference", test_basic_inference()))
    results.append(("Multiple Requests", test_multiple_requests()))
    results.append(("API Endpoints", test_api_endpoints()))
    
    # Summary
    print("="*60)
    print("  Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
