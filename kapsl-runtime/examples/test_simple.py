#!/usr/bin/env python3
"""
Simple Test: No external dependencies

Tests the deployed kapsl-runtime server using only standard library.
"""

import socket
import json
import urllib.request
import sys

def test_http_api():
    """Test HTTP API endpoints"""
    print("="*60)
    print("  API Endpoint Tests")
    print("="*60 + "\n")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: GET /api/models
    tests_total += 1
    try:
        response = urllib.request.urlopen("http://localhost:9095/api/models", timeout=5)
        data = json.loads(response.read())
        
        print("✅ Test 1: GET /api/models")
        print(f"   Models loaded: {len(data)}")
        for model in data:
            print(f"   - Model {model['id']}: {model['name']} ({model['status']})")
            print(f"     Device: {model['device']}, Framework: {model['framework']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
    
    print()
    
    # Test 2: GET /api/health
    tests_total += 1
    try:
        response = urllib.request.urlopen("http://localhost:9095/api/health", timeout=5)
        data = json.loads(response.read())
        
        print("✅ Test 2: GET /api/health")
        print(f"   Status: {data['status']}")
        print(f"   Total models: {data['total_models']}")
        print(f"   Healthy models: {data['healthy_models']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
    
    print()
    
    # Test 3: GET /metrics
    tests_total += 1
    try:
        response = urllib.request.urlopen("http://localhost:9095/metrics", timeout=5)
        metrics_text = response.read().decode('utf-8')
        
        kapsl_metrics = [line for line in metrics_text.split('\n') if line.startswith('kapsl_')]
        
        print("✅ Test 3: GET /metrics")
        print(f"   Prometheus metrics exposed: {len(kapsl_metrics)}")
        print("   Sample metrics:")
        for metric in kapsl_metrics[:5]:
            print(f"   - {metric[:60]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
    
    print()
    
    # Test 4: Check server connectivity
    tests_total += 1
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect("/tmp/kapsl.sock")
        sock.close()
        
        print("✅ Test 4: Unix Socket Connectivity")
        print("   Socket path: /tmp/kapsl.sock")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
    
    print()
    print("="*60)
    print(f"  Results: {tests_passed}/{tests_total} tests passed")
    print("="*60 + "\n")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! Server is healthy.")
        return 0
    else:
        print(f"⚠️  {tests_total - tests_passed} test(s) failed")
        return 1

def main():
    """Run tests"""
    print("\n🚀 kapsl-runtime Streaming Inference Tests\n")
    
    exit_code = test_http_api()
    
    print("\n💡 Next steps:")
    print("   - View web dashboard: http://localhost:9095/")
    print("   - Check metrics: http://localhost:9095/metrics")
    print("   - View logs: tail -f kapsl_streaming.log")
    print("   - Stop server: kill $(cat kapsl.pid) or pkill kapsl")
    print()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
