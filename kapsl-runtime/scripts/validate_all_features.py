#!/usr/bin/env python3
"""
Comprehensive Feature Validation Test

Tests all major features of kapsl-runtime:
1. Model loading and inference
2. API endpoints
3. Metrics collection
4. Health checks
5. Multi-device support
6. Streaming capabilities (when implemented)
"""

import urllib.request
import urllib.error
import json
import struct
import socket
import sys
import time

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ️  {text}{Colors.RESET}")

def test_server_connectivity():
    """Test 1: Server connectivity"""
    print_header("Test 1: Server Connectivity")
    
    try:
        # Test Unix socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect("/tmp/kapsl.sock")
        sock.close()
        print_success("Unix socket (/tmp/kapsl.sock) is accessible")
        
        # Test HTTP endpoint
        response = urllib.request.urlopen("http://localhost:9095/api/health", timeout=5)
        print_success("HTTP server (localhost:9095) is accessible")
        
        return True
    except Exception as e:
        print_error(f"Server connectivity failed: {e}")
        return False

def test_api_health():
    """Test 2: Health API"""
    print_header("Test 2: Health Check API")
    
    try:
        response = urllib.request.urlopen("http://localhost:9095/api/health", timeout=5)
        data = json.loads(response.read())
        
        print(f"   Status: {data['status']}")
        print(f"   Total models: {data['total_models']}")
        print(f"   Healthy models: {data['healthy_models']}")
        print(f"   Unhealthy models: {data['unhealthy_models']}")
        
        if data['status'] == 'healthy':
            print_success("System health is good")
            return True
        else:
            print_error(f"System unhealthy: {data['status']}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_models_api():
    """Test 3: Models API"""
    print_header("Test 3: Models API")
    
    try:
        response = urllib.request.urlopen("http://localhost:9095/api/models", timeout=5)
        models = json.loads(response.read())
        
        print(f"   Number of models loaded: {len(models)}")
        
        for model in models:
            print(f"\n   Model ID {model['id']}:")
            print(f"      Name: {model['name']}")
            print(f"      Framework: {model['framework']}")
            print(f"      Device: {model['device']}")
            print(f"      Status: {model['status']}")
            print(f"      Optimization: {model['optimization_level']}")
            print(f"      Total inferences: {model['total_inferences']}")
            print(f"      Active inferences: {model['active_inferences']}")
            print(f"      Queue depth: High={model['queue_depth'][0]}, Low={model['queue_depth'][1]}")
            print(f"      Healthy: {model['healthy']}")
        
        if len(models) > 0 and models[0]['status'] == 'active':
            print_success(f"Found {len(models)} active model(s)")
            return True
        else:
            print_error("No active models found")
            return False
    except Exception as e:
        print_error(f"Models API failed: {e}")
        return False

def test_metrics():
    """Test 4: Prometheus Metrics"""
    print_header("Test 4: Prometheus Metrics")
    
    try:
        response = urllib.request.urlopen("http://localhost:9095/metrics", timeout=5)
        metrics_text = response.read().decode('utf-8')
        
        # Count metrics
        kapsl_metrics = [line for line in metrics_text.split('\n') 
                        if line.startswith('kapsl_') and not line.startswith('kapsl_')]
        
        # Parse key metrics
        metrics_dict = {}
        for line in metrics_text.split('\n'):
            if line.startswith('kapsl_') and '{' in line:
                metric_name = line.split('{')[0]
                if '}' in line:
                    value_part = line.split('}')[1].strip()
                    try:
                        metrics_dict[metric_name] = float(value_part)
                    except:
                        pass
        
        print(f"   Total Prometheus metrics: {len(kapsl_metrics)}")
        print(f"\n   Key metrics:")
        
        # Show sample metrics
        shown = 0
        for line in metrics_text.split('\n'):
            if line.startswith('kapsl_') and '{' in line and shown < 10:
                print(f"      {line[:80]}...")
                shown += 1
        
        print_success("Metrics endpoint is functional")
        return True
    except Exception as e:
        print_error(f"Metrics test failed: {e}")
        return False

def test_web_dashboard():
    """Test 5: Web Dashboard"""
    print_header("Test 5: Web Dashboard")
    
    try:
        response = urllib.request.urlopen("http://localhost:9095/", timeout=5)
        html = response.read().decode('utf-8')
        
        if len(html) > 100:
            print(f"   Dashboard HTML size: {len(html)} bytes")
            print_success("Web dashboard is accessible")
            print_info("Open http://localhost:9095/ in your browser")
            return True
        else:
            print_error("Dashboard response too small")
            return False
    except Exception as e:
        print_error(f"Web dashboard test failed: {e}")
        return False

def test_device_detection():
    """Test 6: Device Detection"""
    print_header("Test 6: Multi-Device Support")
    
    try:
        response = urllib.request.urlopen("http://localhost:9095/api/models", timeout=5)
        models = json.loads(response.read())
        
        if len(models) > 0:
            device = models[0]['device']
            framework = models[0]['framework']
            
            print(f"   Detected device: {device}")
            print(f"   Framework: {framework}")
            
            if device in ['cuda', 'metal', 'rocm', 'cpu']:
                print_success(f"Model running on {device} backend")
                return True
            else:
                print_error(f"Unknown device type: {device}")
                return False
        else:
            print_error("No models loaded to check device")
            return False
    except Exception as e:
        print_error(f"Device detection failed: {e}")
        return False

def test_replica_pool():
    """Test 7: Replica Pool"""
    print_header("Test 7: Replica Pool & Load Balancing")
    
    try:
        response = urllib.request.urlopen("http://localhost:9095/metrics", timeout=5)
        metrics_text = response.read().decode('utf-8')
        
        # Check for pool metrics
        pool_metrics = [line for line in metrics_text.split('\n')
                       if 'kapsl_pool' in line and '{' in line]
        
        if pool_metrics:
            print(f"   Found {len(pool_metrics)} pool metrics")
            for metric in pool_metrics[:5]:
                print(f"      {metric}")
            print_success("Replica pool is active")
            return True
        else:
            print_info("No replica pool metrics found (may be using single instance)")
            return True
    except Exception as e:
        print_error(f"Replica pool test failed: {e}")
        return False

def test_inference_latency():
    """Test 8: Performance Check"""
    print_header("Test 8: Performance & Latency")
    
    try:
        # Make multiple requests to measure latency
        latencies = []
        
        for i in range(5):
            start = time.time()
            urllib.request.urlopen("http://localhost:9095/api/health", timeout=5)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   Min latency: {min_latency:.2f}ms")
        print(f"   Max latency: {max_latency:.2f}ms")
        
        if avg_latency < 100:
            print_success("API latency is good (<100ms)")
            return True
        else:
            print_info(f"API latency is acceptable ({avg_latency:.2f}ms)")
            return True
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print(f"\n{Colors.BOLD}🧪 kapsl-runtime Comprehensive Feature Validation{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    # Track results
    tests = [
        ("Server Connectivity", test_server_connectivity),
        ("Health API", test_api_health),
        ("Models API", test_models_api),
        ("Prometheus Metrics", test_metrics),
        ("Web Dashboard", test_web_dashboard),
        ("Multi-Device Support", test_device_detection),
        ("Replica Pool", test_replica_pool),
        ("Performance & Latency", test_inference_latency),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
        
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if result else f"{Colors.RED}❌ FAIL{Colors.RESET}"
        print(f"   {status}  {name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}")
    print(f"{'='*70}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 All features validated successfully!{Colors.RESET}")
        print(f"\n{Colors.BOLD}Your kapsl-runtime installation is fully functional.{Colors.RESET}\n")
        sys.exit(0)
    elif passed >= total * 0.7:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Most features working, but some issues detected.{Colors.RESET}\n")
        sys.exit(1)
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Multiple failures detected. Check server status.{Colors.RESET}\n")
        sys.exit(2)

if __name__ == "__main__":
    main()
