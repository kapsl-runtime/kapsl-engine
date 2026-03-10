#!/usr/bin/env python3
"""
Simple script to send inference requests to kapsl-runtime for testing metrics.
"""
import socket
import json
import struct

def send_inference_request(model_id=0, num_requests=5):
    """Send inference requests to the running model."""
    
    host = 'localhost'
    port = 9095
    
    print(f"Connecting to {host}:{port}...")
    
    for i in range(num_requests):
        try:
            # Create a new socket for each request
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            
            # Create a simple inference request
            request = {
                "model_id": model_id,
                "inputs": {
                    "input_data": [1.0, 2.0, 3.0, 4.0]  # Dummy input data
                },
                "priority": 0  # Normal priority
            }
            
            # Serialize to JSON
            request_json = json.dumps(request)
            request_bytes = request_json.encode('utf-8')
            
            # Send length prefix (4 bytes, big-endian)
            length = len(request_bytes)
            sock.sendall(struct.pack('>I', length))
            
            # Send the actual request
            sock.sendall(request_bytes)
            
            print(f"Request {i+1}/{num_requests} sent to model {model_id}")
            
            # Try to receive response (with timeout)
            sock.settimeout(5.0)
            try:
                # Read response length
                length_bytes = sock.recv(4)
                if length_bytes:
                    response_length = struct.unpack('>I', length_bytes)[0]
                    
                    # Read response data
                    response_data = b''
                    while len(response_data) < response_length:
                        chunk = sock.recv(min(response_length - len(response_data), 4096))
                        if not chunk:
                            break
                        response_data += chunk
                    
                    if len(response_data) == response_length:
                        response = json.loads(response_data.decode('utf-8'))
                        print(f"  ✓ Response received: {response.get('status', 'unknown')}")
                    else:
                        print(f"  ⚠ Incomplete response received")
                else:
                    print(f"  ⚠ No response received")
            except socket.timeout:
                print(f"  ⚠ Response timed out (expected for test model)")
            except Exception as e:
                print(f"  ⚠ Error reading response: {e}")
            
            sock.close()
            
        except ConnectionRefusedError:
            print(f"  ✗ Connection refused - is kapsl-runtime running on port {port}?")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Sent {num_requests} inference requests")
    print("Check the dashboard to see updated metrics!")

if __name__ == '__main__':
    import sys
    
    model_id = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Sending {num_requests} inference requests to model {model_id}...")
    send_inference_request(model_id, num_requests)
