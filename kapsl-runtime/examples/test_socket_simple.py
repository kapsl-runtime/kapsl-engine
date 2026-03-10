#!/usr/bin/env python3
"""Simple test of socket connection"""
import socket
import struct

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/kapsl.sock")

# Send a simple request
# Header: model_id=0, op_code=1 (infer), payload_size=10
header = struct.pack('<III', 0, 1, 10)
payload = b'0123456789'

print(f"Sending header ({len(header)} bytes): {header.hex()}")
sock.sendall(header)
print(f"Sending payload ({len(payload)} bytes): {payload.hex()}")
sock.sendall(payload)

# Try to read response
print("Reading response header (8 bytes)...")
resp_header = sock.recv(8)
print(f"Received ({len(resp_header)} bytes): {resp_header.hex()}")

if len(resp_header) == 8:
    status, payload_size = struct.unpack('<II', resp_header)
    print(f"Status: {status}, Payload size: {payload_size}")
    
    if payload_size > 0:
        print(f"Reading payload ({payload_size} bytes)...")
        resp_payload = sock.recv(payload_size)
        print(f"Received payload: {resp_payload[:100]}")

sock.close()
print("✓ Success!")
