#!/usr/bin/env python3
"""
Example: Using LLM Streaming Inference with kapsl-runtime

This demonstrates how to use the streaming inference API for LLM token generation.
"""

import numpy as np
from kapsl_runtime import KapslClient
import time

def main():
    # Connect to runtime
    client = KapslClient("/tmp/kapsl.sock")
    
    print("🚀 LLM Streaming Inference Example\n")
    
    # Prepare input: prompt as token IDs (uint32)
    # In a real scenario, you'd use a tokenizer to convert text to token IDs
    prompt_text = "Hello, how are you?"
    prompt_tokens = [15496, 11, 703, 389, 345, 30]  # Mock token IDs
    
    print(f"📝 Prompt: {prompt_text}")
    print(f"🔢 Tokens: {prompt_tokens}\n")
    
    # Convert to binary format
    input_data = np.array(prompt_tokens, dtype=np.uint32)
    input_shape = [1, len(prompt_tokens)]
    input_bytes = input_data.tobytes()
    
    print("🔄 Starting streaming inference...\n")
    print("Generated tokens:")
    print("-" * 50)
    
    # OPTION 1: Using infer_stream (if implemented in Python bindings)
    # This would yield tokens as they're generated
    start_time = time.time()
    token_count = 0
    
    print("Streaming tokens: ", end="", flush=True)
    
    for token_result in client.infer_stream(
        model_id=0,
        shape=input_shape,
        dtype="uint32",
        data=input_bytes
    ):
        # PyO3 returns Vec<u8> as a list of integers if not handled, but our StreamIterator returns bytes (Vec<u8>)
        # Wait, StreamIterator returns Option<Vec<u8>>. PyO3 converts Vec<u8> to bytes automatically?
        # In infer_impl, we returned Vec<u8> and it came as list of ints.
        # Let's assume it comes as bytes or list of ints.
        
        # If it's bytes:
        if isinstance(token_result, bytes):
            token_id = np.frombuffer(token_result, dtype=np.uint32)[0]
        else:
            # If list of ints
            token_id = np.frombuffer(bytes(token_result), dtype=np.uint32)[0]
            
        print(f"{token_id}", end=" ", flush=True)
        token_count += 1
    
    print("\n")
    elapsed = time.time() - start_time
    
    print(f"⏱️  Total time: {elapsed*1000:.2f}ms")
    print(f"📊 Tokens generated: {token_count}")
    if elapsed > 0:
        print(f"⚡ Tokens/sec: {token_count/elapsed:.1f}")
    
    print("\n" + "-" * 50)
    print("✅ Streaming example completed!")
    print("\n💡 Note: True streaming (token-by-token) requires:")
    print("   1. Python bindings updated to support infer_stream()")
    print("   2. Transport protocol updated for streaming responses")
    print("   3. See llm_backend.rs for the Rust implementation")

if __name__ == "__main__":
    main()
