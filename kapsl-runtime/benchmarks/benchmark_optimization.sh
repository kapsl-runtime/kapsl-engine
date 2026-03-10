#!/bin/bash

# Benchmark script to test different optimization levels

echo "=== ONNX Runtime Graph Optimization Benchmark ==="
echo "Model: MNIST"
echo "Hardware: $(uname -m) CPU"
echo ""

# Create a simple test input (28x28 grayscale image)
python3 -c '
import numpy as np
import json

# Create random test input
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

# Save as binary
with open("test_input.bin", "wb") as f:
    f.write(input_data.tobytes())

print("Created test input: 1x1x28x28 float32")
'

# Test each optimization level
for opt_level in disable basic extended all; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing optimization level: $opt_level"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run the model and capture output
    echo "Loading model..."
    timeout 10s target/release/kapsl --model "scripts/mnist_opt_${opt_level}.aimod" 2>&1 | grep -E "(optimization|Loading|Using|Error)" || echo "Timeout or error"
    
    echo ""
done

echo "=== Benchmark Complete ==="
echo ""
echo "Note: For detailed performance metrics, use the metrics endpoint on port 9095"
