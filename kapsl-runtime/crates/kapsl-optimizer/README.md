# kapsl-optimizer

Graph-level optimization for ONNX models in the kapsl-runtime.

## Status: Stub Implementation

This crate defines the API and structure for graph-level optimization but currently contains stub implementations.

### Why Stub?

The `onnx` crate (v0.1.0) has all protobuf fields marked as private, making direct graph manipulation impossible without:

1. Using a different ONNX manipulation library (e.g., tract, or Python bindings to onnxruntime)
2. Writing custom protobuf wrappers with public fields
3. Using ONNX Runtime's `SessionOptions` for optimization instead of pre-processing

### Current API

The crate provides a complete API surface:

```rust
use kapsl_optimizer::{GraphOptimizer, OptimizationConfig, OptimizationLevel};

// Create optimizer with O2 optimization level
let config = OptimizationConfig::new(OptimizationLevel::O2);
let optimizer = GraphOptimizer::new(config);

// Optimize a model
let report = optimizer.optimize(
    Path::new("model.onnx"),
    Path::new("model_optimized.onnx"),
)?;

log::info!("Reduced from {} to {} nodes", 
    report.original_nodes, 
    report.optimized_nodes
);
```

### Optimization Levels

- **O0**: No optimization
- **O1**: Basic (constant folding, dead code elimination)
- **O2**: Aggressive (+ operator fusion, reshape elimination)
- **O3**: Maximum (all passes + experimental)

### Future Implementation

To complete this crate, we need to either:

1. **Use ONNX Runtime Graph Optimization** - leverage `ort::SessionOptions::with_optimization_level()`
2. **Python Bridge** - use PyO3 to call `onnxruntime.transformers.optimizer`
3. **Custom Implementation** - write custom protobuf structs with public fields

The recommended approach is #1 for simplicity, using ONNX Runtime's built-in graph optimization capabilities.

## Usage

```rust
use kapsl_optimizer::{GraphOptimizer, OptimizationConfig, OptimizationLevel};

let config = OptimizationConfig::new(OptimizationLevel::O2);
let optimizer = GraphOptimizer::new(config);

// Currently just validates and copies the model
let report = optimizer.optimize(input_path, output_path)?;
```

## Testing

```bash
cargo test -p kapsl-optimizer
```
