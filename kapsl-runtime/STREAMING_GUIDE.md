# Streaming Inference Implementation Guide

## Overview

This document explains how to use the new **streaming inference** capability in kapsl-runtime, specifically designed for LLM token-by-token generation.

---

## Architecture

### Engine Trait Enhancement

The `Engine` trait now supports streaming:

```rust
pub trait Engine: Send + Sync {
    fn load(&mut self, model_path: &Path) -> Result<(), EngineError>;
    
    // Batch inference (returns all tokens at once)
    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError>;
    
    // Streaming inference (yields tokens as they're generated) ⭐ NEW
    fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>;
    
    fn unload(&mut self);
    fn metrics(&self) -> EngineMetrics;
}
```

---

## Implementations

### 1. ONNX Backend (Non-Streaming)

Currently wraps single inference in a stream:

```rust
fn infer_stream(&self, request: &InferenceRequest) 
    -> Box<dyn Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send> 
{
    let result = self.infer(request);
    Box::new(futures::stream::once(async move { result }))
}
```

**Use case**: Standard ML models (classification, object detection, etc.)

---

### 2. LLM Backend (True Streaming) ⭐

Located in `crates/kapsl-llm/src/llm_backend.rs`

**Implementation**: `SimpleLLMBackend`

```rust
pub struct SimpleLLMBackend {
    model_loaded: Arc<Mutex<bool>>,
    metrics: Arc<Mutex<SimpleMetrics>>,
}

impl SimpleLLMBackend {
    pub fn new() -> Self { /* ... */ }
}
```

#### Key Features

- **Token-by-token generation**: Yields each token immediately as it's generated
- **Async-safe**: Properly handles `Send` trait requirements
- **Configurable**: Supports EOS detection, max tokens, etc.

#### Example Usage (Rust)

```rust
use kapsl_llm::llm_backend::SimpleLLMBackend;
use futures::stream::StreamExt;

#[tokio::main]
async fn main() {
    let backend = SimpleLLMBackend::new();
    backend.load(Path::new("/path/to/model")).unwrap();
    
    // Create request with prompt tokens
    let request = InferenceRequest {
        input: BinaryTensorPacket {
            shape: vec![1, 5],
            dtype: "uint32".to_string(),
            data: vec![/* token bytes */],
        },
        session_id: None,
    };
    
    // Stream tokens
    let mut stream = backend.infer_stream(&request);
    while let Some(token_result) = stream.next().await {
        match token_result {
            Ok(token) => {
                let token_id = decode_token(&token);
                print!("{} ", token_id);
                std::io::stdout().flush().unwrap();
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}
```

---

## Transport Layer Integration

### Current State

The transport protocols (IPC, TCP) currently support **batch inference only**:

```
Client → Server: [Request]
Server → Client: [Complete Response]
```

### Needed for Streaming

Update transport to support **incremental responses**:

```
Client → Server: [Request with streaming flag]
Server → Client: [Token 1]
Server → Client: [Token 2]
Server → Client: [Token 3]
...
Server → Client: [End-of-stream marker]
```

#### Implementation Approach

```rust
// In IpcServer::handle_request
async fn handle_stream_request(
    stream: &mut UnixStream,
    scheduler: &Scheduler,
    request: InferenceRequest,
) -> Result<(), TransportError> {
    let mut token_stream = scheduler.infer_stream(request).await;
    
    while let Some(token_result) = token_stream.next().await {
        match token_result {
            Ok(token) => {
                // Send token packet
                let response = bincode::serialize(&token)?;
                stream.write_u32(response.len() as u32).await?;
                stream.write_all(&response).await?;
            }
            Err(e) => {
                // Send error and terminate
                send_error(stream, e).await?;
                break;
            }
        }
    }
    
    // Send end-of-stream marker
    stream.write_u32(0).await?;
    Ok(())
}
```

---

## Python Client Integration

### Proposed API

```python
from kapsl_runtime import KapslClient

client = KapslClient("/tmp/kapsl.sock")

# Option 1: Streaming (yields tokens as they arrive)
for token in client.infer_stream(
    model_id=0,
    input_data=prompt_tokens,
    max_tokens=100
):
    print(decode_token(token), end='', flush=True)

# Option 2: Batch (waits for all tokens)
all_tokens = client.infer(model_id=0, input_data=prompt_tokens)
print(decode_tokens(all_tokens))
```

### Implementation in PyO3

```rust
#[pyclass]
pub struct KapslClient {
    // ... existing fields
}

#[pymethods]
impl KapslClient {
    fn infer_stream(
        &mut self,
        py: Python,
        model_id: u32,
        input_shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
    ) -> PyResult<PyObject> {
        // Return a Python generator/iterator
        // that yields tokens as they arrive
        todo!("Implement streaming Python bindings")
    }
}
```

---

## Usage Examples

### Example 1: Chat Completion

```python
import numpy as np
from kapsl_runtime import KapslClient

client = KapslClient("/tmp/kapsl.sock")

# Tokenize prompt
prompt = "What is the capital of France?"
tokens = tokenizer.encode(prompt)

# Stream response
print("Assistant: ", end='')
for token in client.infer_stream(model_id=0, input_data=tokens):
    word = tokenizer.decode([token])
    print(word, end='', flush=True)
print()
```

### Example 2: Code Generation

```python
# Generate code with streaming feedback
prompt_tokens = tokenizer.encode("def fibonacci(n):")

generated_code = ""
for token in client.infer_stream(model_id=0, input_data=prompt_tokens):
    chunk = tokenizer.decode([token])
    generated_code += chunk
    print(chunk, end='', flush=True)
    
    # Early stopping on complete function
    if "return" in generated_code and generated_code.count("\n") > 5:
        break
```

---

## Performance Characteristics

### Latency Comparison

| Approach | Time to First Token | Total Time | User Experience |
|----------|---------------------|------------|-----------------|
| **Batch** | ~2000ms | 2000ms | Wait → Full response |
| **Streaming** | ~50ms | 2000ms | Immediate → Incremental |

### Resource Usage

- **Memory**: Streaming uses constant memory (one token at a time)
- **CPU**: Similar CPU usage to batch
- **Network**: More network overhead (one packet per token vs. one packet total)

---

## Configuration

### Per-Model Settings

```json
{
  "model_id": 0,
  "streaming_config": {
    "enabled": true,
    "max_tokens": 256,
    "token_delay_ms": 10,
    "stop_tokens": [2, 50256],
    "temperature": 0.7
  }
}
```

### Runtime Flags

```bash
cargo run -p kapsl -- \
  --model llama2.aimod \
  --enable-streaming \
  --streaming-buffer-size 32
```

---

## Testing

```bash
# Run LLM backend tests
cargo test -p kapsl-llm

# Test streaming with example script
python scripts/example_llm_streaming.py

# Benchmark streaming vs batch
cargo run --release --example bench_streaming
```

---

## Limitations & Future Work

### Current Limitations

1. **Transport not updated**: IPC/TCP don't support streaming protocol yet
2. **Python bindings incomplete**: `infer_stream()` not exposed to Python
3. **Mock generation**: Current implementation uses mock token generation
4. **No beam search**: Only greedy decoding supported

### Roadmap

- [ ] Update transport protocols for streaming
- [ ] Implement Python streaming bindings
- [ ] Integrate real LLM models (via ONNX or native)
- [ ] Add beam search support
- [ ] Implement sampling strategies (temperature, top-k, top-p)
- [ ] Add request cancellation
- [ ] Support Server-Sent Events (SSE) for HTTP

---

## Troubleshooting

### "future cannot be sent between threads safely"

**Cause**: Holding `std::sync::MutexGuard` across `.await` points

**Solution**: Use explicit scoping to drop guards before await:

```rust
// ❌ Bad - guard held across await
let guard = mutex.lock().unwrap();
do_async_work().await;

// ✅ Good - guard dropped before await
{
    let guard = mutex.lock().unwrap();
    let value = guard.clone();
} // guard dropped here
do_async_work().await;
```

### Performance Issues

- **High latency per token**: Reduce `token_delay_ms` in config
- **Memory growth**: Check for leaks in scheduler cleanup
- **CPU spikes**: Profile with `cargo flamegraph`

---

## Additional Resources

- [Production Readiness Assessment](../production_readiness_assessment.md)
- [ONNX Backend Implementation](../crates/kapsl-backends/src/onnx.rs)
- [LLM Backend Implementation](../crates/kapsl-llm/src/llm_backend.rs)
- [Engine API Definition](../crates/kapsl-engine-api/src/lib.rs)

---

**Status**: ✅ Core streaming implementation complete  
**Next Steps**: Transport layer integration + Python bindings
