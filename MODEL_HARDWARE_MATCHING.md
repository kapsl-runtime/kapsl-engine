# Solving Model-Hardware Mismatch with aimod-runtime

## The Problem

**Model-hardware mismatch** occurs when:
- Model requires GPU but deployed on CPU-only machine
- Model uses CUDA but hardware has AMD GPU (needs ROCm)
- Model uses Apple Metal but deployed on Linux
- Model optimized for TensorRT but TensorRT not available
- Model uses FP16 but hardware doesn't support it

## Current State ❌

The runtime currently **does not solve** this problem:

```rust
// Always uses CPU
let mut backend = OnnxBackend::new();
```

## Solution Architecture ✅

### 1. Enhanced Hardware Detection

**Extend `aimod-hal` to detect GPUs**:

```rust
// framework/aimod-runtime/crates/aimod-hal/src/lib.rs

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub cpu_cores: u32,
    pub total_memory: u64,
    pub os_type: String,
    
    // GPU detection
    pub has_cuda: bool,
    pub cuda_version: Option<String>,
    pub has_metal: bool,      // Apple Silicon
    pub has_rocm: bool,       // AMD
    pub has_tensorrt: bool,
    
    // Capabilities
    pub supports_fp16: bool,
    pub supports_int8: bool,
}

impl DeviceInfo {
    pub fn probe() -> Self {
        Self {
            // ... existing CPU detection
            
            // GPU detection
            has_cuda: Self::detect_cuda(),
            cuda_version: Self::get_cuda_version(),
            has_metal: cfg!(target_os = "macos") && Self::detect_metal(),
            has_tensorrt: Self::detect_tensorrt(),
            
            // Capabilities
            supports_fp16: Self::check_fp16_support(),
            supports_int8: true,  // Most modern hardware
        }
    }
    
    fn detect_cuda() -> bool {
        // Try to load CUDA libraries
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    
    fn detect_metal() -> bool {
        // Check for Metal framework on macOS
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            Command::new("system_profiler")
                .args(&["SPDisplaysDataType"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        }
        #[cfg(not(target_os = "macos"))]
        false
    }
}
```

### 2. Enhanced Manifest with Requirements

**Update manifest to declare requirements**:

```rust
// framework/aimod-runtime/crates/aimod-core/src/manifest.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub project_name: String,
    pub framework: String,
    pub version: String,
    pub created_at: String,
    pub model_file: String,
    
    // Hardware requirements
    #[serde(default)]
    pub hardware_requirements: HardwareRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareRequirements {
    // Preferred execution provider
    pub preferred_provider: Option<String>,  // "cuda", "cpu", "tensorrt", "metal"
    
    // Fallback chain
    pub fallback_providers: Vec<String>,  // ["cuda", "cpu"]
    
    // Minimum requirements
    pub min_memory_mb: Option<u64>,
    pub required_precision: Option<String>,  // "fp32", "fp16", "int8"
    
    // Optional optimizations
    pub optimized_for: Vec<String>,  // ["cuda_11.8", "tensorrt_8.6"]
}
```

**Python packaging creates this**:

```python
# framework/pip-module/testing-models/simple-pytorch-example/package/main.py

metadata = {
    "project_name": project_name,
    "framework": framework,
    "version": version,
    "created_at": datetime.now().isoformat(),
    "model_file": "model.onnx",
    
    # NEW: Hardware requirements
    "hardware_requirements": {
        "preferred_provider": "cuda",      # Prefer GPU
        "fallback_providers": ["cpu"],     # Fallback to CPU
        "min_memory_mb": 2048,             # 2GB minimum
        "required_precision": "fp32",
        "optimized_for": ["cuda_11.8"]
    }
}
```

### 3. Smart Backend Selection

**Create backend factory with hardware matching**:

```rust
// framework/aimod-runtime/crates/aimod-backends/src/factory.rs

pub struct BackendFactory;

impl BackendFactory {
    pub fn create_best_backend(
        manifest: &Manifest,
        device_info: &DeviceInfo,
    ) -> Result<Box<dyn Engine>, String> {
        let requirements = &manifest.hardware_requirements;
        
        // Try preferred provider first
        if let Some(preferred) = &requirements.preferred_provider {
            if let Ok(backend) = Self::try_create(preferred, device_info) {
                println!("✓ Using preferred provider: {}", preferred);
                return Ok(backend);
            } else {
                println!("⚠ Preferred provider '{}' not available", preferred);
            }
        }
        
        // Try fallback providers
        for provider in &requirements.fallback_providers {
            if let Ok(backend) = Self::try_create(provider, device_info) {
                println!("✓ Using fallback provider: {}", provider);
                return Ok(backend);
            }
        }
        
        // Last resort: CPU
        println!("⚠ Falling back to CPU");
        Ok(Box::new(OnnxBackend::new_cpu()))
    }
    
    fn try_create(
        provider: &str,
        device_info: &DeviceInfo,
    ) -> Result<Box<dyn Engine>, String> {
        match provider {
            "cuda" => {
                if !device_info.has_cuda {
                    return Err("CUDA not available".to_string());
                }
                Ok(Box::new(OnnxBackend::new_cuda()?))
            }
            
            "tensorrt" => {
                if !device_info.has_tensorrt {
                    return Err("TensorRT not available".to_string());
                }
                Ok(Box::new(TensorRtBackend::new()?))
            }
            
            "metal" => {
                if !device_info.has_metal {
                    return Err("Metal not available".to_string());
                }
                Ok(Box::new(MetalBackend::new()?))
            }
            
            "cpu" => Ok(Box::new(OnnxBackend::new_cpu())),
            
            _ => Err(format!("Unknown provider: {}", provider))
        }
    }
}
```

### 4. Enhanced ONNX Backend with Execution Providers

**Update `OnnxBackend` to support multiple execution providers**:

```rust
// framework/aimod-runtime/crates/aimod-backends/src/onnx.rs

use ort::execution_providers::{CUDAExecutionProvider, CPUExecutionProvider};

impl OnnxBackend {
    pub fn new_cpu() -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
            provider: ExecutionProvider::CPU,
        }
    }
    
    pub fn new_cuda() -> Result<Self, String> {
        // Check CUDA availability
        if !CUDAExecutionProvider::is_available() {
            return Err("CUDA not available".to_string());
        }
        
        Ok(Self {
            session: Arc::new(Mutex::new(None)),
            provider: ExecutionProvider::CUDA,
        })
    }
    
    fn load(&mut self, model_path: &Path) -> Result<(), EngineError> {
        let session = match self.provider {
            ExecutionProvider::CUDA => {
                Session::builder()?
                    .with_execution_providers([
                        CUDAExecutionProvider::default().build(),
                        CPUExecutionProvider::default().build(), // Fallback
                    ])?
                    .commit_from_file(model_path)?
            }
            ExecutionProvider::CPU => {
                Session::builder()?
                    .with_execution_providers([
                        CPUExecutionProvider::default().build(),
                    ])?
                    .commit_from_file(model_path)?
            }
        };
        
        // Store session
        let mut guard = self.session.lock().unwrap();
        *guard = Some(session);
        Ok(())
    }
}
```

### 5. Updated CLI with Smart Selection

```rust
// framework/aimod-runtime/crates/aimod-cli/src/main.rs

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("Starting aimod-runtime...");
    
    // 1. Hardware Probe
    let device_info = DeviceInfo::probe();
    println!("Device Info: {:?}", device_info);

    // 2. Load Package
    let loader = PackageLoader::load(&args.model)?;
    println!("Manifest: {:?}", loader.manifest);
    
    // 3. Validate Requirements
    validate_requirements(&loader.manifest, &device_info)?;
    
    // 4. Select Best Backend (SMART SELECTION)
    let mut backend = BackendFactory::create_best_backend(
        &loader.manifest,
        &device_info,
    )?;
    
    let model_path = loader.get_model_path();
    backend.load(&model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;
    
    println!("✓ Backend initialized and model loaded.");

    // ... rest of code
}

fn validate_requirements(
    manifest: &Manifest,
    device_info: &DeviceInfo,
) -> Result<(), String> {
    let req = &manifest.hardware_requirements;
    
    // Check memory
    if let Some(min_mem) = req.min_memory_mb {
        let available_mb = device_info.total_memory / 1024;
        if available_mb < min_mem {
            return Err(format!(
                "Insufficient memory: need {}MB, have {}MB",
                min_mem, available_mb
            ));
        }
    }
    
    // Check if ANY provider is available
    let has_any_provider = req.preferred_provider
        .as_ref()
        .map(|p| is_provider_available(p, device_info))
        .unwrap_or(false)
        || req.fallback_providers
            .iter()
            .any(|p| is_provider_available(p, device_info));
    
    if !has_any_provider {
        println!("⚠ WARNING: No preferred providers available, using CPU");
    }
    
    Ok(())
}
```

## Complete Flow

```
┌────────────────────────────────────────────────────────────┐
│ 1. User packages model with requirements                 │
│    idx package                                            │
│    └─> metadata.json includes hardware_requirements      │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 2. Deployment: aimod-cli starts                          │
│    aimod-cli --model model.aimod                         │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 3. Hardware Detection                                     │
│    DeviceInfo::probe()                                    │
│    ├─ CPU: 8 cores, 16GB RAM                             │
│    ├─ CUDA: ✓ Available (v11.8)                          │
│    ├─ TensorRT: ✗ Not available                          │
│    └─ Metal: ✗ Not available (Linux)                     │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 4. Load Manifest & Requirements                          │
│    preferred_provider: "tensorrt"                         │
│    fallback_providers: ["cuda", "cpu"]                    │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 5. Smart Backend Selection                               │
│    ├─ Try TensorRT: ✗ Not available                      │
│    ├─ Try CUDA: ✓ Available!                             │
│    └─ Create OnnxBackend with CUDAExecutionProvider      │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│ 6. Load Model with Selected Backend                      │
│    ✓ Model loaded on CUDA                                │
│    ✓ Ready to serve                                      │
└────────────────────────────────────────────────────────────┘
```

## Benefits

✅ **Automatic fallback**: CUDA → CPU if GPU not available  
✅ **Early validation**: Fail fast if requirements can't be met  
✅ **Portable**: Same `.aimod` works on GPU and CPU machines  
✅ **Optimal performance**: Uses best available hardware  
✅ **Clear errors**: "CUDA required but not available"  

## Usage Example

```bash
# Development (CPU laptop)
aimod-cli --model model.aimod
# → Falls back to CPU, works fine

# Production (GPU server)
aimod-cli --model model.aimod
# → Uses CUDA, faster performance

# Same .aimod package, different hardware, auto-adapts!
```
