use crate::attention::CpuAttention;
use crate::mlp::CpuMlp;
use kapsl_hal::kernel::{AttentionKernel, KernelBackend, KernelBackendType, MlpKernel};

#[derive(Debug)]
pub struct CpuBackend;

impl KernelBackend for CpuBackend {
    fn backend_type(&self) -> KernelBackendType {
        KernelBackendType::Cpu
    }

    fn attention(&self) -> Box<dyn AttentionKernel> {
        Box::new(CpuAttention)
    }

    fn mlp(&self) -> Box<dyn MlpKernel> {
        Box::new(CpuMlp)
    }
}

pub fn create_backend() -> Box<dyn KernelBackend> {
    // For P0, always return CPU backend
    // In the future, detect CUDA/ROCm and return appropriate backend
    log::info!("Creating CPU kernel backend (GPU kernels not yet implemented)");
    Box::new(CpuBackend)
}
