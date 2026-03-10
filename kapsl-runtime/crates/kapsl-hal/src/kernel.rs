use std::fmt::Debug;
use thiserror::Error;

use crate::tensor::{TensorView, TensorViewMut};

#[derive(Error, Debug)]
pub enum KernelError {
    #[error("Backend error: {0}")]
    BackendError(String),
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub scale: Option<f32>,
    pub causal: bool,
}

impl AttentionConfig {
    pub fn scale_for(&self, head_dim: usize) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt())
    }
}

pub trait AttentionKernel: Send + Sync + Debug {
    fn flash_attention_v2(
        &self,
        query: &TensorView<f32>,
        key: &TensorView<f32>,
        value: &TensorView<f32>,
        output: &mut TensorViewMut<f32>,
        config: AttentionConfig,
    ) -> Result<(), KernelError>;

    fn paged_attention_v1(
        &self,
        query: &TensorView<f32>,
        key: &TensorView<f32>,
        value: &TensorView<f32>,
        output: &mut TensorViewMut<f32>,
        config: AttentionConfig,
    ) -> Result<(), KernelError>;
}

pub trait MlpKernel: Send + Sync + Debug {
    fn fused_swiglu(
        &self,
        gate: &TensorView<f32>,
        up: &TensorView<f32>,
        out: &mut TensorViewMut<f32>,
    ) -> Result<(), KernelError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackendType {
    Cpu,
    Cuda,
    Rocm,
}

pub trait KernelBackend: Send + Sync + Debug {
    fn backend_type(&self) -> KernelBackendType;

    fn attention(&self) -> Box<dyn AttentionKernel>;
    fn mlp(&self) -> Box<dyn MlpKernel>;
}
