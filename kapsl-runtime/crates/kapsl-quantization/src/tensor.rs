use half::f16;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum QuantizedTensor {
    Gptq(GptqTensor),
    Awq(AwqTensor),
    Int8(Int8Tensor),
}

#[derive(Debug, Clone)]
pub struct GptqTensor {
    pub qweight: Arc<Vec<u32>>,
    pub qzeros: Arc<Vec<u32>>,
    pub scales: Arc<Vec<f16>>,
    pub g_idx: Option<Arc<Vec<u32>>>,
    pub bias: Option<Arc<Vec<f16>>>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct AwqTensor {
    pub qweight: Arc<Vec<u32>>,
    pub qzeros: Arc<Vec<u32>>,
    pub scales: Arc<Vec<f16>>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Int8Tensor {
    pub weight: Arc<Vec<i8>>,
    pub scale: f32,
    pub zero_point: i32,
    pub shape: Vec<usize>,
}
