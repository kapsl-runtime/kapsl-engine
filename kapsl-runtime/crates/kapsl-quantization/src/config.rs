use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMethod {
    Int8,
    Gptq,
    Awq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quantization_method: QuantizationMethod,
    pub bits: usize,
    pub group_size: usize,
    pub desc_act: bool, // For GPTQ
    pub sym: bool,      // Symmetric quantization
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_method: QuantizationMethod::Int8,
            bits: 8,
            group_size: 128,
            desc_act: false,
            sym: true,
        }
    }
}
