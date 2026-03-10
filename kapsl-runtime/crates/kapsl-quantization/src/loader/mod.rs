use crate::tensor::QuantizedTensor;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub trait ModelLoader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>>;
}

pub mod awq;
pub mod gptq;
pub mod int8;

pub use awq::AwqLoader;
pub use gptq::GptqLoader;
pub use int8::Int8Loader;
