use super::ModelLoader;
use crate::tensor::{Int8Tensor, QuantizedTensor};
use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub struct Int8Loader;

impl ModelLoader for Int8Loader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>> {
        let mut tensors = HashMap::new();

        let entries = fs::read_dir(model_path).context("Failed to read model directory")?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let file_content = fs::read(&path)?;
                let safetensors = SafeTensors::deserialize(&file_content)?;

                for name in safetensors.names() {
                    // Assuming standard naming convention or metadata indicates int8
                    // For P0, we look for .weight and check dtype or metadata
                    // This is a simplified implementation
                    if name.ends_with(".weight") {
                        let tensor = safetensors.tensor(name)?;
                        if tensor.dtype() == safetensors::Dtype::I8 {
                            let data: Vec<i8> = tensor.data().iter().map(|&x| x as i8).collect();

                            // Look for scale and zero_point
                            let base_name = name.trim_end_matches(".weight");
                            let scale_name = format!("{}.scale", base_name);
                            let zp_name = format!("{}.zero_point", base_name);

                            let scale = if let Ok(s) = safetensors.tensor(&scale_name) {
                                f32::from_le_bytes(s.data().try_into().unwrap_or([0; 4]))
                            } else {
                                1.0
                            };

                            let zero_point = if let Ok(z) = safetensors.tensor(&zp_name) {
                                i32::from_le_bytes(z.data().try_into().unwrap_or([0; 4]))
                            } else {
                                0
                            };

                            let q_tensor = QuantizedTensor::Int8(Int8Tensor {
                                weight: Arc::new(data),
                                scale,
                                zero_point,
                                shape: tensor.shape().to_vec(),
                            });

                            tensors.insert(base_name.to_string(), q_tensor);
                        }
                    }
                }
            }
        }

        Ok(tensors)
    }
}
