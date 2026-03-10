use super::ModelLoader;
use crate::tensor::{AwqTensor, QuantizedTensor};
use anyhow::{Context, Result};
use half::f16;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub struct AwqLoader;

impl ModelLoader for AwqLoader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>> {
        let mut tensors = HashMap::new();

        // Find all safetensors files
        let entries = fs::read_dir(model_path).context("Failed to read model directory")?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let file_content = fs::read(&path)?;
                let safetensors = SafeTensors::deserialize(&file_content)?;

                for name in safetensors.names() {
                    if name.ends_with(".qweight") {
                        let base_name = name.trim_end_matches(".qweight");

                        let qweight = safetensors.tensor(name)?;
                        let qzeros_name = format!("{}.qzeros", base_name);
                        let scales_name = format!("{}.scales", base_name);

                        if let (Ok(qzeros), Ok(scales)) = (
                            safetensors.tensor(&qzeros_name),
                            safetensors.tensor(&scales_name),
                        ) {
                            let qweight_data: Vec<u32> = qweight
                                .data()
                                .chunks_exact(4)
                                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            let qzeros_data: Vec<u32> = qzeros
                                .data()
                                .chunks_exact(4)
                                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            let scales_data: Vec<f16> = scales
                                .data()
                                .chunks_exact(2)
                                .map(|c| f16::from_le_bytes(c.try_into().unwrap()))
                                .collect();

                            let tensor = QuantizedTensor::Awq(AwqTensor {
                                qweight: Arc::new(qweight_data),
                                qzeros: Arc::new(qzeros_data),
                                scales: Arc::new(scales_data),
                                shape: qweight.shape().to_vec(),
                            });

                            tensors.insert(base_name.to_string(), tensor);
                        }
                    }
                }
            }
        }

        Ok(tensors)
    }
}
