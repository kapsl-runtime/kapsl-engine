use super::ModelLoader;
use crate::tensor::{GptqTensor, QuantizedTensor};
use anyhow::{Context, Result};
use half::f16;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub struct GptqLoader;

impl ModelLoader for GptqLoader {
    fn load(&self, model_path: &Path) -> Result<HashMap<String, QuantizedTensor>> {
        let mut tensors = HashMap::new();

        // Find all safetensors files
        let entries = fs::read_dir(model_path).context("Failed to read model directory")?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let file_content = fs::read(&path)?; // unsafe for mmap potential, but here just read
                let safetensors = SafeTensors::deserialize(&file_content)?;

                // Iterate over tensors and group them by layer/name
                // This is a simplified logic; real GPTQ loading needs to group qweight, qzeros, scales
                // For P0, we assume we can identify them by suffix

                // Note: This is a placeholder for the complex logic of identifying which tensors belong together
                // In a real implementation, we'd need to parse the model structure.
                // Here we just look for specific patterns.

                for name in safetensors.names() {
                    if name.ends_with(".qweight") {
                        let base_name = name.trim_end_matches(".qweight");

                        // Try to find associated tensors
                        let qweight = safetensors.tensor(name)?;
                        let qzeros_name = format!("{}.qzeros", base_name);
                        let scales_name = format!("{}.scales", base_name);

                        if let (Ok(qzeros), Ok(scales)) = (
                            safetensors.tensor(&qzeros_name),
                            safetensors.tensor(&scales_name),
                        ) {
                            // Load data
                            // Note: safetensors gives u8 bytes, we need to cast
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

                            let tensor = QuantizedTensor::Gptq(GptqTensor {
                                qweight: Arc::new(qweight_data),
                                qzeros: Arc::new(qzeros_data),
                                scales: Arc::new(scales_data),
                                g_idx: None, // Optional, load if present
                                bias: None,
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
