#[cfg(test)]
mod tests {
    use super::super::*;
    use half::f16;
    use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, NamedTensor, TensorDtype};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    fn f64_bytes(values: &[f64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    fn f16_bytes(values: &[f16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_bits().to_ne_bytes().to_vec())
            .collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| v.to_ne_bytes().to_vec())
            .collect()
    }

    #[test]
    fn test_validate_float32_success() {
        let values = vec![0.0f32, 1.0f32, -2.5f32, 3.25f32];
        let packet = BinaryTensorPacket {
            shape: vec![2, 2],
            dtype: TensorDtype::Float32,
            data: f32_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((shape, PreparedInput::F32(vec))) => {
                assert_eq!(shape, vec![2, 2]);
                assert_eq!(vec.len(), 4);
                for (a, b) in vec.iter().zip(values.iter()) {
                    assert_eq!(a, b);
                }
            }
            other => panic!("Expected prepared f32 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_float32_scalar_empty_shape() {
        let values = vec![42.0f32];
        let packet = BinaryTensorPacket {
            shape: vec![],
            dtype: TensorDtype::Float32,
            data: f32_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((shape, PreparedInput::F32(vec))) => {
                assert!(shape.is_empty());
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared f32 scalar input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_float32_bad_length() {
        // Provide only 3 floats for a 2x2 shape (needs 4)
        let values = vec![0.0f32, 1.0f32, -2.5f32];
        let packet = BinaryTensorPacket {
            shape: vec![2, 2],
            dtype: TensorDtype::Float32,
            data: f32_bytes(&values),
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Data length mismatch"));
        } else {
            panic!("Expected InvalidInput error for bad length");
        }
    }

    #[test]
    fn test_validate_int32_bad_length() {
        let values = vec![1i32];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Int32,
            data: i32_bytes(&values),
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Data length mismatch"));
        } else {
            panic!("Expected InvalidInput error for bad int32 length");
        }
    }

    #[test]
    fn test_validate_unsupported_dtype() {
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Utf8,
            data: vec![b'a', b'b'],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Unsupported dtype"));
        } else {
            panic!("Expected InvalidInput for unsupported dtype");
        }
    }

    #[test]
    fn test_validate_invalid_shape_dimension_zero() {
        let packet = BinaryTensorPacket {
            shape: vec![0],
            dtype: TensorDtype::Float32,
            data: vec![],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Invalid shape dimension"));
        } else {
            panic!("Expected InvalidInput error for invalid shape");
        }
    }

    #[test]
    fn test_validate_invalid_shape_dimension_negative() {
        let packet = BinaryTensorPacket {
            shape: vec![-1],
            dtype: TensorDtype::Float32,
            data: vec![],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Invalid shape dimension"));
        } else {
            panic!("Expected InvalidInput error for invalid shape");
        }
    }

    #[test]
    fn test_validate_shape_multiplication_overflow() {
        let packet = BinaryTensorPacket {
            shape: vec![i64::MAX, i64::MAX],
            dtype: TensorDtype::Float32,
            data: vec![],
        };

        let res = validate_and_prepare_input(&packet);
        assert!(res.is_err());
        if let Err(EngineError::InvalidInput { message, .. }) = res {
            assert!(message.contains("Shape multiplication overflow"));
        } else {
            panic!("Expected InvalidInput error for overflow");
        }
    }

    #[test]
    fn test_validate_int32_success() {
        let values = vec![1i32, 2i32, 3i32];
        let packet = BinaryTensorPacket {
            shape: vec![3],
            dtype: TensorDtype::Int32,
            data: i32_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::I32(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared i32 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_float16_success() {
        let values = vec![f16::from_f32(1.5), f16::from_f32(-2.25)];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Float16,
            data: f16_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::F16(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared f16 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_float64_success() {
        let values = vec![1.0f64, -3.5f64];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Float64,
            data: f64_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::F64(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared f64 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_int64_success() {
        let values = vec![10i64, -20i64];
        let packet = BinaryTensorPacket {
            shape: vec![2],
            dtype: TensorDtype::Int64,
            data: i64_bytes(&values),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::I64(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared i64 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_uint8_success() {
        let values = vec![1u8, 200u8, 17u8];
        let packet = BinaryTensorPacket {
            shape: vec![3],
            dtype: TensorDtype::Uint8,
            data: values.clone(),
        };

        match validate_and_prepare_input(&packet) {
            Ok((_shape, PreparedInput::U8(vec))) => {
                assert_eq!(vec, values);
            }
            other => panic!("Expected prepared u8 input, got: {:?}", other),
        }
    }

    #[test]
    fn test_infer_returns_model_not_loaded() {
        let backend = OnnxBackend::new_cpu();
        let request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1],
                dtype: TensorDtype::Float32,
                data: f32_bytes(&[0.0f32]),
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let res = backend.infer(&request);
        assert!(matches!(res, Err(EngineError::ModelNotLoaded)));
    }

    #[test]
    fn test_builder_device_id_validation() {
        let res = OnnxBackend::builder().with_device_id(-1);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "Device ID must be non-negative");

        let res = OnnxBackend::builder().with_device_id(0);
        assert!(res.is_ok());
    }

    #[test]
    fn test_builder_construction_defaults() {
        let backend = OnnxBackend::builder().build();
        // Check defaults: CPU, Level3, Device 0
        match backend.provider {
            ExecutionProvider::CPU => (),
            _ => panic!("Expected CPU provider by default"),
        }
        assert_eq!(backend.device_id, 0);
        // Optimization level 3 maps to 3
        assert_eq!(backend.optimization_level, 3);
    }

    #[test]
    fn test_builder_all_optimization_level_mapping() {
        let backend = OnnxBackend::builder()
            .with_optimization_level(GraphOptimizationLevel::All)
            .build();
        assert_eq!(backend.optimization_level, 4);
        assert_eq!(backend.get_opt_level(), GraphOptimizationLevel::All);
    }

    #[test]
    fn test_new_cuda_builder_settings() {
        let backend = OnnxBackend::new_cuda_with_optimization(GraphOptimizationLevel::Level1, 2)
            .expect("builder should accept device id");
        match backend.provider {
            ExecutionProvider::CUDA => (),
            _ => panic!("Expected CUDA provider"),
        }
        assert_eq!(backend.device_id, 2);
        assert_eq!(backend.optimization_level, 1);
    }

    #[test]
    fn test_new_cuda_negative_device_id_rejected() {
        let res = OnnxBackend::new_cuda(-1);
        if let Err(msg) = res {
            assert_eq!(msg, "Device ID must be non-negative");
        } else {
            panic!("Expected error for negative device id");
        }
    }

    #[test]
    fn test_duplicate_additional_input_name_rejected() {
        let additional_inputs = vec![
            NamedTensor {
                name: "dup".to_string(),
                tensor: BinaryTensorPacket {
                    shape: vec![1],
                    dtype: TensorDtype::Uint8,
                    data: vec![1],
                },
            },
            NamedTensor {
                name: "dup".to_string(),
                tensor: BinaryTensorPacket {
                    shape: vec![1],
                    dtype: TensorDtype::Uint8,
                    data: vec![2],
                },
            },
        ];

        let result = ensure_unique_additional_input_names(&additional_inputs);
        assert!(matches!(result, Err(EngineError::InvalidInput { .. })));
    }
}
