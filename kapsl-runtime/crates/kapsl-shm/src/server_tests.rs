#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::allocator::SimpleShmAllocator;
    use async_trait::async_trait;
    use kapsl_engine_api::{
        BinaryTensorPacket, EngineError, EngineMetrics, EngineModelInfo, EngineStream, TensorDtype,
    };
    use kapsl_scheduler::Priority;
    use kapsl_transport::{RequestMetadata, ResponseMetadata};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    struct OkScheduler {
        output: BinaryTensorPacket,
    }

    #[async_trait]
    impl ReplicaScheduler for OkScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            (0, 0)
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn get_metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn infer(
            &self,
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Ok(self.output.clone())
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<EngineStream, EngineError> {
            Err(EngineError::backend("streaming not supported in tests"))
        }
    }

    struct ErrScheduler;

    #[async_trait]
    impl ReplicaScheduler for ErrScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            (0, 0)
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn get_metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn infer(
            &self,
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Err(EngineError::backend("boom"))
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<EngineStream, EngineError> {
            Err(EngineError::backend("streaming not supported in tests"))
        }
    }

    struct MetadataScheduler {
        info: EngineModelInfo,
    }

    #[async_trait]
    impl ReplicaScheduler for MetadataScheduler {
        fn get_queue_depth(&self) -> (usize, usize) {
            (0, 0)
        }

        fn is_healthy(&self) -> bool {
            true
        }

        fn get_metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        fn model_info(&self) -> Option<EngineModelInfo> {
            Some(self.info.clone())
        }

        async fn infer(
            &self,
            _request: &InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<BinaryTensorPacket, EngineError> {
            Err(EngineError::backend("not used"))
        }

        async fn infer_stream(
            &self,
            _request: InferenceRequest,
            _priority: Priority,
            _force_cpu: bool,
        ) -> Result<EngineStream, EngineError> {
            Err(EngineError::backend("not used"))
        }
    }

    fn alloc_buffer(size_bytes: usize) -> (Vec<u64>, *mut u8) {
        let len = size_bytes.div_ceil(8);
        let mut backing = vec![0u64; len];
        let base = backing.as_mut_ptr() as *mut u8;
        (backing, base)
    }

    fn test_allocator() -> SimpleShmAllocator {
        SimpleShmAllocator::new_with_ttl(128 * 1024, 10_000_000, 100, Duration::from_secs(60))
    }

    #[test]
    fn test_tensor_roundtrip() {
        let (_backing, base) = alloc_buffer(1024);

        let packet = BinaryTensorPacket {
            shape: vec![1, 2],
            dtype: TensorDtype::Int32,
            data: vec![1u8, 0, 0, 0, 2, 0, 0, 0],
        };

        unsafe {
            write_tensor_to_shm(base, 0, &packet);
            let read_back = read_tensor_from_shm(base, 0);
            assert_eq!(read_back.shape, packet.shape);
            assert_eq!(read_back.dtype, packet.dtype);
            assert_eq!(read_back.data, packet.data);
        }
    }

    #[test]
    fn test_error_roundtrip() {
        let (_backing, base) = alloc_buffer(25 * 1024 * 1024);
        let error_msg = "failed request";
        let allocator = test_allocator();

        let offset = write_error_to_shm(base, &allocator, None, error_msg).expect("error slot");
        unsafe {
            let len = *(base.add(offset) as *const u64) as usize;
            assert_eq!(len, error_msg.len());
            let bytes =
                std::slice::from_raw_parts(base.add(offset + std::mem::size_of::<u64>()), len);
            assert_eq!(bytes, error_msg.as_bytes());
        }
    }

    #[tokio::test]
    async fn test_request_success_flow() {
        let (_backing, base) = alloc_buffer(25 * 1024 * 1024);
        let input = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 128, 63],
        };

        let output = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 0, 64],
        };

        let scheduler = OkScheduler {
            output: output.clone(),
        };
        let allocator = test_allocator();
        unsafe {
            write_tensor_to_shm(base, 0, &input);
        }

        let request = ShmRequest {
            metadata: RequestMetadata::new(7, 1, 0, false),
            tensor_offset: 0,
            tensor_size: (std::mem::size_of::<TensorHeader>() + input.data.len()) as u64,
        };

        let tensor = unsafe { read_tensor_from_shm(base, request.tensor_offset as usize) };
        let request_obj = InferenceRequest {
            input: tensor,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let result = scheduler
            .infer(&request_obj, Priority::LatencyCritical, false)
            .await
            .unwrap();

        let result_size = std::mem::size_of::<TensorHeader>() + result.data.len();
        let result_offset = allocator.try_allocate(result_size).expect("result slot");
        unsafe {
            write_tensor_to_shm(base, result_offset, &result);
        }

        let response = ShmResponse {
            metadata: ResponseMetadata::success(request.metadata.request_id, 123),
            result_offset: result_offset as u64,
            result_size: (std::mem::size_of::<TensorHeader>() + result.data.len()) as u64,
            error_offset: 0,
        };

        let roundtrip = unsafe { read_tensor_from_shm(base, response.result_offset as usize) };
        assert!(response.metadata.is_success());
        assert_eq!(roundtrip.data, output.data);
        assert_eq!(roundtrip.dtype, output.dtype);
    }

    #[tokio::test]
    async fn test_request_error_flow() {
        let (_backing, base) = alloc_buffer(25 * 1024 * 1024);
        let input = BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: vec![0, 0, 128, 63],
        };

        let scheduler = ErrScheduler;
        let allocator = test_allocator();
        unsafe {
            write_tensor_to_shm(base, 0, &input);
        }

        let request = ShmRequest {
            metadata: RequestMetadata::new(9, 1, 1, true),
            tensor_offset: 0,
            tensor_size: (std::mem::size_of::<TensorHeader>() + input.data.len()) as u64,
        };

        let tensor = unsafe { read_tensor_from_shm(base, request.tensor_offset as usize) };
        let request_obj = InferenceRequest {
            input: tensor,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let result = scheduler
            .infer(&request_obj, Priority::Throughput, true)
            .await;
        assert!(result.is_err());

        let error_offset = write_error_to_shm(base, &allocator, None, "boom").expect("error slot");
        let response = ShmResponse {
            metadata: ResponseMetadata::error(request.metadata.request_id, 77),
            result_offset: 0,
            result_size: 0,
            error_offset: error_offset as u64,
        };

        let err_bytes = unsafe {
            let len = *(base.add(error_offset) as *const u64) as usize;
            std::slice::from_raw_parts(base.add(error_offset + std::mem::size_of::<u64>()), len)
        };
        assert!(!response.metadata.is_success());
        assert_eq!(err_bytes, b"boom");
    }

    #[test]
    fn test_model_aware_budget_derivation_uses_shapes_and_dtypes() {
        let info = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 1000]],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec!["float32".to_string()],
            framework: Some("onnx".to_string()),
            model_version: Some("1.0".to_string()),
            peak_concurrency: Some(4),
        };

        let mut schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> = HashMap::new();
        schedulers.insert(1, Arc::new(MetadataScheduler { info }));

        let budgets = derive_model_aware_class_budgets(&schedulers, 512 * 1024 * 1024);
        assert!(!budgets.is_empty());
        assert!(budgets.iter().any(|b| b.slot_size >= 1024 * 1024));
    }

    #[test]
    fn test_model_aware_budget_derivation_scales_weight_by_peak_concurrency() {
        let low = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec![],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec![],
            framework: Some("onnx".to_string()),
            model_version: Some("1.0".to_string()),
            peak_concurrency: Some(1),
        };

        let high = EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec![],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec![],
            framework: Some("onnx".to_string()),
            model_version: Some("1.0".to_string()),
            peak_concurrency: Some(8),
        };

        let mut schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>> = HashMap::new();
        schedulers.insert(1, Arc::new(MetadataScheduler { info: low }));
        schedulers.insert(2, Arc::new(MetadataScheduler { info: high }));

        let budgets = derive_model_aware_class_budgets(&schedulers, 512 * 1024 * 1024);
        let max_weight = budgets
            .iter()
            .map(|budget| budget.weight)
            .max()
            .unwrap_or(0);
        assert!(max_weight >= 9);
    }
}
