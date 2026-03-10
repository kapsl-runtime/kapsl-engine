#[cfg(test)]
mod tests {
    use super::super::{ShmError, ShmManager};

    #[test]
    fn test_offsets_within_bounds() {
        let name = format!("/test_shm_offsets_{}", std::process::id());
        let size = 2 * 1024 * 1024;

        let manager = match ShmManager::create(&name, size) {
            Ok(manager) => manager,
            Err(ShmError::ShmemError(shared_memory::ShmemError::MapCreateFailed(_))) => {
                eprintln!("Skipping shared memory offsets test (mapping creation failed)");
                return;
            }
            Err(err) => panic!("Failed to create shared memory: {}", err),
        };

        let request_offset = manager.request_queue_offset();
        let response_offset = manager.response_queue_offset();
        let tensor_offset = manager.tensor_pool_offset();
        let max_tensor_size = manager.max_tensor_size();

        assert!(request_offset < response_offset);
        assert!(response_offset < tensor_offset);
        assert!(tensor_offset < size);
        assert!(tensor_offset + max_tensor_size <= size);
    }
}
