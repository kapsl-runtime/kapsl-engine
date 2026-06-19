use super::*;

#[test]
fn test_state_dir_namespaces_runtime_state_paths() {
    let state_dir = PathBuf::from("state");
    let args = Args {
        model: vec![],
        transport: "socket".to_string(),
        socket: "dummy.sock".to_string(),
        bind: "127.0.0.1".to_string(),
        port: 9096,
        batch_size: 4,
        scheduler_queue_size: 256,
        scheduler_max_micro_batch: 4,
        scheduler_queue_delay_ms: 2,
        performance_profile: PerformanceProfile::Standard,
        metrics_port: 9095,
        http_bind: "127.0.0.1".to_string(),
        state_dir: Some(state_dir.clone()),
        topology: "data-parallel".to_string(),
        tp_degree: 1,
        worker: false,
        worker_model_id: None,
        onnx_memory_pattern: None,
        onnx_disable_cpu_mem_arena: None,
        onnx_session_buckets: None,
        onnx_bucket_dim_granularity: None,
        onnx_bucket_max_dims: None,
        onnx_peak_concurrency_hint: None,
        onnx_model_tuning: Vec::new(),
        shm_size_mb: None,
        kv_compression_bits: Some(3 as u8),
    };

    let layout = resolve_runtime_state_layout(&args);
    assert_eq!(layout.rag_root, state_dir.join("rag-data"));
    assert_eq!(layout.extensions_root, state_dir.join("extensions"));
    assert_eq!(
        layout.extensions_config_root,
        state_dir.join("extensions-config")
    );
    assert_eq!(
        layout.auth_store_path,
        state_dir.join(DEFAULT_AUTH_STORE_FILENAME)
    );
}
