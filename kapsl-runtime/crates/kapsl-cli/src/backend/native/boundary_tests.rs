//! Load a real cdylib built against the published ABI. The only substitution
//! is the physical pool; loading, initialization, callbacks and dispatch are
//! the production host path. No CUDA driver, GPU or performance gates.
use super::*;
use allocator::tests::FakePool;
use futures::StreamExt;
use kapsl_engine_api::{MemoryAllocationSource, MemoryDomain};

fn pack() -> Arc<ActiveNativePack> {
    static PACK: OnceLock<Arc<ActiveNativePack>> = OnceLock::new();
    Arc::clone(PACK.get_or_init(|| {
        let executable = std::env::current_exe().unwrap();
        let root = executable.parent().unwrap();
        let filename = format!(
            "{}kapsl_native_test_adapter{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        );
        assert!(
            root.join(&filename).is_file(),
            "Cargo must build the native adapter cdylib at {}",
            root.join(&filename).display()
        );
        let mut manifest = tests::test_manifest("cuda", kapsl_native_test_adapter::CAPABILITIES);
        manifest.backend = "fake-native".into();
        manifest.entrypoint = filename;
        load_native_backend_pack(&manifest, root).unwrap()
    }))
}

fn model(mode: &str) -> Manifest {
    serde_json::from_value(serde_json::json!({
        "project_name": mode, "framework": "onnx", "version": "1.0",
        "created_at": "2026-09-10", "model_file": "fake.onnx"
    }))
    .unwrap()
}

fn engine(
    pool: &Arc<FakePool>,
    owner: (u32, u32),
    mode: &str,
) -> Result<NativePackedEngine, String> {
    engine_with_options(pool, owner, mode, &serde_json::Map::new())
}

fn engine_with_options(
    pool: &Arc<FakePool>,
    owner: (u32, u32),
    mode: &str,
    options: &serde_json::Map<String, serde_json::Value>,
) -> Result<NativePackedEngine, String> {
    let host = pool.host(owner.0, owner.1, 2048);
    NativePackInstance::initialize_with_host(
        pack(),
        &model(mode),
        "cuda",
        host,
        0,
        owner.0,
        owner.1,
        options,
    )
    .map(|instance| NativePackedEngine { instance })
}

fn request() -> InferenceRequest {
    InferenceRequest::new(BinaryTensorPacket::new(vec![1], TensorDtype::Uint8, vec![7]).unwrap())
}

fn probe(instance: &NativePackInstance, counter: u32) -> u64 {
    unsafe {
        let function: libloading::Symbol<'_, unsafe extern "C" fn(*mut c_void, u32) -> u64> =
            instance.pack.library.get(b"kapsl_test_probe_v1\0").unwrap();
        function(instance.handle, counter)
    }
}

async fn wait_started(instance: &NativePackInstance, count: u64) {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while probe(instance, 0) < count {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("fake adapter never entered inference");
}

#[tokio::test]
async fn packaged_assets_reach_the_adapter_after_source_and_extraction_cleanup() {
    use crate::features::packaging::{
        create_kapsl_package_from_context, ContextPackageRequest, TempDirGuard,
    };
    use kapsl_core::PackageLoader;

    let temp = TempDirGuard::new("native-package-assets").unwrap();
    let context = temp.path().join("source");
    std::fs::create_dir_all(context.join("graphs")).unwrap();
    std::fs::create_dir_all(context.join("assets")).unwrap();
    for (relative, contents) in [
        ("graphs/model.onnx", "fake model"),
        ("vocab.json", "root vocabulary"),
        ("graphs/vocab.json", "graph vocabulary"),
        ("assets/config.json", "model configuration"),
    ] {
        std::fs::write(context.join(relative), contents).unwrap();
    }
    let mut manifest = model("package-assets");
    manifest.model_file = "graphs/model.onnx".into();
    std::fs::write(
        context.join("metadata.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let package = temp.path().join("model.aimod");
    create_kapsl_package_from_context(ContextPackageRequest {
        output_override: Some(&package),
        ..ContextPackageRequest::new(&context)
    })
    .unwrap();
    let loader = PackageLoader::load(&package).unwrap();
    let model_path = loader.get_model_path();
    let extracted = loader.extracted_path.clone();
    let pool = Arc::new(FakePool::default());
    let instance = NativePackInstance::initialize_with_host(
        pack(),
        &loader.manifest,
        "cuda",
        pool.host(7, 2, 2048),
        0,
        7,
        2,
        &serde_json::Map::new(),
    )
    .unwrap();
    drop(loader);
    assert!(!extracted.exists());
    std::fs::remove_dir_all(&context).unwrap();
    std::fs::remove_file(&package).unwrap();

    let mut engine = NativePackedEngine { instance };
    // The loaded cdylib reads the model and root/nested assets from the exact
    // path passed through the production native ABI, using only the cache.
    for _ in 0..2 {
        engine.load(&model_path).await.unwrap();
        assert_eq!(engine.infer(&request()).unwrap().data, vec![42]);
        engine.unload();
        assert_eq!(pool.bytes((7, 2)), 0);
    }
}

#[test]
fn opaque_adapter_options_cross_the_loaded_native_boundary() {
    let pool = Arc::new(FakePool::default());
    let options = serde_json::json!({
        "vendor_config": { "strategy": "paged", "shape": [2, 4], "enabled": true },
        "extension": null,
    });
    let engine =
        engine_with_options(&pool, (7, 2), "normal", options.as_object().unwrap()).unwrap();
    let received: serde_json::Value = engine
        .instance
        .call_json_report(engine.instance.pack.api.model_info.unwrap())
        .unwrap();
    let received = &received["options"];
    for (key, value) in options.as_object().unwrap() {
        assert_eq!(&received[key], value);
    }
    assert_eq!(received["provider"], "cuda");
    assert_eq!(received["descriptor"]["backend"], "fake-native");
    assert_eq!(
        received["pack_root"],
        serde_json::json!(engine.instance.pack.root)
    );
    assert!(received.get("onnx_tuning").is_none());
}

#[test]
fn adapter_options_cannot_override_engine_owned_configuration() {
    let pool = Arc::new(FakePool::default());
    for key in [
        "provider",
        "accelerator_profile",
        "pack_version",
        "descriptor",
        "pack_root",
        "entrypoint",
    ] {
        let options = serde_json::Map::from_iter([(key.into(), serde_json::json!("override"))]);
        let error = engine_with_options(&pool, (7, 2), "initialize-allocates", &options)
            .err()
            .expect("engine-owned configuration must reject overrides");
        assert!(
            error.contains(&format!("adapter option `{key}` conflicts")),
            "{error}"
        );
        assert_eq!(pool.bytes((7, 2)), 0);
    }
}

#[tokio::test]
async fn loaded_adapter_exercises_reporting_batching_streaming_and_reload() {
    let pool = Arc::new(FakePool::default());
    let mut engine = engine(&pool, (7, 2), "normal").unwrap();
    assert_eq!(engine.instance.pack.descriptor["backend"], "fake-native");
    assert!(engine.infer(&request()).is_err());
    assert_eq!(
        engine
            .planned_memory(Path::new("fake.onnx"))
            .unwrap()
            .allocations[0]
            .bytes,
        256
    );
    assert_eq!(
        engine.planned_request_memory(&request()).allocations[0].bytes,
        64
    );
    assert!(engine.health_check().is_err());
    engine.load(Path::new("fake.onnx")).await.unwrap();
    assert!(engine.load(Path::new("fake.onnx")).await.is_err());
    engine.health_check().unwrap();
    assert_eq!(
        engine.model_info().unwrap().framework.as_deref(),
        Some("fake")
    );
    assert_eq!(engine.max_batch(), 4);
    assert!(!engine.self_batches());
    assert_eq!(
        engine.batching_policy().mode,
        BatchingMode::RequestCoalescing
    );
    assert_eq!(engine.infer(&request()).unwrap().data, [42]);
    let batch = engine.infer_batch(&[request(), request()]).unwrap();
    assert_eq!(batch.len(), 2);
    assert!(batch.iter().all(|output| output.data == [42]));
    let chunks: Vec<_> = engine.infer_stream(&request()).collect().await;
    assert_eq!(chunks.len(), 2);
    assert!(chunks.into_iter().all(|chunk| chunk.unwrap().data == [42]));
    let report = engine.actual_memory();
    assert_eq!(
        report.bytes_for_domain(&MemoryDomain::Cuda { device_id: 0 }),
        512
    );
    assert!(report
        .allocations
        .iter()
        .all(|row| row.source == MemoryAllocationSource::RuntimeManaged));
    assert!(report
        .allocations
        .iter()
        .any(|row| row.allocation_id.contains("scope-4-")
            && row.allocation_id.contains("requests-[")));
    let metrics = engine.metrics();
    assert_eq!(metrics.throughput, 3.0);
    assert_eq!(metrics.batch_size, 2);
    assert_eq!(metrics.memory_usage, 512);
    engine.instance.unload().unwrap();
    assert_eq!(pool.bytes((7, 2)), 0);
    assert!(engine.actual_memory().allocations.is_empty());
    assert!(engine.infer(&request()).is_err());
    engine.load(Path::new("fake.onnx")).await.unwrap();
    assert_eq!(engine.infer(&request()).unwrap().data, [42]);
    drop(engine);
    assert_eq!(pool.bytes((7, 2)), 0);
}

#[tokio::test]
async fn shared_pool_isolates_models_and_replicas_through_unload_and_bad_ownership() {
    let pool = Arc::new(FakePool::default());
    let mut first = engine(&pool, (7, 2), "invalid-scope").unwrap();
    let mut replica = engine(&pool, (7, 3), "normal").unwrap();
    let mut other = engine(&pool, (8, 2), "normal").unwrap();
    for engine in [&mut first, &mut replica, &mut other] {
        engine.load(Path::new("fake.onnx")).await.unwrap();
    }
    assert_eq!(first.infer(&request()).unwrap().data, [42]);
    assert_eq!(replica.infer(&request()).unwrap().data, [42]);
    assert_eq!(other.infer(&request()).unwrap().data, [42]);
    first.instance.unload().unwrap();
    assert_eq!(pool.bytes((7, 2)), 0);
    assert_eq!(pool.bytes((7, 3)), 320);
    assert_eq!(pool.bytes((8, 2)), 320);
    drop(replica);
    drop(other);
    assert_eq!(pool.bytes((7, 3)), 0);
    assert_eq!(pool.bytes((8, 2)), 0);
}

#[tokio::test]
async fn failure_paths_reclaim_only_after_the_adapter_releases_ownership() {
    let pool = Arc::new(FakePool::default());
    for mode in ["fail-initialize", "fail-initialize-null"] {
        assert!(engine(&pool, (7, 2), mode).is_err());
        assert_eq!(pool.bytes((7, 2)), 0, "{mode}");
    }
    for mode in ["fail-load", "fail-load-cleanup"] {
        let mut engine = engine(&pool, (7, 2), mode).unwrap();
        assert!(engine.load(Path::new("fake.onnx")).await.is_err());
        assert!(engine.infer(&request()).is_err());
        assert_eq!(
            pool.bytes((7, 2)),
            if mode == "fail-load" { 0 } else { 256 }
        );
        drop(engine);
        assert_eq!(pool.bytes((7, 2)), 0, "{mode}");
    }
    for mode in ["fail-infer", "leak-unload", "fail-unload"] {
        let mut engine = engine(&pool, (7, 2), mode).unwrap();
        engine.load(Path::new("fake.onnx")).await.unwrap();
        assert_eq!(engine.infer(&request()).is_err(), mode == "fail-infer");
        assert_eq!(pool.bytes((7, 2)), 320);
        if mode == "fail-unload" {
            assert!(engine.instance.unload().is_err());
            assert_eq!(pool.bytes((7, 2)), 320);
            assert!(engine.infer(&request()).is_err());
            assert!(engine.load(Path::new("fake.onnx")).await.is_err());
        }
        engine.instance.unload().unwrap();
        assert_eq!(pool.bytes((7, 2)), 0, "{mode}");
    }
}

#[tokio::test]
async fn synchronization_failure_blocks_reload_and_preserves_memory_reporting() {
    let pool = Arc::new(FakePool::default());
    let mut engine = engine(&pool, (7, 2), "leak-unload").unwrap();
    engine.load(Path::new("fake.onnx")).await.unwrap();
    pool.fail_sync(true);
    assert!(engine.instance.unload().is_err());
    assert_eq!(engine.actual_memory().allocations[0].bytes, 256);
    assert_eq!(engine.metrics().memory_usage, 256);
    assert!(engine.health_check().is_err());
    assert!(engine.load(Path::new("fake.onnx")).await.is_err());
    pool.fail_sync(false);
    engine.instance.unload().unwrap();
    engine.load(Path::new("fake.onnx")).await.unwrap();
    drop(engine);
    assert_eq!(pool.bytes((7, 2)), 0);
}

#[tokio::test]
async fn initialization_allocations_remain_owned_through_load_and_unload() {
    let pool = Arc::new(FakePool::default());
    let mut engine = engine(&pool, (7, 2), "initialize-allocates").unwrap();
    assert_eq!(pool.bytes((7, 2)), 256);
    engine.load(Path::new("fake.onnx")).await.unwrap();
    assert_eq!(pool.bytes((7, 2)), 512);
    engine.instance.unload().unwrap();
    assert_eq!(pool.bytes((7, 2)), 0);
}

#[tokio::test]
async fn stream_chunks_cannot_cross_request_ownership() {
    let pool = Arc::new(FakePool::default());
    let mut engine = engine(&pool, (7, 2), "stream-wrong-owner").unwrap();
    engine.load(Path::new("fake.onnx")).await.unwrap();
    let chunks: Vec<_> = engine.infer_stream(&request()).collect().await;
    assert_eq!(chunks.len(), 1);
    assert!(chunks
        .into_iter()
        .next()
        .unwrap()
        .unwrap_err()
        .to_string()
        .contains("different request"));
    engine.instance.unload().unwrap();
    assert_eq!(pool.bytes((7, 2)), 0);
}

#[tokio::test]
async fn cancellation_covers_single_batch_and_concurrent_dispatch_ownership() {
    let pool = Arc::new(FakePool::default());
    let mut engine = engine(&pool, (7, 2), "wait-cancel").unwrap();
    engine.load(Path::new("fake.onnx")).await.unwrap();
    let mut first = request();
    let mut second = request();
    let first_token = CancellationToken::new();
    let second_token = CancellationToken::new();
    first.cancellation = Some(first_token.clone());
    second.cancellation = Some(second_token.clone());
    let instance = Arc::clone(&engine.instance);
    let single = tokio::task::spawn_blocking(move || instance.infer(&first));
    let instance = Arc::clone(&engine.instance);
    let batch = tokio::task::spawn_blocking(move || instance.infer_batch(&[second, request()]));
    wait_started(&engine.instance, 2).await;
    assert_eq!(pool.bytes((7, 2)), 448);
    first_token.cancel();
    assert!(matches!(
        single.await.unwrap().unwrap_err(),
        EngineError::Cancelled { .. }
    ));
    second_token.cancel();
    assert!(matches!(
        batch.await.unwrap().unwrap_err(),
        EngineError::Cancelled { .. }
    ));
    assert_eq!(probe(&engine.instance, 1), 2);
    assert_eq!(
        pool.bytes((7, 2)),
        448,
        "request completion must not free retained arenas"
    );
    engine.instance.unload().unwrap();
    assert_eq!(pool.bytes((7, 2)), 0);
}

#[tokio::test]
async fn dropping_a_live_stream_cancels_before_unload_reclaims_memory() {
    let pool = Arc::new(FakePool::default());
    let mut engine = engine(&pool, (7, 2), "stream-wait-cancel").unwrap();
    engine.load(Path::new("fake.onnx")).await.unwrap();
    let mut stream = engine.infer_stream(&request());
    assert_eq!(stream.next().await.unwrap().unwrap().data, [42]);
    assert_eq!(pool.bytes((7, 2)), 320);
    drop(stream);
    assert_eq!(probe(&engine.instance, 1), 1);
    engine.instance.unload().unwrap();
    assert_eq!(pool.bytes((7, 2)), 0);
}

#[test]
fn adapter_requires_full_versioned_host_table_and_every_allocator_callback() {
    let pack = pack();
    let pool = Arc::new(FakePool::default());
    let host = pool.host(7, 2, 1024);
    let valid = *unsafe { KapslBackendHostScopedAllocatorV1::from_base(host.table()) }.unwrap();
    let mut invalid = Vec::new();
    macro_rules! invalid { ($field:ident $(.$sub:ident)?, $value:expr) => {{ let mut table = valid; table.$field $(.$sub)? = $value; invalid.push(table); }}; }
    invalid!(
        base.struct_size,
        std::mem::size_of::<KapslBackendHostV1>() as u32
    );
    invalid!(base.abi_version, 99);
    invalid!(scoped_allocator_version, 99);
    invalid!(reserved, 1);
    invalid!(allocate_device_scoped, None);
    invalid!(base.allocate_device, None);
    invalid!(base.free_device, None);
    invalid!(base.synchronize_device, None);
    for table in invalid {
        let config = KapslBackendConfigV1 {
            struct_size: std::mem::size_of::<KapslBackendConfigV1>() as u32,
            device_id: 0,
            model_id: 7,
            replica_id: 2,
            require_governed_device_memory: 1,
            reserved: 0,
            profile: KapslSlice::empty(),
            manifest_json: KapslSlice::empty(),
            options_json: KapslSlice::empty(),
            host: &table.base,
        };
        let mut handle = std::ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        assert_eq!(
            unsafe { pack.api.initialize.unwrap()(&config, &mut handle, &mut error) },
            KAPSL_STATUS_INCOMPATIBLE_ABI
        );
        assert!(handle.is_null());
    }
}

#[test]
fn advertised_capabilities_and_required_functions_are_audited_against_loaded_table() {
    let pack = pack();
    macro_rules! missing { ($($field:ident),+ $(,)?) => { $(
        let mut api = pack.api;
        api.$field = None;
        assert!(validate_native_backend_api(&pack.manifest, &api).is_err(), "accepted missing {}", stringify!($field));
    )+ }; }
    missing!(
        describe,
        initialize,
        planned_memory,
        load_model,
        planned_request_memory,
        infer,
        actual_memory,
        metrics,
        model_info,
        batching_policy,
        health_check,
        unload,
        shutdown,
        release_result,
        free_buffer,
        infer_batch,
        release_batch_result,
        infer_stream,
        cancel
    );
    let mut api = pack.api;
    api.capabilities |= KAPSL_BACKEND_CAP_KV_PARTICIPANT;
    let mut manifest = pack.manifest.clone();
    manifest.capabilities = pack_capabilities_from_abi(api.capabilities);
    assert!(validate_native_backend_api(&manifest, &api).is_err());
    for field in [
        "backend",
        "profiles",
        "schema_version",
        "backend_abi",
        "wire_format",
        "execution_mode",
        "formats",
        "tasks",
        "governed_device_memory",
    ] {
        let mut descriptor = pack.descriptor.clone();
        descriptor[field] = serde_json::Value::Null;
        assert!(
            validate_native_backend_descriptor(&pack.manifest, &pack.api, &descriptor).is_err(),
            "accepted descriptor mismatch: {field}"
        );
    }
}
