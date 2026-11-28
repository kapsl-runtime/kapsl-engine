use super::*;
use base64::engine::general_purpose::STANDARD as BASE64;
use std::net::SocketAddr;
use warp::http::StatusCode;
use warp::Reply;

fn test_device_info() -> Arc<DeviceInfo> {
    Arc::new(DeviceInfo {
        cpu_cores: 1,
        total_memory: 10 * 1024 * 1024,
        os_type: "test".to_string(),
        os_release: "test".to_string(),
        has_cuda: false,
        has_metal: false,
        has_rocm: false,
        has_directml: false,
        devices: Vec::new(),
    })
}

fn test_pressure_config() -> Arc<RuntimePressureConfig> {
    Arc::new(RuntimePressureConfig {
        memory_conserve_ratio: 0.7,
        memory_emergency_ratio: 0.9,
        gpu_util_conserve_ratio: 0.8,
        gpu_util_emergency_ratio: 0.95,
        gpu_mem_conserve_ratio: 0.8,
        gpu_mem_emergency_ratio: 0.95,
        conserve_max_new_tokens: Some(256),
        emergency_max_new_tokens: Some(128),
    })
}

fn test_inference_service(models: Arc<ModelManager>) -> Arc<InferenceService> {
    InferenceService::new(
        models,
        ResourcePressure::new(
            Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8)),
            test_pressure_config(),
        ),
        Arc::new(ModelTelemetry::default()),
    )
}

#[cfg(feature = "grpc-server")]
#[tokio::test]
async fn grpc_uses_shared_scheduler_pressure_and_session_isolation() {
    use kapsl_grpc::{
        inference::{
            infer_parameter::ParameterChoice, model_infer_request::InferInputTensor,
            InferParameter, InferTensorContents, ModelInferRequest,
        },
        kapsl::v1::kapsl_inference_client::KapslInferenceClient,
        tonic::{Code, Request},
    };
    let captured = Arc::new(Mutex::new(Vec::new()));
    let models = ModelManager::new(Arc::new(ModelRegistry::new()));
    models.registry().register(ModelInfo::new(
        9,
        "sessions".into(),
        "1".into(),
        "test".into(),
        "cpu".into(),
        "all".into(),
        "/unused".into(),
    ));
    let engine: EngineHandle = Arc::new(SessionCaptureEngine {
        observed_sessions: captured.clone(),
    });
    let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);
    models.install_loaded(9, PathBuf::from("/unused"), Arc::new(pool), vec![]);
    let pressure_state = Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8));
    let inference = InferenceService::new(
        models.clone(),
        ResourcePressure::new(pressure_state.clone(), test_pressure_config()),
        Arc::new(ModelTelemetry::default()),
    );
    let dir = tempfile::tempdir().unwrap();
    let auth = Arc::new(RwLock::new(ApiAuthState {
        role_tokens: ApiRoleTokenConfig {
            reader_token: Some("reader-token".into()),
            writer_token: None,
            admin_token: Some("admin-token".into()),
        },
        store_path: dir.path().join("auth.json"),
        store: ApiAuthStoreFile::default(),
        key_hash_index: HashMap::new(),
    }));
    let mut server = kapsl_grpc::start_server(
        kapsl_grpc::GrpcServerConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            max_message_bytes: 1024 * 1024,
            server_version: "test".into(),
        },
        Arc::new(GrpcEngine { models, inference }),
        Arc::new(GrpcAuthorizer(auth.clone())),
    )
    .await
    .unwrap();
    let mut client = KapslInferenceClient::connect(format!("http://{}", server.bound_addr()))
        .await
        .unwrap();
    let wire = || {
        let mut wire = ModelInferRequest {
            model_name: "sessions".into(),
            inputs: vec![InferInputTensor {
                name: "input".into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                contents: Some(InferTensorContents {
                    bytes_contents: vec![b"hello".to_vec()],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        wire.parameters.insert(
            "session_id".into(),
            InferParameter {
                parameter_choice: Some(ParameterChoice::StringParam("same-session".into())),
            },
        );
        wire
    };
    for token in ["reader-token", "admin-token"] {
        let mut request = Request::new(wire());
        request
            .metadata_mut()
            .insert("authorization", format!("Bearer {token}").parse().unwrap());
        let mut stream = client.infer_stream(request).await.unwrap().into_inner();
        let packet = stream.message().await.unwrap().unwrap();
        assert_eq!(&packet.raw_output_contents[0][4..], b"ok");
        assert!(stream.message().await.unwrap().is_none());
    }
    let sessions = captured.lock().clone();
    assert_eq!(sessions.len(), 2);
    assert_ne!(sessions[0], sessions[1]);
    assert_eq!(
        sessions[0],
        scope_session_id_for_authorization(Some("same-session"), Some("Bearer reader-token"))
    );
    // Emergency pressure rejects throughput work at the shared inference boundary.
    pressure_state.store(RuntimePressureState::Emergency as u8, Ordering::SeqCst);
    let mut request = wire();
    request.inputs[0].datatype = "UINT8".into();
    request.inputs[0].shape = vec![8, 75_000];
    request.inputs[0].contents = None;
    request.raw_input_contents = vec![vec![0; 600_000]];
    let mut request = Request::new(request);
    request
        .metadata_mut()
        .insert("authorization", "Bearer reader-token".parse().unwrap());
    let error = client.infer_stream(request).await.unwrap_err();
    assert_eq!(error.code(), Code::ResourceExhausted);
    assert_eq!(captured.lock().len(), 2);
    auth.write().role_tokens.reader_token = None;
    let mut request = Request::new(wire());
    request
        .metadata_mut()
        .insert("authorization", "Bearer reader-token".parse().unwrap());
    assert_eq!(
        client.infer_stream(request).await.unwrap_err().code(),
        Code::Unauthenticated
    );
    server.shutdown().await;
}

fn unique_temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("kapsl-route-test-{}-{}", name, std::process::id()))
}

#[tokio::test]
async fn test_static_routes_serve_embedded_index() {
    let routes = build_static_routes();

    let response = warp::test::request().path("/").reply(&routes).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = String::from_utf8_lossy(response.body());
    assert!(body.contains("<!doctype html") || body.contains("<!DOCTYPE html"));
}

#[tokio::test]
async fn test_system_routes_report_health_and_pressure_state() {
    let registry = Arc::new(ModelRegistry::new());
    registry.register(ModelInfo::new(
        7,
        "alpha".to_string(),
        "1".to_string(),
        "onnx".to_string(),
        "cpu".to_string(),
        "all".to_string(),
        "/tmp/alpha.aimod".to_string(),
    ));
    registry.register(ModelInfo::new(
        9,
        "beta".to_string(),
        "1".to_string(),
        "onnx".to_string(),
        "cpu".to_string(),
        "all".to_string(),
        "/tmp/beta.aimod".to_string(),
    ));
    let models = ModelManager::new(registry);
    let device_info = test_device_info();
    let resources = RuntimeResources::new(device_info.as_ref()).expect("runtime resources");
    let mut memory_plan = MemoryPlan::new();
    memory_plan.push(MemoryClaim::runtime(
        MemoryDomain::Host,
        MemoryOwner::new(7, 0),
        MemoryAllocationClass::PersistentWeights,
        300,
    ));
    memory_plan.push(MemoryClaim::runtime(
        MemoryDomain::Host,
        MemoryOwner::new(7, 1),
        MemoryAllocationClass::KvCache,
        100,
    ));
    memory_plan.push(MemoryClaim::runtime(
        MemoryDomain::Host,
        MemoryOwner::new(9, 0),
        MemoryAllocationClass::ModelSession,
        100,
    ));
    memory_plan.push(MemoryClaim::runtime(
        MemoryDomain::HostPinned {
            provider: "onnx".to_string(),
            device_id: Some(0),
        },
        MemoryOwner::new(7, 0),
        MemoryAllocationClass::TransientWorkspace,
        500,
    ));
    let _memory_lease = resources
        .memory()
        .admit(&memory_plan)
        .expect("memory plan admitted");
    let runtime_samples = Arc::new(RwLock::new(RuntimeSamples {
        process_memory_bytes: 123,
        total_system_memory_bytes: Some(456),
        gpu_utilization: 7.5,
        gpu_memory_bytes: Some(10),
        gpu_memory_total_bytes: Some(20),
        foreign_gpu_memory_bytes: Some(4),
        collected_at_ms: 789,
    }));
    let pressure_state = Arc::new(AtomicU8::new(RuntimePressureState::Conserve as u8));
    let routes = build_system_routes(
        models,
        device_info,
        runtime_samples,
        pressure_state,
        resources,
    );

    let health = warp::test::request()
        .path("/api/health")
        .reply(&routes)
        .await;
    assert_eq!(health.status(), StatusCode::OK);
    let health_json: serde_json::Value =
        serde_json::from_slice(health.body()).expect("health json");
    assert_eq!(health_json["status"], "healthy");
    assert_eq!(health_json["total_models"], 2);

    let stats = warp::test::request()
        .path("/api/system/stats")
        .reply(&routes)
        .await;
    assert_eq!(stats.status(), StatusCode::OK);
    let stats_json: serde_json::Value = serde_json::from_slice(stats.body()).expect("stats json");
    assert_eq!(stats_json["process_memory_bytes"], 123);
    assert_eq!(stats_json["total_system_memory_bytes"], 456);
    assert_eq!(stats_json["gpu_memory_total_bytes"], 20);
    assert_eq!(stats_json["foreign_gpu_memory_bytes"], 4);
    assert_eq!(stats_json["pressure_state"], "conserve");
    // CUDA driver/runtime versions and the GPU sample freshness are served
    // here; the values are null/"fresh" on a host without a GPU.
    for key in ["driver_version", "driver_cuda_version", "runtime_version"] {
        assert!(
            stats_json["cuda"].get(key).is_some(),
            "stats `cuda` lacks `{key}`: {}",
            stats_json["cuda"]
        );
    }
    assert_eq!(stats_json["gpu_sampling"], "fresh");
    assert!(stats_json["gpu_sampling_age_secs"].is_null());
    for domain in stats_json["memory_authority"]["domains"]
        .as_array()
        .unwrap()
    {
        assert_eq!(
            domain["available_bytes"],
            domain["authority_available_bytes"]
        );
    }
    assert_eq!(stats_json["memory_authority"]["model_bytes"], 1000);
    assert_eq!(stats_json["memory_authority"]["domain_used_bytes"], 1000);
    assert_eq!(
        stats_json["memory_authority"]["domains"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(
        stats_json["memory_authority"]["domain_budget_bytes"],
        stats_json["memory_authority"]["domains"][0]["budget_bytes"]
    );
    assert_eq!(stats_json["memory_authority"]["models"][0]["model_id"], 7);
    assert_eq!(stats_json["memory_authority"]["models"][0]["name"], "alpha");
    assert_eq!(
        stats_json["memory_authority"]["models"][0]["replica_count"],
        2
    );
    assert_eq!(
        stats_json["memory_authority"]["models"][0]["percentage"],
        90.0
    );
    assert_eq!(stats_json["memory_authority"]["models"][1]["model_id"], 9);
    assert_eq!(
        stats_json["memory_authority"]["models"][1]["percentage"],
        10.0
    );
}

#[tokio::test]
async fn test_auth_login_route_allows_local_loopback_when_auth_disabled() {
    let auth_state = Arc::new(RwLock::new(ApiAuthState {
        role_tokens: ApiRoleTokenConfig::default(),
        store_path: unique_temp_path("auth-store").join("auth-store.json"),
        store: ApiAuthStoreFile::default(),
        key_hash_index: HashMap::new(),
    }));
    let routes = build_auth_routes(auth_state).login;

    let response = warp::test::request()
        .method("POST")
        .path("/api/auth/login")
        .remote_addr(SocketAddr::from(([127, 0, 0, 1], 45_001)))
        .header("content-type", "application/json")
        .body("{}")
        .reply(&routes)
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("login json");
    assert_eq!(body["authenticated"], true);
    assert_eq!(body["mode"], "local-loopback");
    assert_eq!(body["access"]["admin"], true);
}

#[tokio::test]
async fn test_metrics_auth_does_not_intercept_login_or_role_routes() {
    let now = now_unix_seconds();
    let credentials = [
        ("reader-user", "reader-api-key", ApiRole::Reader),
        ("writer-user", "writer-api-key", ApiRole::Writer),
        ("admin-user", "admin-api-key", ApiRole::Admin),
    ];
    let store = ApiAuthStoreFile {
        users: credentials
            .iter()
            .map(|(user_id, _, role)| ApiAuthUser {
                id: (*user_id).to_string(),
                username: (*user_id).to_string(),
                display_name: None,
                role: *role,
                status: ApiUserStatus::Active,
                created_at: now,
                updated_at: now,
            })
            .collect(),
        api_keys: credentials
            .iter()
            .enumerate()
            .map(|(index, (user_id, token, _))| ApiAuthKey {
                id: format!("key-{index}"),
                user_id: (*user_id).to_string(),
                name: format!("key-{index}"),
                key_prefix: token.chars().take(8).collect(),
                key_hash: sha256_hex(token),
                scopes: Vec::new(),
                created_at: now,
                created_by: None,
                last_used_at: None,
                expires_at: None,
                revoked_at: None,
            })
            .collect(),
    };
    let auth_state = Arc::new(RwLock::new(ApiAuthState {
        role_tokens: ApiRoleTokenConfig::default(),
        store_path: unique_temp_path("role-route-auth-store").join("auth-store.json"),
        key_hash_index: ApiAuthState::build_key_hash_index(&store),
        store,
    }));
    let resources = RuntimeResources::new(test_device_info().as_ref()).expect("runtime resources");
    let metrics_route =
        build_metrics_route(Arc::new(Registry::new()), auth_state.clone(), resources);
    let login_route = build_auth_routes(auth_state.clone()).login;

    let reader_route = warp::path!("api" / "reader-probe")
        .and(warp::get())
        .map(|| warp::reply::json(&serde_json::json!({ "role": "reader" })).into_response());
    let reader_route = api_auth_filter(ApiRole::Reader, ApiScope::Read, auth_state.clone())
        .and(reader_route)
        .map(|response: warp::reply::Response| response);
    let writer_route = warp::path!("api" / "writer-probe")
        .and(warp::post())
        .map(|| warp::reply::json(&serde_json::json!({ "role": "writer" })).into_response());
    let writer_route = api_auth_filter(ApiRole::Writer, ApiScope::Write, auth_state)
        .and(writer_route)
        .map(|response: warp::reply::Response| response);
    let api_routes = reader_route
        .or(writer_route)
        .unify()
        .or_else(map_api_auth_rejection);
    let routes = metrics_route.or(login_route).or(api_routes);

    let login = warp::test::request()
        .method("POST")
        .path("/api/auth/login")
        .remote_addr(SocketAddr::from(([127, 0, 0, 1], 45_002)))
        .header("content-type", "application/json")
        .body(r#"{"token":"reader-api-key"}"#)
        .reply(&routes)
        .await;
    assert_eq!(login.status(), StatusCode::OK);
    let login_body: serde_json::Value =
        serde_json::from_slice(login.body()).expect("login response json");
    assert_eq!(login_body["role"], "reader");
    assert_eq!(login_body["mode"], "api-key");

    let reader = warp::test::request()
        .path("/api/reader-probe")
        .header("authorization", "Bearer reader-api-key")
        .reply(&routes)
        .await;
    assert_eq!(reader.status(), StatusCode::OK);

    let writer_read = warp::test::request()
        .path("/api/reader-probe")
        .header("authorization", "Bearer writer-api-key")
        .reply(&routes)
        .await;
    assert_eq!(writer_read.status(), StatusCode::OK);

    let writer = warp::test::request()
        .method("POST")
        .path("/api/writer-probe")
        .header("authorization", "Bearer writer-api-key")
        .reply(&routes)
        .await;
    assert_eq!(writer.status(), StatusCode::OK);

    let reader_write = warp::test::request()
        .method("POST")
        .path("/api/writer-probe")
        .header("authorization", "Bearer reader-api-key")
        .reply(&routes)
        .await;
    assert_eq!(reader_write.status(), StatusCode::FORBIDDEN);

    let reader_metrics = warp::test::request()
        .path("/metrics")
        .header("authorization", "Bearer reader-api-key")
        .reply(&routes)
        .await;
    assert_eq!(reader_metrics.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn test_rag_query_route_rejects_missing_workspace_before_store_access() {
    let rag_root = unique_temp_path("rag");
    let docs_root = rag_root.join("docs");
    fs::create_dir_all(&docs_root).expect("create rag docs dir");
    let rag = RagService::new(
        Arc::new(
            SqliteVectorStore::open(&rag_root.join("vectors.sqlite3")).expect("open vector store"),
        ),
        Arc::new(FsDocStore::new(&docs_root)),
    );
    let routes = build_rag_routes(rag);

    let response = warp::test::request()
        .method("POST")
        .path("/api/rag/query")
        .header("content-type", "application/json")
        .body(r#"{"workspace_id":"","query":"hello"}"#)
        .reply(&routes)
        .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("rag json");
    assert_eq!(body["error"], "workspace_id is required");
}

#[tokio::test]
async fn test_engine_browse_route_lists_model_relevant_files() {
    let browse_root = unique_temp_path("browse");
    fs::create_dir_all(&browse_root).expect("create browse dir");
    fs::write(browse_root.join("model.gguf"), b"test").expect("write model file");
    fs::write(browse_root.join("notes.txt"), b"ignore").expect("write ignored file");

    let routes = build_engine_routes().admin;
    let response = warp::test::request()
        .path(&format!(
            "/api/engine/browse?path={}",
            browse_root.to_string_lossy()
        ))
        .reply(&routes)
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("browse json");
    let names = body["entries"]
        .as_array()
        .expect("entries array")
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"model.gguf"));
    assert!(!names.contains(&"notes.txt"));
}

#[tokio::test]
async fn test_infer_stream_route_returns_not_found_for_unknown_model() {
    let rag_root = unique_temp_path("infer-stream-rag");
    let docs_root = rag_root.join("docs");
    fs::create_dir_all(&docs_root).expect("create rag docs dir");
    let rag = RagService::new(
        Arc::new(
            SqliteVectorStore::open(&rag_root.join("vectors.sqlite3")).expect("open vector store"),
        ),
        Arc::new(FsDocStore::new(&docs_root)),
    );

    let models = ModelManager::new(Arc::new(ModelRegistry::new()));
    let route = build_model_infer_stream_route(ModelInferStreamRouteConfig {
        models: models.clone(),
        inference: test_inference_service(models),
        log_sensitive_ids: false,
        rag,
    });

    let response = warp::test::request()
        .method("POST")
        .path("/api/models/0/infer/stream")
        .header("content-type", "application/json")
        .body(r#"{"input":{"shape":[1],"dtype":"string","data":[104,105]}}"#)
        .reply(&route)
        .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value =
        serde_json::from_slice(response.body()).expect("infer-stream json");
    assert_eq!(body["error"], "Model 0 not found");
}

/// What a backend receives of a request's generation parameters.
type SeenGeneration = (Option<u32>, Option<u32>, Option<f32>);

struct GenerationCaptureEngine {
    seen: Arc<Mutex<Vec<SeenGeneration>>>,
}

#[async_trait::async_trait]
impl Engine for GenerationCaptureEngine {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        let metadata = request.metadata.as_ref();
        self.seen.lock().push((
            metadata.and_then(|m| m.max_new_tokens),
            metadata.and_then(|m| m.min_new_tokens),
            metadata.and_then(|m| m.temperature),
        ));
        BinaryTensorPacket::new(vec![1, 2], TensorDtype::Utf8, b"ok".to_vec())
    }

    fn infer_stream(&self, request: &InferenceRequest) -> kapsl_engine_api::EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

fn generation_models(seen: Arc<Mutex<Vec<SeenGeneration>>>) -> Arc<ModelManager> {
    let models = ModelManager::new(Arc::new(ModelRegistry::new()));
    models.registry().register(ModelInfo::new(
        4,
        "gen".into(),
        "1".into(),
        "llm".into(),
        "cpu".into(),
        "all".into(),
        "/unused".into(),
    ));
    let engine: EngineHandle = Arc::new(GenerationCaptureEngine { seen });
    let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);
    models.install_loaded(4, PathBuf::from("/unused"), Arc::new(pool), vec![]);
    models
}

fn generation_rag(name: &str) -> RagService {
    let root = unique_temp_path(name);
    let docs = root.join("docs");
    fs::create_dir_all(&docs).expect("create rag docs dir");
    RagService::new(
        Arc::new(
            SqliteVectorStore::open(&root.join("vectors.sqlite3")).expect("open vector store"),
        ),
        Arc::new(FsDocStore::new(&docs)),
    )
}

const GEN_INPUT: &str = r#""input":{"shape":[2],"dtype":"string","data":[104,105]}"#;

/// Run 9b, check l, through the real HTTP handlers: the body the harness posts (top-level
/// `max_tokens`, `min_tokens`, `temperature` beside `input`) must reach the backend. Before the fix
/// the handlers' fast path deserialized the body straight into an `InferenceRequest`, never reached
/// the adapter, and dropped all three: a real GGUF answered `5 times 3 is 15.` (16 bytes) to
/// `max_tokens: 3` and to `min_tokens: 20` alike.
#[tokio::test]
async fn the_infer_route_delivers_top_level_generation_parameters_to_the_backend() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let models = generation_models(seen.clone());
    let route = build_model_infer_route(ModelInferRouteConfig {
        models: models.clone(),
        inference: test_inference_service(models),
        log_sensitive_ids: false,
        rag: generation_rag("gen-infer-rag"),
    });
    let post = |body: String| {
        let route = route.clone();
        async move {
            warp::test::request()
                .method("POST")
                .path("/api/models/4/infer")
                .header("content-type", "application/json")
                .body(body)
                .reply(&route)
                .await
        }
    };
    let response = post(format!(
        r#"{{{GEN_INPUT},"max_tokens":3,"min_tokens":20,"temperature":0}}"#
    ))
    .await;
    assert_eq!(response.status(), StatusCode::OK, "{:?}", response.body());
    // the other spelling
    post(format!(r#"{{{GEN_INPUT},"min_new_tokens":7}}"#)).await;
    // a value the body carries in its own `metadata` object wins over the top-level one
    post(format!(
        r#"{{{GEN_INPUT},"max_tokens":3,"metadata":{{"max_new_tokens":9}}}}"#
    ))
    .await;
    // nothing generation-related: nothing invented
    post(format!(r#"{{{GEN_INPUT}}}"#)).await;
    assert_eq!(
        *seen.lock(),
        vec![
            (Some(3), Some(20), Some(0.0)),
            (None, Some(7), None),
            (Some(9), None, None),
            (None, None, None),
        ]
    );
}

#[tokio::test]
async fn the_infer_stream_route_delivers_top_level_generation_parameters_to_the_backend() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let models = generation_models(seen.clone());
    let route = build_model_infer_stream_route(ModelInferStreamRouteConfig {
        models: models.clone(),
        inference: test_inference_service(models),
        log_sensitive_ids: false,
        rag: generation_rag("gen-stream-rag"),
    });
    let response = warp::test::request()
        .method("POST")
        .path("/api/models/4/infer/stream")
        .header("content-type", "application/json")
        .body(format!(
            r#"{{{GEN_INPUT},"max_tokens":3,"min_tokens":20,"temperature":0}}"#
        ))
        .reply(&route)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    // the stream is consumed with the body
    let _ = response.body();
    assert_eq!(*seen.lock(), vec![(Some(3), Some(20), Some(0.0))]);
}

fn test_openai_routes() -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let models = ModelManager::new(Arc::new(ModelRegistry::new()));
    build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference: test_inference_service(models),
    })
}

struct OpenAiTestEngine;

#[async_trait::async_trait]
impl Engine for OpenAiTestEngine {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, _request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        BinaryTensorPacket::new(vec![1, 16], TensorDtype::Utf8, b"hello from Kapsl".to_vec())
    }

    fn infer_stream(&self, _request: &InferenceRequest) -> kapsl_engine_api::EngineStream {
        let packets = ["hello ", "from ", "Kapsl"].map(|text| {
            BinaryTensorPacket::new(
                vec![1, text.len() as i64],
                TensorDtype::Utf8,
                text.as_bytes().to_vec(),
            )
        });
        Box::pin(futures::stream::iter(packets))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

struct SessionCaptureEngine {
    observed_sessions: Arc<Mutex<Vec<Option<String>>>>,
}

#[async_trait::async_trait]
impl Engine for SessionCaptureEngine {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        self.observed_sessions
            .lock()
            .push(request.session_id.clone());
        BinaryTensorPacket::new(vec![1, 2], TensorDtype::Utf8, b"ok".to_vec())
    }

    fn infer_stream(&self, request: &InferenceRequest) -> kapsl_engine_api::EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

struct OpenAiEmbeddingTestEngine;

#[async_trait::async_trait]
impl Engine for OpenAiEmbeddingTestEngine {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        if request.input.dtype != TensorDtype::Int64 || request.input.shape.len() != 2 {
            return Err(EngineError::InvalidInput {
                message: "embedding test engine expected a rank-2 int64 input".to_string(),
                source: None,
            });
        }
        for required in ["attention_mask", "token_type_ids"] {
            if !request
                .additional_inputs
                .iter()
                .any(|input| input.name == required)
            {
                return Err(EngineError::InvalidInput {
                    message: format!("embedding test engine is missing {required}"),
                    source: None,
                });
            }
        }

        let first_token = request
            .input
            .data
            .as_chunks::<8>()
            .0
            .first()
            .map(|bytes| i64::from_ne_bytes(*bytes))
            .unwrap_or_default() as f32;
        let values = [first_token, 2.0, 3.0, 4.0];
        let data = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        BinaryTensorPacket::new(vec![1, values.len() as i64], TensorDtype::Float32, data)
    }

    fn infer_stream(&self, request: &InferenceRequest) -> kapsl_engine_api::EngineStream {
        let result = self.infer(request);
        Box::pin(futures::stream::once(async move { result }))
    }

    fn unload(&mut self) {}

    fn metrics(&self) -> EngineMetrics {
        EngineMetrics::default()
    }

    fn model_info(&self) -> Option<EngineModelInfo> {
        Some(EngineModelInfo {
            input_names: vec![
                "input_ids".to_string(),
                "attention_mask".to_string(),
                "token_type_ids".to_string(),
            ],
            output_names: vec!["sentence_embedding".to_string()],
            input_shapes: vec![vec![-1, -1], vec![-1, -1], vec![-1, -1]],
            output_shapes: vec![vec![-1, 4]],
            input_dtypes: vec![
                "int64".to_string(),
                "int64".to_string(),
                "int64".to_string(),
            ],
            output_dtypes: vec!["float32".to_string()],
            framework: Some("onnx".to_string()),
            model_version: Some("1".to_string()),
            peak_concurrency: Some(1),
        })
    }

    fn health_check(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

fn test_openai_routes_with_model() -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    const MODEL_ID: u32 = 7;
    let registry = Arc::new(ModelRegistry::new());
    registry.register(ModelInfo::new(
        MODEL_ID,
        "test-model".to_string(),
        "1".to_string(),
        "gguf".to_string(),
        "cpu".to_string(),
        "all".to_string(),
        "/tmp/test-model.aimod".to_string(),
    ));
    let models = ModelManager::new(registry);
    let engine: EngineHandle = Arc::new(OpenAiTestEngine);
    let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);
    models.install_loaded(
        MODEL_ID,
        PathBuf::from("/tmp/test-model.aimod"),
        Arc::new(pool),
        vec![],
    );
    build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference: test_inference_service(models),
    })
}

fn test_openai_routes_with_session_capture(
    observed_sessions: Arc<Mutex<Vec<Option<String>>>>,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    const MODEL_ID: u32 = 9;
    let registry = Arc::new(ModelRegistry::new());
    registry.register(ModelInfo::new(
        MODEL_ID,
        "session-test-model".to_string(),
        "1".to_string(),
        "gguf".to_string(),
        "cpu".to_string(),
        "all".to_string(),
        "/tmp/session-test-model.aimod".to_string(),
    ));
    let models = ModelManager::new(registry);
    let engine: EngineHandle = Arc::new(SessionCaptureEngine { observed_sessions });
    let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);
    models.install_loaded(
        MODEL_ID,
        PathBuf::from("/tmp/session-test-model.aimod"),
        Arc::new(pool),
        vec![],
    );
    build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference: test_inference_service(models),
    })
}

fn test_openai_routes_with_embedding_model(
    model_path: PathBuf,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    const MODEL_ID: u32 = 8;
    let registry = Arc::new(ModelRegistry::new());
    registry.register(
        ModelInfo::new(
            MODEL_ID,
            "test-embedding-model".to_string(),
            "1".to_string(),
            "onnx".to_string(),
            "cpu".to_string(),
            "all".to_string(),
            model_path.to_string_lossy().to_string(),
        )
        .with_model_axes(
            Some("onnx".to_string()),
            Some("embedding".to_string()),
            Some("embed".to_string()),
            None,
        ),
    );
    let models = ModelManager::new(registry);
    let engine: EngineHandle = Arc::new(OpenAiEmbeddingTestEngine);
    let scheduler = Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
    pool.add_replica(0, scheduler);
    models.install_loaded(MODEL_ID, model_path, Arc::new(pool), vec![]);
    build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference: test_inference_service(models),
    })
}

/// A model the idle unloader stopped, with idle reload on. `reload` decides
/// how the reload a request triggers ends.
fn idle_unloaded_model(
    reload: Result<(), IdleReloadError>,
) -> (Arc<ModelManager>, Arc<InferenceService>) {
    let (models, inference, _) = counted_idle_unloaded_model(reload);
    (models, inference)
}

/// [`idle_unloaded_model`], also counting the loads its reloader starts.
fn counted_idle_unloaded_model(
    reload: Result<(), IdleReloadError>,
) -> (Arc<ModelManager>, Arc<InferenceService>, Arc<AtomicUsize>) {
    const MODEL_ID: u32 = 7;
    let loads = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(ModelRegistry::new());
    registry.register(ModelInfo::new(
        MODEL_ID,
        "test-model".to_string(),
        "1".to_string(),
        "gguf".to_string(),
        "cpu".to_string(),
        "all".to_string(),
        "/tmp/test-model.aimod".to_string(),
    ));
    let models = ModelManager::new(registry);
    models.configure_idle_unload(IdleUnloadConfig {
        reload_on_request: true,
        ..IdleUnloadConfig::default()
    });
    models.install_loaded(
        MODEL_ID,
        PathBuf::from("/tmp/test-model.aimod"),
        Arc::new(ReplicaPool::new(PoolStrategy::LeastLoaded)),
        vec![],
    );
    models.begin_idle_unload(MODEL_ID);
    models.stop_runtime(MODEL_ID);

    let reloader = IdleReloader::new(models.clone(), Duration::from_secs(10), {
        let loads = loads.clone();
        move |_,
              _,
              _|
              -> futures::future::BoxFuture<'static, Result<LoadedModel, IdleReloadError>> {
            loads.fetch_add(1, Ordering::SeqCst);
            let reload = reload.clone();
            Box::pin(async move {
                reload.map(|()| {
                    let engine: EngineHandle = Arc::new(OpenAiTestEngine);
                    let scheduler =
                        Arc::new(Scheduler::new(vec![engine], 1, 1, 8, true, 1, 0, None));
                    let pool = ReplicaPool::new(PoolStrategy::LeastLoaded);
                    pool.add_replica(0, scheduler);
                    (Arc::new(pool), Vec::new())
                })
            })
        }
    });
    let inference = InferenceService::new_with_idle_reloader(
        models.clone(),
        ResourcePressure::new(
            Arc::new(AtomicU8::new(RuntimePressureState::Normal as u8)),
            test_pressure_config(),
        ),
        Arc::new(ModelTelemetry::default()),
        reloader,
    );
    (models, inference, loads)
}

fn memory_held_by_training() -> IdleReloadError {
    IdleReloadError::MemoryUnavailable {
        message: "model 7 was stopped for being idle and cannot be reloaded right now: the memory it needs is in use, possibly by a training job".to_string(),
    }
}

#[tokio::test]
async fn an_idle_unloaded_model_stays_advertised_and_a_chat_request_reloads_it() {
    let (models, inference) = idle_unloaded_model(Ok(()));
    let routes = build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference,
    });

    let listed = warp::test::request()
        .path("/v1/models")
        .reply(&routes)
        .await;
    let listed: serde_json::Value = serde_json::from_slice(listed.body()).expect("models json");
    assert_eq!(listed["data"].as_array().expect("data array").len(), 1);
    assert!(!models.contains_pool(7));

    let response = warp::test::request()
        .method("POST")
        .path("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#)
        .reply(&routes)
        .await;
    assert_eq!(response.status(), StatusCode::OK, "{:?}", response.body());
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("chat json");
    assert_eq!(body["choices"][0]["message"]["content"], "hello from Kapsl");
    assert!(models.contains_pool(7));
    assert!(!models.is_idle_unloaded(7));
}

#[tokio::test]
async fn openai_requests_get_503_and_retry_after_when_the_idle_reload_is_denied() {
    let (models, inference) = idle_unloaded_model(Err(memory_held_by_training()));
    let routes = build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference,
    });

    for (path, body) in [
        (
            "/v1/chat/completions",
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        ("/v1/responses", r#"{"model":"test-model","input":"hi"}"#),
    ] {
        let response = warp::test::request()
            .method("POST")
            .path(path)
            .header("content-type", "application/json")
            .body(body)
            .reply(&routes)
            .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{path}");
        assert_eq!(response.headers()["retry-after"], "10", "{path}");
        let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
        assert_eq!(body["error"]["type"], "server_error", "{path}");
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("test-model"), "{message}");
        assert!(message.contains("training job"), "{message}");
    }
    assert!(!models.contains_pool(7));
    assert!(models.reloads_on_request(7), "a later request tries again");
}

#[tokio::test]
async fn native_infer_routes_report_a_denied_idle_reload_as_unavailable() {
    let (models, inference) = idle_unloaded_model(Err(memory_held_by_training()));
    let infer = build_model_infer_route(ModelInferRouteConfig {
        models: models.clone(),
        inference: inference.clone(),
        log_sensitive_ids: false,
        rag: generation_rag("idle-reload-infer-rag"),
    });
    let stream = build_model_infer_stream_route(ModelInferStreamRouteConfig {
        models: models.clone(),
        inference,
        log_sensitive_ids: false,
        rag: generation_rag("idle-reload-stream-rag"),
    });

    let responses = [
        warp::test::request()
            .method("POST")
            .path("/api/models/7/infer")
            .header("content-type", "application/json")
            .body(format!(r#"{{{GEN_INPUT}}}"#))
            .reply(&infer)
            .await,
        warp::test::request()
            .method("POST")
            .path("/api/models/7/infer/stream")
            .header("content-type", "application/json")
            .body(format!(r#"{{{GEN_INPUT}}}"#))
            .reply(&stream)
            .await,
    ];
    for response in responses {
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
        assert_eq!(body["reason"], "idle_reload_memory_unavailable");
        assert_eq!(body["retry_after_seconds"], 10);
        assert!(body["error"]
            .as_str()
            .expect("error")
            .contains("training job"));
    }

    // A model that is simply unknown keeps its 404.
    let unknown = warp::test::request()
        .method("POST")
        .path("/api/models/99/infer")
        .header("content-type", "application/json")
        .body(format!(r#"{{{GEN_INPUT}}}"#))
        .reply(&infer)
        .await;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
}

/// The failure seen on a GPU shared with training: the managed vLLM planner
/// of an idle reload ran out of CUDA memory.
#[tokio::test]
async fn a_planner_out_of_memory_idle_reload_is_503_with_retry_after_also_when_cached() {
    let load_error = managed_vllm_planner_load_error_for_tests(
        "runtime_geometry_unavailable",
        "certified runtime geometry provider failed: CUDA out of memory. Tried to allocate 260.00 MiB. Process 27985 has 21.83 GiB memory in use.",
    );
    let (models, inference, loads) = counted_idle_unloaded_model(Err(
        IdleReloadError::from_load_error(7, load_error.as_ref()),
    ));
    let routes = build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference: inference.clone(),
    });

    // The first request runs the reload; the second gets its cached failure.
    for attempt in 1..=2 {
        let response = warp::test::request()
            .method("POST")
            .path("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#)
            .reply(&routes)
            .await;
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "attempt {attempt}"
        );
        assert_eq!(response.headers()["retry-after"], "10", "attempt {attempt}");
        let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
        let message = body["error"]["message"].as_str().expect("message");
        assert!(
            message.contains("cannot be reloaded right now"),
            "{message}"
        );
        assert!(message.contains("CUDA out of memory"), "{message}");
        assert!(message.contains("21.83 GiB memory in use"), "{message}");
    }
    assert_eq!(loads.load(Ordering::SeqCst), 1);

    let infer = build_model_infer_route(ModelInferRouteConfig {
        models: models.clone(),
        inference,
        log_sensitive_ids: false,
        rag: generation_rag("idle-reload-planner-oom-rag"),
    });
    let response = warp::test::request()
        .method("POST")
        .path("/api/models/7/infer")
        .header("content-type", "application/json")
        .body(format!(r#"{{{GEN_INPUT}}}"#))
        .reply(&infer)
        .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert_eq!(body["reason"], "idle_reload_memory_unavailable");
    assert_eq!(body["retry_after_seconds"], 10);
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert!(models.reloads_on_request(7), "a later request tries again");
}

#[tokio::test]
async fn other_planner_failures_of_an_idle_reload_stay_500_without_retry_after() {
    let load_error = managed_vllm_planner_load_error_for_tests(
        "planning_failed",
        "--max-model-len must be positive",
    );
    let (models, inference) = idle_unloaded_model(Err(IdleReloadError::from_load_error(
        7,
        load_error.as_ref(),
    )));
    let routes = build_openai_routes(OpenAiRoutesConfig { models, inference });

    let response = warp::test::request()
        .method("POST")
        .path("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#)
        .reply(&routes)
        .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(response.headers().get("retry-after").is_none());
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("could not be reloaded"), "{message}");
    assert!(
        message.contains("--max-model-len must be positive"),
        "{message}"
    );
}

async fn post_chat_completion(body: &str) -> warp::http::Response<warp::hyper::body::Bytes> {
    warp::test::request()
        .method("POST")
        .path("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(body)
        .reply(&test_openai_routes())
        .await
}

async fn post_response(body: &str) -> warp::http::Response<warp::hyper::body::Bytes> {
    warp::test::request()
        .method("POST")
        .path("/v1/responses")
        .header("content-type", "application/json")
        .body(body)
        .reply(&test_openai_routes())
        .await
}

async fn post_embeddings(body: &str) -> warp::http::Response<warp::hyper::body::Bytes> {
    warp::test::request()
        .method("POST")
        .path("/v1/embeddings")
        .header("content-type", "application/json")
        .body(body)
        .reply(&test_openai_routes())
        .await
}

#[tokio::test]
async fn test_openai_models_route_returns_an_empty_list_shape() {
    let response = warp::test::request()
        .path("/v1/models")
        .reply(&test_openai_routes())
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("models json");
    assert_eq!(body["object"], "list");
    assert_eq!(
        body["data"].as_array().expect("data array").len(),
        0,
        "no pools are registered, so nothing should be advertised"
    );
}

#[tokio::test]
async fn test_openai_chat_rejects_malformed_json_in_openai_error_shape() {
    let response = post_chat_completion("{ not json").await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    // The official SDKs read the nested object, not a bare string.
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("Invalid chat completion payload"));
}

#[tokio::test]
async fn test_openai_chat_returns_not_found_for_unknown_model() {
    let response =
        post_chat_completion(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
            .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("no models are currently loaded"));
}

#[tokio::test]
async fn test_openai_chat_rejects_multiple_choices_before_touching_the_pool() {
    let response = post_chat_completion(
        r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"n":3}"#,
    )
    .await;

    // `n` is rejected rather than silently honoured as n=1, and it is checked
    // before model resolution so the message is about the real problem.
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("n=3 is not supported"));
}

#[tokio::test]
async fn test_openai_chat_scopes_same_session_id_to_authorization_credential() {
    let observed_sessions = Arc::new(Mutex::new(Vec::new()));
    let route = test_openai_routes_with_session_capture(Arc::clone(&observed_sessions));
    let body = r#"{"model":"session-test-model","messages":[{"role":"user","content":"hi"}]}"#;

    for credential in ["Bearer key-a", "Bearer key-b"] {
        let response = warp::test::request()
            .method("POST")
            .path("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", credential)
            .header("x-kapsl-session", "shared-session")
            .body(body)
            .reply(&route)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let observed = observed_sessions.lock();
    assert_eq!(observed.len(), 2);
    let first = observed[0].as_deref().expect("first scoped session");
    let second = observed[1].as_deref().expect("second scoped session");
    assert_ne!(first, second);
    assert!(first.starts_with("ks1_"));
    assert!(!first.contains("shared-session"));
    assert!(!first.contains("key-a"));
}

#[tokio::test]
async fn test_openai_responses_rejects_malformed_json_in_openai_error_shape() {
    let response = post_response("{ not json").await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("Invalid response payload"));
}

#[tokio::test]
async fn test_openai_responses_rejects_explicit_storage() {
    let response = post_response(r#"{"model":"test-model","input":"hi","store":true}"#).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("stateless"));
}

#[tokio::test]
async fn test_openai_responses_returns_not_found_for_unknown_model() {
    let response = post_response(r#"{"model":"missing","input":"hi"}"#).await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("no models are currently loaded"));
}

#[tokio::test]
async fn test_openai_responses_non_streaming_uses_response_object_shape() {
    let response = warp::test::request()
        .method("POST")
        .path("/v1/responses")
        .header("content-type", "application/json")
        .body(r#"{"model":"test-model","input":"hi","max_output_tokens":20}"#)
        .reply(&test_openai_routes_with_model())
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("response json");
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["store"], false);
    assert_eq!(body["max_output_tokens"], 20);
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "hello from Kapsl");
}

#[tokio::test]
async fn test_openai_responses_streams_typed_lifecycle_events() {
    let response = warp::test::request()
        .method("POST")
        .path("/v1/responses")
        .header("content-type", "application/json")
        .body(r#"{"model":"test-model","input":"hi","stream":true}"#)
        .reply(&test_openai_routes_with_model())
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers()["content-type"]
        .to_str()
        .expect("content type")
        .starts_with("text/event-stream"));
    let stream = String::from_utf8_lossy(response.body());
    assert!(stream.contains("event: response.created"));
    assert!(stream.contains("event: response.completed"));
    let events: Vec<serde_json::Value> = stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("stream data should be JSON"))
        .collect();
    let types: Vec<&str> = events
        .iter()
        .map(|event| event["type"].as_str().expect("event type"))
        .collect();
    assert_eq!(types.first(), Some(&"response.created"));
    assert_eq!(types.last(), Some(&"response.completed"));
    let text: String = events
        .iter()
        .filter(|event| event["type"] == "response.output_text.delta")
        .filter_map(|event| event["delta"].as_str())
        .collect();
    assert_eq!(text, "hello from Kapsl");
    assert_eq!(
        events.last().expect("completed event")["response"]["output"][0]["content"][0]["text"],
        "hello from Kapsl"
    );
    assert!(!stream.contains("[DONE]"));
}

#[tokio::test]
async fn test_openai_embeddings_rejects_malformed_json_in_openai_error_shape() {
    let response = post_embeddings("{ not json").await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("Invalid embeddings payload"));
}

#[tokio::test]
async fn test_openai_embeddings_returns_not_found_for_unknown_model() {
    let response = post_embeddings(r#"{"model":"missing","input":"hello"}"#).await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("no models are currently loaded"));
}

#[tokio::test]
async fn test_openai_embeddings_rejects_a_generation_model() {
    let response = warp::test::request()
        .method("POST")
        .path("/v1/embeddings")
        .header("content-type", "application/json")
        .body(r#"{"model":"test-model","input":[1,2]}"#)
        .reply(&test_openai_routes_with_model())
        .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("error json");
    assert!(body["error"]["message"]
        .as_str()
        .expect("message")
        .contains("cannot serve embeddings"));
}

#[tokio::test]
async fn test_openai_embeddings_accepts_token_arrays_and_preserves_order() {
    let route = test_openai_routes_with_embedding_model(PathBuf::from("/tmp/embedding.onnx"));
    let response = warp::test::request()
        .method("POST")
        .path("/v1/embeddings")
        .header("content-type", "application/json")
        .body(
            r#"{"model":"test-embedding-model","input":[[11,12],[21]],"encoding_format":"float"}"#,
        )
        .reply(&route)
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("embedding json");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "test-embedding-model");
    assert_eq!(body["data"][0]["object"], "embedding");
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(body["data"][0]["embedding"][0], 11.0);
    assert_eq!(body["data"][1]["index"], 1);
    assert_eq!(body["data"][1]["embedding"][0], 21.0);
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 3);
}

#[tokio::test]
async fn test_openai_embeddings_tokenizes_strings_with_packaged_tokenizer() {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    let root = unique_temp_path("openai-embedding-tokenizer");
    fs::create_dir_all(&root).expect("create tokenizer fixture dir");
    let model_path = root.join("model.onnx");
    fs::write(&model_path, []).expect("create placeholder model");
    let vocabulary = HashMap::from([
        ("[UNK]".to_string(), 0),
        ("[PAD]".to_string(), 1),
        ("hello".to_string(), 2),
        ("world".to_string(), 3),
    ]);
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("word-level model");
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .save(root.join("tokenizer.json"), false)
        .expect("save tokenizer fixture");

    let route = test_openai_routes_with_embedding_model(model_path);
    let response = warp::test::request()
        .method("POST")
        .path("/v1/embeddings")
        .header("content-type", "application/json")
        .body(r#"{"model":"test-embedding-model","input":"hello world"}"#)
        .reply(&route)
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("embedding json");
    assert_eq!(body["data"][0]["embedding"][0], 2.0);
    assert_eq!(body["usage"]["prompt_tokens"], 2);
    fs::remove_dir_all(root).expect("remove tokenizer fixture dir");
}

#[tokio::test]
async fn test_openai_embeddings_base64_encodes_requested_dimensions() {
    let route = test_openai_routes_with_embedding_model(PathBuf::from("/tmp/embedding.onnx"));
    let response = warp::test::request()
        .method("POST")
        .path("/v1/embeddings")
        .header("content-type", "application/json")
        .body(
            r#"{"model":"test-embedding-model","input":[3,4],"dimensions":2,"encoding_format":"base64"}"#,
        )
        .reply(&route)
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(response.body()).expect("embedding json");
    let encoded = body["data"][0]["embedding"]
        .as_str()
        .expect("base64 embedding");
    let bytes = BASE64.decode(encoded).expect("decode embedding");
    assert_eq!(bytes.len(), 2 * std::mem::size_of::<f32>());
    let first = f32::from_le_bytes(bytes[..4].try_into().expect("first float"));
    let second = f32::from_le_bytes(bytes[4..].try_into().expect("second float"));
    assert!((first - (3.0 / 13.0f32.sqrt())).abs() < 1e-6);
    assert!((second - (2.0 / 13.0f32.sqrt())).abs() < 1e-6);
}

#[tokio::test]
async fn test_role_token_routes_never_return_token_values() {
    let auth_state = Arc::new(RwLock::new(ApiAuthState {
        role_tokens: ApiRoleTokenConfig {
            reader_token: Some("reader-secret-value".into()),
            writer_token: None,
            admin_token: Some("admin-secret-value".into()),
        },
        store_path: unique_temp_path("role-token-redaction").join("auth-store.json"),
        store: ApiAuthStoreFile::default(),
        key_hash_index: HashMap::new(),
    }));
    let routes = build_auth_routes(auth_state).admin;

    let read = warp::test::request()
        .method("GET")
        .path("/api/auth/roles")
        .reply(&routes)
        .await;
    assert_eq!(read.status(), StatusCode::OK);
    let body = String::from_utf8_lossy(read.body()).to_string();
    for secret in ["reader-secret-value", "admin-secret-value"] {
        assert!(!body.contains(secret), "GET leaked a token: {body}");
    }
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["reader"]["configured"], true);
    assert_eq!(json["writer"]["configured"], false);
    assert_eq!(json["admin"]["configured"], true);

    let update = warp::test::request()
        .method("POST")
        .path("/api/auth/roles")
        .json(&serde_json::json!({
            "reader_token": "rotated-reader-value",
            "writer_token": "rotated-writer-value",
            "admin_token": "rotated-admin-value",
        }))
        .reply(&routes)
        .await;
    assert_eq!(update.status(), StatusCode::OK);
    let body = String::from_utf8_lossy(update.body()).to_string();
    assert!(!body.contains("rotated-"), "POST leaked a token: {body}");
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["writer"]["configured"], true);
}
