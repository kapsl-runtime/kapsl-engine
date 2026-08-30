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

fn test_openai_routes() -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    let models = ModelManager::new(Arc::new(ModelRegistry::new()));
    build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference: test_inference_service(models),
        log_sensitive_ids: false,
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
            .chunks_exact(8)
            .next()
            .map(|bytes| i64::from_ne_bytes(bytes.try_into().expect("int64 token")))
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
        log_sensitive_ids: false,
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
        log_sensitive_ids: false,
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
        log_sensitive_ids: false,
    })
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
