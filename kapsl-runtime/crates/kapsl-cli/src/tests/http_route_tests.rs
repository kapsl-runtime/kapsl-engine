use super::*;
use std::net::SocketAddr;
use warp::http::StatusCode;

fn test_device_info() -> Arc<DeviceInfo> {
    Arc::new(DeviceInfo {
        cpu_cores: 1,
        total_memory: 0,
        os_type: "test".to_string(),
        os_release: "test".to_string(),
        has_cuda: false,
        has_metal: false,
        has_rocm: false,
        has_directml: false,
        devices: Vec::new(),
    })
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
    let model_registry = Arc::new(ModelRegistry::new());
    let replica_pools: ReplicaPools = Arc::new(RwLock::new(HashMap::new()));
    let runtime_samples = Arc::new(RwLock::new(RuntimeSamples {
        process_memory_bytes: 123,
        total_system_memory_bytes: Some(456),
        gpu_utilization: 7.5,
        gpu_memory_bytes: Some(10),
        gpu_memory_total_bytes: Some(20),
        collected_at_ms: 789,
    }));
    let pressure_state = Arc::new(AtomicU8::new(RuntimePressureState::Conserve as u8));
    let routes = build_system_routes(
        model_registry,
        replica_pools,
        test_device_info(),
        runtime_samples,
        pressure_state,
    );

    let health = warp::test::request()
        .path("/api/health")
        .reply(&routes)
        .await;
    assert_eq!(health.status(), StatusCode::OK);
    let health_json: serde_json::Value =
        serde_json::from_slice(health.body()).expect("health json");
    assert_eq!(health_json["status"], "healthy");
    assert_eq!(health_json["total_models"], 0);

    let stats = warp::test::request()
        .path("/api/system/stats")
        .reply(&routes)
        .await;
    assert_eq!(stats.status(), StatusCode::OK);
    let stats_json: serde_json::Value = serde_json::from_slice(stats.body()).expect("stats json");
    assert_eq!(stats_json["process_memory_bytes"], 123);
    assert_eq!(stats_json["pressure_state"], "conserve");
}

#[tokio::test]
async fn test_auth_login_route_allows_local_loopback_when_auth_disabled() {
    let auth_state = Arc::new(RwLock::new(ApiAuthState {
        legacy_tokens: ApiRoleTokenConfig::default(),
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
async fn test_rag_query_route_rejects_missing_workspace_before_store_access() {
    let rag_root = unique_temp_path("rag");
    let docs_root = rag_root.join("docs");
    fs::create_dir_all(&docs_root).expect("create rag docs dir");
    let rag_state = RagRuntimeState {
        vector_store: Arc::new(
            SqliteVectorStore::open(&rag_root.join("vectors.sqlite3")).expect("open vector store"),
        ),
        doc_store: FsDocStore::new(&docs_root),
    };
    let routes = build_rag_routes(rag_state);

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
