use super::*;

#[test]
fn test_authorization_matches_token_plain_and_bearer() {
    assert!(authorization_matches_token(Some("token123"), "token123"));
    assert!(authorization_matches_token(
        Some("Bearer token123"),
        "token123"
    ));
    assert!(authorization_matches_token(
        Some("bearer token123"),
        "token123"
    ));
}

#[test]
fn test_format_authorization_header_from_remote_token() {
    assert_eq!(
        format_authorization_header(Some("abc123")),
        Some("Bearer abc123".to_string())
    );
    assert_eq!(
        format_authorization_header(Some("Bearer abc123")),
        Some("Bearer abc123".to_string())
    );
}

#[test]
fn test_resolved_remote_token_prefers_explicit_token() {
    assert_eq!(
        resolved_remote_token(DEFAULT_REMOTE_URL, Some("abc123")),
        Some("Bearer abc123".to_string())
    );
    assert_eq!(
        resolved_remote_token(DEFAULT_REMOTE_URL, Some("Bearer abc123")),
        Some("Bearer abc123".to_string())
    );
}

#[test]
fn test_auth_base_url_from_remote_url() {
    assert_eq!(
        auth_base_url_from_remote_url("https://api.kapsl.net/v1").expect("valid"),
        "https://api.kapsl.net"
    );
    assert_eq!(
        auth_base_url_from_remote_url("https://idx.example.com/api/v1").expect("valid"),
        "https://idx.example.com"
    );
    assert!(auth_base_url_from_remote_url("oci://ghcr.io").is_err());
}

#[test]
fn test_resolved_login_remote_url_uses_store_fallback() {
    let old_home = std::env::var_os("HOME");
    let old_remote_url = std::env::var_os(REMOTE_URL_ENV);
    let old_placeholder_url = std::env::var_os(REMOTE_PLACEHOLDER_URL_ENV);
    let unique = format!("test-home-{}", std::process::id());
    let temp_home = std::env::temp_dir().join(unique);
    fs::create_dir_all(&temp_home).expect("create temp home");
    std::env::set_var("HOME", &temp_home);
    std::env::remove_var(REMOTE_URL_ENV);
    std::env::remove_var(REMOTE_PLACEHOLDER_URL_ENV);

    let expected = "https://idx.example.com/v1";
    store_remote_token_for_remote(expected, "Bearer abc").expect("store token");

    assert_eq!(resolved_login_remote_url(None), expected);

    if let Some(value) = old_home {
        std::env::set_var("HOME", value);
    } else {
        std::env::remove_var("HOME");
    }
    if let Some(value) = old_remote_url {
        std::env::set_var(REMOTE_URL_ENV, value);
    } else {
        std::env::remove_var(REMOTE_URL_ENV);
    }
    if let Some(value) = old_placeholder_url {
        std::env::set_var(REMOTE_PLACEHOLDER_URL_ENV, value);
    } else {
        std::env::remove_var(REMOTE_PLACEHOLDER_URL_ENV);
    }
}

#[test]
fn test_is_likely_headless_session_detects_ssh_env() {
    let old_ssh_connection = std::env::var_os("SSH_CONNECTION");
    let old_ssh_client = std::env::var_os("SSH_CLIENT");
    let old_ssh_tty = std::env::var_os("SSH_TTY");
    std::env::remove_var("SSH_CONNECTION");
    std::env::remove_var("SSH_CLIENT");
    std::env::remove_var("SSH_TTY");
    assert!(!is_likely_headless_session());

    std::env::set_var("SSH_CONNECTION", "1");
    assert!(is_likely_headless_session());

    if let Some(value) = old_ssh_connection {
        std::env::set_var("SSH_CONNECTION", value);
    } else {
        std::env::remove_var("SSH_CONNECTION");
    }
    if let Some(value) = old_ssh_client {
        std::env::set_var("SSH_CLIENT", value);
    } else {
        std::env::remove_var("SSH_CLIENT");
    }
    if let Some(value) = old_ssh_tty {
        std::env::set_var("SSH_TTY", value);
    } else {
        std::env::remove_var("SSH_TTY");
    }
}

#[test]
fn test_device_code_login_rejects_non_github_provider() {
    let error = perform_device_code_login_flow(
        "https://idx.example.com/v1",
        OAuthProvider::Google,
        60,
        true,
    )
    .expect_err("expected provider restriction error");
    assert!(error.contains("supports only --provider github"));
}

#[test]
fn test_authorization_mismatch_rejected() {
    assert!(!authorization_matches_token(
        Some("Bearer wrong"),
        "token123"
    ));
    assert!(!authorization_matches_token(Some("wrong"), "token123"));
    assert!(!authorization_matches_token(None, "token123"));
}

#[test]
fn test_api_role_resolution_and_hierarchy() {
    let config = ApiRoleTokenConfig {
        reader_token: Some("reader-token".to_string()),
        writer_token: Some("writer-token".to_string()),
        admin_token: Some("admin-token".to_string()),
    };

    assert_eq!(
        config.role_from_authorization_header(Some("Bearer reader-token")),
        Some(ApiRole::Reader)
    );
    assert_eq!(
        config.role_from_authorization_header(Some("writer-token")),
        Some(ApiRole::Writer)
    );
    assert_eq!(
        config.role_from_authorization_header(Some("Bearer admin-token")),
        Some(ApiRole::Admin)
    );
    assert_eq!(config.role_from_authorization_header(Some("invalid")), None);

    assert!(ApiRole::Admin.allows(ApiRole::Admin));
    assert!(ApiRole::Admin.allows(ApiRole::Writer));
    assert!(ApiRole::Admin.allows(ApiRole::Reader));
    assert!(ApiRole::Writer.allows(ApiRole::Writer));
    assert!(ApiRole::Writer.allows(ApiRole::Reader));
    assert!(!ApiRole::Writer.allows(ApiRole::Admin));
    assert!(!ApiRole::Reader.allows(ApiRole::Writer));
}

#[test]
fn test_api_key_scope_hierarchy_and_wildcards() {
    assert!(key_scopes_allow(&["*:*".to_string()], ApiScope::Admin));
    assert!(key_scopes_allow(&["api:*".to_string()], ApiScope::Write));
    assert!(key_scopes_allow(&["api:admin".to_string()], ApiScope::Read));
    assert!(key_scopes_allow(&["write".to_string()], ApiScope::Read));
    assert!(!key_scopes_allow(
        &["api:read".to_string()],
        ApiScope::Write
    ));
    assert!(!key_scopes_allow(&["read".to_string()], ApiScope::Admin));
    assert!(key_scopes_allow(&[], ApiScope::Admin));
}

#[test]
fn test_loopback_remote_detection() {
    let loopback: std::net::SocketAddr = "127.0.0.1:9095".parse().expect("valid socket");
    let non_loopback: std::net::SocketAddr = "10.1.2.3:9095".parse().expect("valid socket");
    assert!(is_loopback_remote(Some(loopback)));
    assert!(!is_loopback_remote(Some(non_loopback)));
    assert!(!is_loopback_remote(None));
}

#[test]
fn test_api_role_update_requires_admin_token_when_enabled() {
    let mut config = ApiRoleTokenConfig::default();

    let result = config.update_from_payload(ApiRoleTokenConfig {
        reader_token: Some("reader-token".to_string()),
        writer_token: None,
        admin_token: None,
    });
    assert!(result.is_err());

    let result = config.update_from_payload(ApiRoleTokenConfig {
        reader_token: Some("reader-token".to_string()),
        writer_token: Some("writer-token".to_string()),
        admin_token: Some("admin-token".to_string()),
    });
    assert!(result.is_ok());
    assert!(config.auth_enabled());
}

#[test]
fn test_redact_identifier_for_logs() {
    assert_eq!(
        redact_identifier_for_logs("request-abcde", false),
        "requ...[redacted]"
    );
    assert_eq!(
        redact_identifier_for_logs("request-abcde", true),
        "request-abcde"
    );
    assert_eq!(redact_identifier_for_logs("-", false), "-");
}

fn make_test_auth_state() -> ApiAuthState {
    let now = now_unix_seconds();
    let mut bytes = [0u8; 4];
    OsRng.fill_bytes(&mut bytes);
    let unique_suffix = bytes
        .iter()
        .map(|byte| format!("{:02x}", byte))
        .collect::<String>();
    let store_path = std::env::temp_dir().join(format!("kapsl-auth-state-{}.json", unique_suffix));
    ApiAuthState {
        legacy_tokens: ApiRoleTokenConfig::default(),
        store_path,
        store: ApiAuthStoreFile {
            users: vec![
                ApiAuthUser {
                    id: "user-admin".to_string(),
                    username: "admin".to_string(),
                    display_name: Some("Admin".to_string()),
                    role: ApiRole::Admin,
                    status: ApiUserStatus::Active,
                    created_at: now,
                    updated_at: now,
                },
                ApiAuthUser {
                    id: "user-reader".to_string(),
                    username: "reader".to_string(),
                    display_name: Some("Reader".to_string()),
                    role: ApiRole::Reader,
                    status: ApiUserStatus::Active,
                    created_at: now,
                    updated_at: now,
                },
            ],
            api_keys: Vec::new(),
        },
        key_hash_index: HashMap::new(),
    }
}

#[test]
fn test_first_api_key_must_be_admin() {
    let mut state = make_test_auth_state();
    let result = state.create_api_key(
        "user-reader",
        CreateApiKeyRequest {
            name: "reader-key".to_string(),
            scopes: None,
            expires_in_days: None,
        },
    );
    assert!(result.is_err());
    let _ = fs::remove_file(&state.store_path);
}

#[test]
fn test_generated_api_key_authenticates_user_role() {
    let mut state = make_test_auth_state();
    let created = state
        .create_api_key(
            "user-admin",
            CreateApiKeyRequest {
                name: "admin-key".to_string(),
                scopes: Some(vec!["*:*".to_string()]),
                expires_in_days: Some(30),
            },
        )
        .expect("create key");
    assert_eq!(
        state.role_from_authorization_header(Some(&format!("Bearer {}", created.raw_key))),
        Some(ApiRole::Admin)
    );
    let _ = fs::remove_file(&state.store_path);
}

#[test]
fn test_generated_api_key_preserves_scopes_for_authorization() {
    let mut state = make_test_auth_state();
    let created = state
        .create_api_key(
            "user-admin",
            CreateApiKeyRequest {
                name: "scoped-admin-key".to_string(),
                scopes: Some(vec!["api:read".to_string()]),
                expires_in_days: Some(30),
            },
        )
        .expect("create key");

    let grant = state
        .grant_from_authorization_header_read(Some(&format!("Bearer {}", created.raw_key)))
        .expect("grant");
    let ApiAuthGrantMatch {
        grant: ApiAuthGrant { role, scopes },
        matched_key_index,
    } = grant;
    assert!(matched_key_index.is_some());
    let scopes = scopes.expect("scopes");
    assert_eq!(role, ApiRole::Admin);
    assert!(key_scopes_allow(&scopes, ApiScope::Read));
    assert!(!key_scopes_allow(&scopes, ApiScope::Write));
    assert!(!key_scopes_allow(&scopes, ApiScope::Admin));
    let _ = fs::remove_file(&state.store_path);
}

fn parse_and_tune(argv: &[&str]) -> (Args, AppliedPerformanceTuning) {
    let argv: Vec<String> = argv.iter().map(|arg| (*arg).to_string()).collect();
    let (mut args, matches) = parse_runtime_args_and_matches(&argv).expect("valid args");
    let tuning = apply_performance_profile(&mut args, &matches);
    (args, tuning)
}

fn onnx_tuning_env_names() -> [&'static str; 6] {
    [
        ORT_MEMORY_PATTERN_ENV,
        ORT_DISABLE_CPU_MEM_ARENA_ENV,
        ORT_SESSION_BUCKETS_ENV,
        ORT_BUCKET_DIM_GRANULARITY_ENV,
        ORT_BUCKET_MAX_DIMS_ENV,
        MODEL_PEAK_CONCURRENCY_ENV,
    ]
}

fn env_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

fn lock_env_tests() -> std::sync::MutexGuard<'static, ()> {
    env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn save_env(names: &[&'static str]) -> Vec<(&'static str, Option<std::ffi::OsString>)> {
    names
        .iter()
        .map(|name| (*name, std::env::var_os(*name)))
        .collect()
}

fn restore_env(saved: Vec<(&'static str, Option<std::ffi::OsString>)>) {
    for (name, value) in saved {
        if let Some(value) = value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }
}

fn clear_env(names: &[&str]) {
    for name in names {
        std::env::remove_var(name);
    }
}

#[test]
fn test_throughput_profile_tunes_defaults() {
    let (args, tuning) = parse_and_tune(&["kapsl", "--performance-profile", "throughput"]);
    assert_eq!(args.batch_size, 16);
    assert_eq!(args.transport, "hybrid");
    assert_eq!(args.scheduler_queue_size, 2048);
    assert_eq!(args.scheduler_max_micro_batch, 16);
    assert_eq!(args.scheduler_queue_delay_ms, 6);
    assert_eq!(tuning.batch_size, Some(16));
    assert_eq!(tuning.transport.as_deref(), Some("hybrid"));
    assert_eq!(tuning.scheduler_queue_size, Some(2048));
    assert_eq!(tuning.scheduler_max_micro_batch, Some(16));
    assert_eq!(tuning.scheduler_queue_delay_ms, Some(6));
    assert_eq!(tuning.media_preprocess, None);
}

#[test]
fn test_profile_does_not_override_explicit_batch_or_transport() {
    let (args, tuning) = parse_and_tune(&[
        "kapsl",
        "--performance-profile",
        "throughput",
        "--batch-size",
        "2",
        "--transport",
        "tcp",
    ]);
    assert_eq!(args.batch_size, 2);
    assert_eq!(args.transport, "tcp");
    assert_eq!(tuning.batch_size, None);
    assert_eq!(tuning.transport, None);
}

#[test]
fn test_standard_profile_keeps_existing_defaults() {
    // Explicitly pass --performance-profile standard to verify Standard is a no-op.
    let (args, tuning) = parse_and_tune(&["kapsl", "--performance-profile", "standard"]);
    assert_eq!(args.batch_size, 4);
    assert_eq!(args.transport, "socket");
    assert_eq!(args.scheduler_queue_size, 256);
    assert_eq!(args.scheduler_max_micro_batch, 4);
    assert_eq!(args.scheduler_queue_delay_ms, 2);
    assert_eq!(args.performance_profile, PerformanceProfile::Standard);
    assert_eq!(tuning.batch_size, None);
    assert_eq!(tuning.transport, None);
    assert_eq!(tuning.scheduler_queue_size, None);
    assert_eq!(tuning.scheduler_max_micro_batch, None);
    assert_eq!(tuning.scheduler_queue_delay_ms, None);
}

#[test]
fn test_auto_profile_is_default_and_uses_conservative_defaults_with_no_model() {
    // No --model → model_size_mb=0 → conservative defaults (same values as Standard).
    // tuning fields are Some because Auto applied them.
    let (args, tuning) = parse_and_tune(&["kapsl"]);
    assert_eq!(args.performance_profile, PerformanceProfile::Auto);
    assert_eq!(args.batch_size, 4);
    assert_eq!(args.scheduler_queue_size, 256);
    assert_eq!(args.scheduler_max_micro_batch, 4);
    assert_eq!(args.scheduler_queue_delay_ms, 2);
    assert!(tuning.batch_size.is_some());
    assert!(tuning.auto_tune_rationale.is_some());
    let rationale = tuning.auto_tune_rationale.unwrap();
    assert!(
        rationale.contains("unknown"),
        "rationale should say unknown: {}",
        rationale
    );
}

#[test]
fn test_auto_profile_respects_explicit_batch_size() {
    let (args, tuning) = parse_and_tune(&["kapsl", "--batch-size", "1"]);
    assert_eq!(args.performance_profile, PerformanceProfile::Auto);
    assert_eq!(args.batch_size, 1); // user explicit value preserved
    assert_eq!(tuning.batch_size, None); // not overridden by auto-tune
}

#[test]
fn test_auto_tuned_gguf_prefill_chunk_uses_memory_headroom() {
    assert_eq!(
        auto_tuned_gguf_prefill_chunk_size(9_000, 5 * 1024, 1),
        Some(32)
    );
    assert_eq!(
        auto_tuned_gguf_prefill_chunk_size(3_000, 16 * 1024, 4),
        Some(128)
    );
    assert_eq!(
        auto_tuned_gguf_prefill_chunk_size(500, 64 * 1024, 1),
        Some(512)
    );
    assert_eq!(auto_tuned_gguf_prefill_chunk_size(0, 16 * 1024, 4), None);
}

#[test]
fn test_auto_profile_does_not_export_gguf_prefill_chunk_before_model_resolution() {
    static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let _guard = ENV_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();

    let old_prefill = std::env::var_os(GGUF_PREFILL_CHUNK_SIZE_ENV);
    std::env::remove_var(GGUF_PREFILL_CHUNK_SIZE_ENV);

    let model_path = std::env::temp_dir().join(format!(
        "kapsl-gguf-prefill-autotune-{}-{}.gguf",
        std::process::id(),
        now_unix_seconds()
    ));
    let file = File::create(&model_path).expect("create sparse test model");
    file.set_len(9 * 1024 * 1024 * 1024)
        .expect("size sparse test model");
    drop(file);

    let (_args, _tuning) = parse_and_tune(&[
        "kapsl",
        "--model",
        model_path.to_str().expect("utf8 temp path"),
    ]);

    assert!(std::env::var_os(GGUF_PREFILL_CHUNK_SIZE_ENV).is_none());

    let _ = fs::remove_file(&model_path);
    if let Some(value) = old_prefill {
        std::env::set_var(GGUF_PREFILL_CHUNK_SIZE_ENV, value);
    } else {
        std::env::remove_var(GGUF_PREFILL_CHUNK_SIZE_ENV);
    }
}

#[test]
fn test_onnx_tuning_profile_resolves_global_and_per_model_overrides() {
    let _guard = lock_env_tests();
    let env_names = onnx_tuning_env_names();
    let saved_env = save_env(&env_names);
    clear_env(&env_names);

    let (args, _) = parse_and_tune(&[
        "kapsl",
        "--onnx-memory-pattern",
        "false",
        "--onnx-model-tuning",
        "*:session_buckets=2",
        "--onnx-model-tuning",
        "7:disable_cpu_mem_arena=true,peak_concurrency=8",
    ]);

    let profile = build_onnx_tuning_profile(&args).expect("valid ONNX tuning profile");
    let model_7 = profile.resolve(7);
    assert_eq!(model_7.memory_pattern, Some(false));
    assert_eq!(model_7.session_buckets, Some(2));
    assert_eq!(model_7.disable_cpu_mem_arena, Some(true));
    assert_eq!(model_7.peak_concurrency_hint, Some(8));

    let model_9 = profile.resolve(9);
    assert_eq!(model_9.memory_pattern, Some(false));
    assert_eq!(model_9.session_buckets, Some(2));
    assert_eq!(model_9.disable_cpu_mem_arena, Some(false));
    assert_eq!(
        model_9.peak_concurrency_hint,
        auto_onnx_runtime_tuning(&args).peak_concurrency_hint
    );

    restore_env(saved_env);
}

#[test]
fn test_onnx_tuning_profile_uses_env_as_override_below_cli() {
    let _guard = lock_env_tests();
    let env_names = onnx_tuning_env_names();
    let saved_env = save_env(&env_names);
    clear_env(&env_names);
    std::env::set_var(ORT_MEMORY_PATTERN_ENV, "false");
    std::env::set_var(ORT_SESSION_BUCKETS_ENV, "6");
    std::env::set_var(MODEL_PEAK_CONCURRENCY_ENV, "3");

    let (args, _) = parse_and_tune(&[
        "kapsl",
        "--onnx-session-buckets",
        "2",
        "--onnx-model-tuning",
        "7:peak_concurrency=8",
    ]);

    let profile = build_onnx_tuning_profile(&args).expect("valid ONNX tuning profile");
    let model_9 = profile.resolve(9);
    assert_eq!(model_9.memory_pattern, Some(false));
    assert_eq!(model_9.session_buckets, Some(2));
    assert_eq!(model_9.peak_concurrency_hint, Some(3));

    let model_7 = profile.resolve(7);
    assert_eq!(model_7.peak_concurrency_hint, Some(8));

    restore_env(saved_env);
}

#[test]
fn test_onnx_tuning_profile_rejects_unknown_keys() {
    let _guard = lock_env_tests();
    let env_names = onnx_tuning_env_names();
    let saved_env = save_env(&env_names);
    clear_env(&env_names);

    let (args, _) = parse_and_tune(&["kapsl", "--onnx-model-tuning", "3:not_a_real_key=1"]);
    let err = build_onnx_tuning_profile(&args).expect_err("unknown key should fail");
    assert!(err.contains("unknown ONNX tuning key"));

    restore_env(saved_env);
}

#[test]
fn test_media_validation_rejects_shape_mismatch() {
    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 3, 128, 128],
        dtype: TensorDtype::Float32,
        data: vec![0; 3 * 128 * 128 * 4],
    });
    let model_info = EngineModelInfo {
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        input_shapes: vec![vec![1, 3, 224, 224]],
        output_shapes: vec![vec![]],
        input_dtypes: vec!["float32".to_string()],
        output_dtypes: vec![],
        framework: Some("onnx".to_string()),
        model_version: None,
        peak_concurrency: None,
    };
    let result = validate_inference_request_against_model_info(&request, &model_info);
    assert!(result.is_err());
    assert!(result
        .expect_err("shape mismatch should fail")
        .contains("shape mismatch"));
}

#[test]
fn test_media_validation_accepts_dynamic_dims_and_dtype_match() {
    let request = InferenceRequest::new(BinaryTensorPacket {
        shape: vec![1, 3, 320, 320],
        dtype: TensorDtype::Float32,
        data: vec![0; 3 * 320 * 320 * 4],
    });
    let model_info = EngineModelInfo {
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        input_shapes: vec![vec![1, 3, -1, -1]],
        output_shapes: vec![vec![]],
        input_dtypes: vec!["float32".to_string()],
        output_dtypes: vec![],
        framework: Some("onnx".to_string()),
        model_version: None,
        peak_concurrency: None,
    };
    let result = validate_inference_request_against_model_info(&request, &model_info);
    assert!(result.is_ok());
}

#[test]
fn test_status_code_for_engine_error_overloaded_and_timeout() {
    let overloaded = EngineError::overloaded("queue full");
    assert_eq!(
        status_code_for_engine_error(&overloaded),
        warp::http::StatusCode::TOO_MANY_REQUESTS
    );

    let timeout = EngineError::timeout("deadline exceeded");
    assert_eq!(
        status_code_for_engine_error(&timeout),
        warp::http::StatusCode::GATEWAY_TIMEOUT
    );
}

#[test]
fn test_status_code_for_engine_error_backend_and_input() {
    let backend = EngineError::backend("runtime panic");
    assert_eq!(
        status_code_for_engine_error(&backend),
        warp::http::StatusCode::INTERNAL_SERVER_ERROR
    );

    let invalid = EngineError::invalid_input("bad tensor shape");
    assert_eq!(
        status_code_for_engine_error(&invalid),
        warp::http::StatusCode::BAD_REQUEST
    );
}

#[test]
fn test_parse_queue_overflow_policy_literal() {
    assert_eq!(
        parse_queue_overflow_policy_literal("block"),
        Some(kapsl_scheduler::QueueOverflowPolicy::Block)
    );
    assert_eq!(
        parse_queue_overflow_policy_literal("latest_only"),
        Some(kapsl_scheduler::QueueOverflowPolicy::DropNewest)
    );
    assert_eq!(
        parse_queue_overflow_policy_literal("drop_oldest"),
        Some(kapsl_scheduler::QueueOverflowPolicy::DropOldest)
    );
    assert_eq!(parse_queue_overflow_policy_literal("unknown"), None);
}

#[test]
fn test_manifest_queue_overflow_policy_prefers_runtime_server_path() {
    let metadata: serde_yaml::Value = serde_yaml::from_str(
        r#"
runtime:
  server:
queue_overflow_policy: drop_oldest
"#,
    )
    .expect("parse yaml");
    let manifest = Manifest {
        project_name: "demo".to_string(),
        framework: "onnx".to_string(),
        version: "1.0.0".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        model_file: "model.onnx".to_string(),
        format: None,
        model_type: None,
        task: None,
        metadata: Some(metadata),
        hardware_requirements: kapsl_core::HardwareRequirements::default(),
        cron_jobs: Vec::new(),
    };

    let policy = manifest_queue_overflow_policy(&manifest);
    assert_eq!(
        policy,
        Some(kapsl_scheduler::QueueOverflowPolicy::DropOldest)
    );
}

#[test]
fn test_effective_topology_choice_falls_back_tensor_parallel_to_data_parallel() {
    let manifest = Manifest {
        project_name: "demo".to_string(),
        framework: "onnx".to_string(),
        version: "1.0.0".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        model_file: "model.onnx".to_string(),
        format: None,
        model_type: None,
        task: None,
        metadata: None,
        hardware_requirements: kapsl_core::HardwareRequirements::default(),
        cron_jobs: Vec::new(),
    };

    let choice = resolve_effective_topology_choice(&manifest, "tensor-parallel", 4, None);
    assert!(!choice.use_pipeline_backend);
    assert_eq!(choice.worker_topology, "data-parallel");
    assert_eq!(choice.worker_tp_degree, 1);
    assert!(matches!(
        choice.mesh_topology,
        kapsl_hal::device_mesh::MeshTopology::DataParallel
    ));
}

#[test]
fn test_effective_topology_choice_uses_pipeline_metadata_stage_count() {
    let metadata: serde_yaml::Value = serde_yaml::from_str(
        r#"
llm:
  pipeline:
    stages:
      - stage0.onnx
      - stage1.onnx
"#,
    )
    .expect("parse yaml");
    let manifest = Manifest {
        project_name: "demo".to_string(),
        framework: "llm".to_string(),
        version: "1.0.0".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        model_file: "model.onnx".to_string(),
        format: None,
        model_type: None,
        task: None,
        metadata: Some(metadata),
        hardware_requirements: kapsl_core::HardwareRequirements::default(),
        cron_jobs: Vec::new(),
    };
    let stages = manifest_llm_pipeline_stages(&manifest);
    assert_eq!(
        stages.as_deref(),
        Some(&["stage0.onnx".to_string(), "stage1.onnx".to_string()][..])
    );

    let choice =
        resolve_effective_topology_choice(&manifest, "pipeline-parallel", 8, stages.as_deref());
    assert!(choice.use_pipeline_backend);
    assert_eq!(choice.worker_topology, "pipeline-parallel");
    assert_eq!(choice.worker_tp_degree, 2);
    assert!(matches!(
        choice.mesh_topology,
        kapsl_hal::device_mesh::MeshTopology::PipelineParallel { stages: 2 }
    ));
}

#[test]
fn test_effective_topology_choice_falls_back_without_pipeline_metadata() {
    let manifest = Manifest {
        project_name: "demo".to_string(),
        framework: "llm".to_string(),
        version: "1.0.0".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        model_file: "model.onnx".to_string(),
        format: None,
        model_type: None,
        task: None,
        metadata: None,
        hardware_requirements: kapsl_core::HardwareRequirements::default(),
        cron_jobs: Vec::new(),
    };

    let choice = resolve_effective_topology_choice(&manifest, "pipeline-parallel", 4, None);
    assert!(!choice.use_pipeline_backend);
    assert_eq!(choice.worker_topology, "data-parallel");
    assert_eq!(choice.worker_tp_degree, 1);
    assert!(matches!(
        choice.mesh_topology,
        kapsl_hal::device_mesh::MeshTopology::DataParallel
    ));
}

#[test]
fn test_evaluate_runtime_pressure_state_transitions() {
    let config = RuntimePressureConfig {
        memory_conserve_ratio: 0.7,
        memory_emergency_ratio: 0.9,
        gpu_util_conserve_ratio: 0.8,
        gpu_util_emergency_ratio: 0.95,
        gpu_mem_conserve_ratio: 0.8,
        gpu_mem_emergency_ratio: 0.95,
        conserve_max_new_tokens: Some(256),
        emergency_max_new_tokens: Some(128),
    };

    let normal = RuntimeSamples {
        process_memory_bytes: 2 * 1024,
        total_system_memory_bytes: Some(10 * 1024),
        gpu_utilization: 0.2,
        gpu_memory_bytes: Some(100),
        gpu_memory_total_bytes: Some(1000),
        collected_at_ms: 0,
    };
    assert_eq!(
        evaluate_runtime_pressure_state(&normal, &config),
        RuntimePressureState::Normal
    );

    let conserve = RuntimeSamples {
        process_memory_bytes: 8 * 1024,
        total_system_memory_bytes: Some(10 * 1024),
        ..normal.clone()
    };
    assert_eq!(
        evaluate_runtime_pressure_state(&conserve, &config),
        RuntimePressureState::Conserve
    );

    let emergency = RuntimeSamples {
        gpu_utilization: 0.97,
        ..normal
    };
    assert_eq!(
        evaluate_runtime_pressure_state(&emergency, &config),
        RuntimePressureState::Emergency
    );
}
