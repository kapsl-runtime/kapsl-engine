use super::*;
use kapsl_core::HardwareRequirements;
use std::sync::{Mutex, OnceLock};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn test_manifest(framework: &str) -> Manifest {
    Manifest {
        project_name: "test".to_string(),
        framework: framework.to_string(),
        version: "1.0.0".to_string(),
        created_at: "0".to_string(),
        model_file: match framework {
            "gguf" => "model.gguf".to_string(),
            "llm" => "model.onnx".to_string(),
            _ => "model.bin".to_string(),
        },
        format: None,
        model_type: None,
        task: None,
        metadata: None,
        hardware_requirements: HardwareRequirements::default(),
        cron_jobs: Vec::new(),
    }
}

fn test_manifest_with_llm_metadata(metadata: &str) -> Manifest {
    let mut manifest = test_manifest("llm");
    manifest.metadata = Some(serde_yaml::from_str(metadata).expect("valid metadata"));
    manifest
}

#[test]
fn exports_batch_size_hint_for_gguf_models() {
    let _guard = env_lock().lock().unwrap();
    std::env::remove_var(GGUF_MAX_CONCURRENT_ENV);
    std::env::remove_var(GGUF_TARGET_CONCURRENCY_ENV);

    export_gguf_auto_sizing_hint(&test_manifest("gguf"), 3, None);

    assert!(std::env::var_os(GGUF_MAX_CONCURRENT_ENV).is_none());
    assert_eq!(
        std::env::var(GGUF_TARGET_CONCURRENCY_ENV).ok().as_deref(),
        Some("3")
    );

    std::env::remove_var(GGUF_MAX_CONCURRENT_ENV);
    std::env::remove_var(GGUF_TARGET_CONCURRENCY_ENV);
}

#[test]
fn does_not_export_batch_size_hint_for_onnx_generate_models() {
    let _guard = env_lock().lock().unwrap();
    std::env::remove_var(GGUF_MAX_CONCURRENT_ENV);
    std::env::remove_var(GGUF_TARGET_CONCURRENCY_ENV);

    export_gguf_auto_sizing_hint(&test_manifest("llm"), 3, None);

    assert!(std::env::var_os(GGUF_MAX_CONCURRENT_ENV).is_none());
    assert!(std::env::var_os(GGUF_TARGET_CONCURRENCY_ENV).is_none());
}

#[test]
fn gguf_auto_sizing_respects_manual_env_values() {
    let _guard = env_lock().lock().unwrap();
    let old_prefill = std::env::var_os(GGUF_PREFILL_CHUNK_SIZE_ENV);
    let old_max = std::env::var_os(GGUF_MAX_CONCURRENT_ENV);
    let old_target = std::env::var_os(GGUF_TARGET_CONCURRENCY_ENV);
    std::env::remove_var(GGUF_PREFILL_CHUNK_SIZE_ENV);
    std::env::remove_var(GGUF_MAX_CONCURRENT_ENV);
    std::env::remove_var(GGUF_TARGET_CONCURRENCY_ENV);
    std::env::set_var(GGUF_PREFILL_CHUNK_SIZE_ENV, "999");
    std::env::set_var(GGUF_MAX_CONCURRENT_ENV, "99");
    std::env::set_var(GGUF_TARGET_CONCURRENCY_ENV, "99");

    let model_path = std::env::temp_dir().join(format!(
        "kapsl-gguf-resolved-prefill-{}-{}.gguf",
        std::process::id(),
        now_unix_seconds()
    ));
    let file = File::create(&model_path).expect("create sparse test model");
    file.set_len(3 * 1024 * 1024 * 1024)
        .expect("size sparse test model");
    drop(file);

    export_gguf_auto_sizing_hint(&test_manifest("gguf"), 4, Some(model_path.as_path()));

    assert_eq!(
        std::env::var(GGUF_PREFILL_CHUNK_SIZE_ENV).ok().as_deref(),
        Some("999")
    );
    assert_eq!(
        std::env::var(GGUF_MAX_CONCURRENT_ENV).ok().as_deref(),
        Some("99")
    );
    assert_eq!(
        std::env::var(GGUF_TARGET_CONCURRENCY_ENV).ok().as_deref(),
        Some("99")
    );

    let _ = fs::remove_file(&model_path);
    if let Some(value) = old_prefill {
        std::env::set_var(GGUF_PREFILL_CHUNK_SIZE_ENV, value);
    } else {
        std::env::remove_var(GGUF_PREFILL_CHUNK_SIZE_ENV);
    }
    if let Some(value) = old_max {
        std::env::set_var(GGUF_MAX_CONCURRENT_ENV, value);
    } else {
        std::env::remove_var(GGUF_MAX_CONCURRENT_ENV);
    }
    if let Some(value) = old_target {
        std::env::set_var(GGUF_TARGET_CONCURRENCY_ENV, value);
    } else {
        std::env::remove_var(GGUF_TARGET_CONCURRENCY_ENV);
    }
}

#[test]
fn llm_isolation_metadata_is_advisory_unless_strict_or_env_forces_it() {
    let _guard = env_lock().lock().unwrap();
    let old_isolate = std::env::var_os(LLM_ISOLATE_PROCESS_ENV);
    let old_strict = std::env::var_os(LLM_ISOLATE_PROCESS_STRICT_ENV);
    std::env::remove_var(LLM_ISOLATE_PROCESS_ENV);
    std::env::remove_var(LLM_ISOLATE_PROCESS_STRICT_ENV);

    let advisory = test_manifest_with_llm_metadata("llm:\n  isolate_process: true\n");
    assert!(!resolve_isolate_process(&advisory));

    let strict = test_manifest_with_llm_metadata(
        "llm:\n  isolate_process: true\n  isolate_process_strict: true\n",
    );
    assert!(resolve_isolate_process(&strict));

    std::env::set_var(LLM_ISOLATE_PROCESS_ENV, "1");
    assert!(resolve_isolate_process(&advisory));
    std::env::set_var(LLM_ISOLATE_PROCESS_ENV, "0");
    assert!(!resolve_isolate_process(&strict));

    if let Some(value) = old_isolate {
        std::env::set_var(LLM_ISOLATE_PROCESS_ENV, value);
    } else {
        std::env::remove_var(LLM_ISOLATE_PROCESS_ENV);
    }
    if let Some(value) = old_strict {
        std::env::set_var(LLM_ISOLATE_PROCESS_STRICT_ENV, value);
    } else {
        std::env::remove_var(LLM_ISOLATE_PROCESS_STRICT_ENV);
    }
}
