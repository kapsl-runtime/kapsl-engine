use super::*;

#[test]
fn test_context_build_creates_source_metadata_when_missing() {
    let temp_dir = TempDirGuard::new("kapsl-build-metadata").expect("temp dir");
    let context_dir = temp_dir.path();
    fs::write(context_dir.join("model.onnx"), b"dummy model").expect("model file");

    let response = create_kapsl_package_from_context(
        context_dir,
        None,
        None,
        None,
        None,
        None,
        None,
        AxisOverrides::default(),
        false,
    )
    .expect("context package");

    let metadata_path = context_dir.join("metadata.json");
    let response_metadata_path = PathBuf::from(
        response
            .metadata_path
            .as_deref()
            .expect("created metadata path"),
    );
    assert_eq!(
        response_metadata_path
            .canonicalize()
            .expect("response path"),
        metadata_path.canonicalize().expect("metadata path")
    );
    assert!(metadata_path.exists());

    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&metadata_path).expect("metadata"))
            .expect("valid metadata json");
    assert_eq!(manifest["model_file"], "model.onnx");
    assert_eq!(manifest["framework"], "onnx");
    assert_eq!(manifest["version"], "1.0.0");
}

#[test]
fn test_context_build_does_not_overwrite_existing_source_metadata() {
    let temp_dir = TempDirGuard::new("kapsl-build-existing-metadata").expect("temp dir");
    let context_dir = temp_dir.path();
    fs::write(context_dir.join("model.onnx"), b"dummy model").expect("model file");
    let existing_metadata = r#"{
  "project_name": "existing-project",
  "framework": "onnx",
  "version": "2.0.0",
  "model_file": "model.onnx",
  "metadata": {
"owner": "test"
  }
}"#;
    let metadata_path = context_dir.join("metadata.json");
    fs::write(&metadata_path, existing_metadata).expect("metadata file");

    let response = create_kapsl_package_from_context(
        context_dir,
        None,
        None,
        None,
        None,
        None,
        None,
        AxisOverrides::default(),
        false,
    )
    .expect("context package");

    assert_eq!(response.metadata_path, None);
    assert_eq!(
        fs::read_to_string(&metadata_path).expect("metadata"),
        existing_metadata
    );
    assert_eq!(response.project_name, "existing-project");
    assert_eq!(response.version, "2.0.0");
}

#[test]
fn test_prompt_with_default_falls_back_on_empty_input() {
    let mut reader = Cursor::new(b"\n".to_vec());
    let value = prompt_with_default(&mut reader, "Version", "1.0.0").expect("prompt");
    assert_eq!(value, "1.0.0");
}

#[test]
fn test_prompt_with_default_trims_provided_value() {
    let mut reader = Cursor::new(b"  custom-name  \n".to_vec());
    let value = prompt_with_default(&mut reader, "Project name", "kapsl-model").expect("prompt");
    assert_eq!(value, "custom-name");
}

#[test]
fn test_prompt_non_empty_retries_until_value_provided() {
    // First line is whitespace-only (rejected), second line is accepted.
    let mut reader = Cursor::new(b"   \nactual-framework\n".to_vec());
    let value = prompt_non_empty_with_default(&mut reader, "Framework", "").expect("prompt");
    assert_eq!(value, "actual-framework");
}

#[test]
fn test_prompt_provider_defaults_to_cpu_on_empty_input() {
    let mut reader = Cursor::new(b"\n".to_vec());
    let value = prompt_provider_with_default(&mut reader, None).expect("prompt");
    assert_eq!(value.as_deref(), Some("cpu"));
}

#[test]
fn test_prompt_provider_uses_existing_default_on_empty_input() {
    let mut reader = Cursor::new(b"\n".to_vec());
    let value = prompt_provider_with_default(&mut reader, Some("cuda")).expect("prompt");
    assert_eq!(value.as_deref(), Some("cuda"));
}

#[test]
fn test_prompt_model_file_rejects_missing_then_accepts_existing() {
    let temp_dir = TempDirGuard::new("kapsl-prompt-model-file").expect("temp dir");
    let context_dir = temp_dir.path();
    fs::write(context_dir.join("model.onnx"), b"dummy model").expect("model file");

    // First line points at a non-existent file (rejected), second is valid.
    let mut reader = Cursor::new(b"missing.onnx\nmodel.onnx\n".to_vec());
    let resolved =
        prompt_model_file_with_default(&mut reader, context_dir, "model.onnx").expect("prompt");
    assert_eq!(
        resolved.canonicalize().expect("resolved"),
        context_dir
            .join("model.onnx")
            .canonicalize()
            .expect("model")
    );
}

#[test]
fn test_prompt_aborts_on_end_of_input_instead_of_looping() {
    // Regression: an exhausted reader (or Ctrl-D) must abort, never spin a
    // retry loop forever. Only invalid input is provided, then EOF.
    let temp_dir = TempDirGuard::new("kapsl-prompt-eof").expect("temp dir");
    let context_dir = temp_dir.path();
    let mut reader = Cursor::new(b"does-not-exist.onnx\n".to_vec());
    let result = prompt_model_file_with_default(&mut reader, context_dir, "also-missing.onnx");
    assert!(
        result.is_err(),
        "EOF after invalid input must error, not hang"
    );

    let mut reader = Cursor::new(Vec::new());
    assert!(prompt_with_default(&mut reader, "Anything", "default").is_err());
}

#[test]
fn test_resolve_axis_triple_fills_defaults_and_legacy_framework() {
    let ax =
        |f: Option<&'static str>, m: Option<&'static str>, t: Option<&'static str>| AxisOverrides {
            format: f,
            model_type: m,
            task: t,
        };

    // --format gguf alone -> causal-lm / generate, legacy framework "gguf".
    assert_eq!(
        resolve_axis_triple("onnx", ax(Some("gguf"), None, None)),
        (
            "gguf".into(),
            "causal-lm".into(),
            "generate".into(),
            "gguf".into()
        )
    );

    // --task embed alone -> model type inferred as embedding (coherent pair).
    assert_eq!(
        resolve_axis_triple("onnx", ax(None, None, Some("embed"))),
        (
            "onnx".into(),
            "embedding".into(),
            "embed".into(),
            "onnx".into()
        )
    );

    // onnx + generate -> legacy framework "llm".
    assert_eq!(
        resolve_axis_triple(
            "onnx",
            ax(Some("onnx"), Some("causal-lm"), Some("generate"))
        )
        .3,
        "llm"
    );

    // No overrides -> falls back to the inferred default format.
    assert_eq!(
        resolve_axis_triple("gguf", AxisOverrides::default()).0,
        "gguf"
    );
}
