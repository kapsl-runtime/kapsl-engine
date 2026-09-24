use super::*;
use ed25519_dalek::{Signer, SigningKey};
use flate2::write::GzEncoder;
use flate2::Compression;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Barrier};
use tar::Builder;

fn sign_index(key: &SigningKey, bytes: &[u8]) -> String {
    let mut message = b"kapsl-backend-index-v1\0".to_vec();
    message.extend_from_slice(bytes);
    format!("ed25519:{}", BASE64.encode(key.sign(&message).to_bytes()))
}

fn sign_artifact(key: &SigningKey, digest: &str) -> String {
    let message = format!("kapsl-backend-artifact-v1\0sha256:{digest}");
    format!(
        "ed25519:{}",
        BASE64.encode(key.sign(message.as_bytes()).to_bytes())
    )
}

fn build_pack(path: &Path) {
    let file = File::create(path).unwrap();
    let gzip = GzEncoder::new(file, Compression::fast());
    let mut builder = Builder::new(gzip);
    let payload = BackendPayloadManifest {
        schema_version: 1,
        backend: "vllm".to_string(),
        profile: MANAGED_VLLM_PACK_PROFILE.to_string(),
        pack_version: "0.26.1".to_string(),
        runtime_abi: 1,
        adapter_abi: None,
        platform: current_platform(),
        execution_mode: BackendExecutionMode::External,
        kv_mode: None,
        entrypoint: "bin/python".to_string(),
    };
    let manifest = serde_json::to_vec_pretty(&payload).unwrap();
    let mut header = tar::Header::new_gnu();
    header.set_size(manifest.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, PAYLOAD_MANIFEST_NAME, manifest.as_slice())
        .unwrap();
    let executable = b"#!/bin/sh\nexit 0\n";
    let mut header = tar::Header::new_gnu();
    header.set_size(executable.len() as u64);
    header.set_mode(0o755);
    header.set_cksum();
    builder
        .append_data(&mut header, "bin/python", executable.as_slice())
        .unwrap();
    builder.finish().unwrap();
}

fn fixture() -> (
    tempfile::TempDir,
    BackendManager,
    BackendPackManifest,
    SigningKey,
) {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("pack.tar.gz");
    build_pack(&artifact);
    let signing = SigningKey::from_bytes(&[7_u8; 32]);
    let digest = sha256_file(&artifact).unwrap();
    let executable_digest = format!("{:x}", Sha256::digest(b"#!/bin/sh\nexit 0\n"));
    let pack = BackendPackManifest {
        schema_version: 1,
        backend: "vllm".to_string(),
        profile: MANAGED_VLLM_PACK_PROFILE.to_string(),
        pack_version: "0.26.1".to_string(),
        runtime_abi: 1,
        adapter_abi: None,
        compatible_kapsl: ">=0.2.3, <0.3.0".to_string(),
        platform: current_platform(),
        architecture: std::env::consts::ARCH.to_string(),
        accelerator_profile: "cuda".to_string(),
        accelerator_requirements: Default::default(),
        minimum_cuda: Some("12.0".to_string()),
        minimum_driver: None,
        execution_mode: BackendExecutionMode::External,
        kv_mode: None,
        formats: Vec::new(),
        model_types: Vec::new(),
        tasks: Vec::new(),
        capabilities: Default::default(),
        memory_behavior: Default::default(),
        entrypoint: "bin/python".to_string(),
        artifact: format!("file://{}", artifact.display()),
        download_bytes: fs::metadata(&artifact).unwrap().len(),
        installed_bytes: 4096,
        sha256: digest.clone(),
        signature: sign_artifact(&signing, &digest),
        memory: BackendMemoryManifest::default(),
        installer: BackendInstaller::Extract,
        files_blake3: None,
        files: BTreeMap::from([("bin/python".to_string(), executable_digest)]),
        licenses: Vec::new(),
        priority: 1,
    };
    let index = BackendIndex {
        schema_version: 1,
        runtime_version: "0.2.3".to_string(),
        generated_at: "2026-08-25T00:00:00Z".to_string(),
        packs: vec![pack.clone()],
    };
    let index_path = root.path().join("backend-index.json");
    let bytes = serde_json::to_vec_pretty(&index).unwrap();
    fs::write(&index_path, &bytes).unwrap();
    fs::write(
        format!("{}.sig", index_path.display()),
        sign_index(&signing, &bytes),
    )
    .unwrap();
    let manager = BackendManager::for_test(
        root.path().join("cache"),
        "0.2.3",
        index_path,
        signing.verifying_key(),
        false,
    );
    (root, manager, pack, signing)
}

fn cuda_target() -> BackendTarget {
    BackendTarget {
        platform: current_platform(),
        architecture: std::env::consts::ARCH.to_string(),
        accelerator: BackendAccelerator::Cuda,
        cuda_version: Some("13.0".to_string()),
        driver_version: Some("580.1".to_string()),
    }
}

fn cpu_target() -> BackendTarget {
    BackendTarget {
        platform: current_platform(),
        architecture: std::env::consts::ARCH.to_string(),
        accelerator: BackendAccelerator::Cpu,
        cuda_version: None,
        driver_version: None,
    }
}

fn tensorrt_target() -> BackendTarget {
    BackendTarget {
        accelerator: BackendAccelerator::TensorRt,
        ..cuda_target()
    }
}

fn standard_native_pack(
    mut pack: BackendPackManifest,
    backend: &str,
    profile: &str,
    accelerator: &str,
) -> BackendPackManifest {
    let uses_device = accelerator != "cpu";
    pack.backend = backend.to_string();
    pack.profile = profile.to_string();
    pack.adapter_abi = Some(STANDARD_NATIVE_ADAPTER_ABI.to_string());
    pack.accelerator_profile = accelerator.to_string();
    pack.accelerator_requirements = BackendAcceleratorRequirements {
        kind: Some(accelerator.to_string()),
        execution_providers: vec![accelerator.to_string()],
        implicit_cpu_fallback: Some(false),
    };
    pack.minimum_cuda = uses_device.then(|| "12.0".to_string());
    pack.execution_mode = BackendExecutionMode::Native;
    pack.kv_mode = None;
    pack.formats = vec!["onnx".to_string()];
    pack.model_types = Vec::new();
    pack.tasks = vec!["forward".to_string(), "generate".to_string()];
    pack.capabilities = BackendPackCapabilities {
        batching: true,
        streaming: true,
        cancellation: true,
        memory_reporting: true,
        governed_device_allocator: uses_device,
        scoped_device_allocator: uses_device,
        kv_participation: false,
        concurrent_inference: true,
    };
    pack.memory_behavior = BackendMemoryBehavior {
        allocation_scope: uses_device.then(|| "kapsl-scoped-device-allocator-v1".to_string()),
        device_allocation: Some(if uses_device {
            "host-governed-scoped".to_string()
        } else {
            "none".to_string()
        }),
        planned_reporting: true,
        live_reporting: true,
        request_reporting: true,
        synchronize_before_free: uses_device,
    };
    pack
}

fn add_signed_blake3(root: &Path, pack: &mut BackendPackManifest, signing: &SigningKey) {
    pack.files_blake3 = Some(BTreeMap::from([(
        pack.entrypoint.clone(),
        blake3::hash(b"#!/bin/sh\nexit 0\n").to_hex().to_string(),
    )]));
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: "0.2.3".into(),
        generated_at: "2026-09-19T00:00:00Z".into(),
        packs: vec![pack.clone()],
    };
    let bytes = serde_json::to_vec(&index).unwrap();
    fs::write(root.join("backend-index.json"), &bytes).unwrap();
    fs::write(
        root.join("backend-index.json.sig"),
        sign_index(signing, &bytes),
    )
    .unwrap();
}

#[test]
fn legacy_manifest_omits_blake3_and_round_trips_unchanged() {
    let (_root, manager, pack, _) = fixture();
    let bytes = serde_json::to_vec(&pack).unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("files_blake3"));
    let restored: BackendPackManifest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(restored, pack);
    let path = manager.ensure_pack(&restored).unwrap();
    fs::write(path.join(&restored.entrypoint), b"tampered").unwrap();
    assert!(!manager.installed_pack_is_valid(&restored).unwrap());
}

#[test]
fn blake3_requires_the_complete_legacy_file_set_and_valid_digests() {
    let (root, manager, mut pack, signing) = fixture();
    add_signed_blake3(root.path(), &mut pack, &signing);
    manager.validate_pack_manifest(&pack, None).unwrap();
    for invalid in [
        BTreeMap::new(),
        BTreeMap::from([("extra".into(), "00".repeat(32))]),
        BTreeMap::from([(pack.entrypoint.clone(), "z".repeat(64))]),
        BTreeMap::from([(pack.entrypoint.clone(), "00".repeat(31))]),
        BTreeMap::from([(pack.entrypoint.clone(), "00".repeat(33))]),
    ] {
        let mut invalid_pack = pack.clone();
        invalid_pack.files_blake3 = Some(invalid);
        assert!(manager.validate_pack_manifest(&invalid_pack, None).is_err());
    }
    let mut invalid_pack = pack.clone();
    invalid_pack
        .files_blake3
        .as_mut()
        .unwrap()
        .insert("../escape".into(), "00".repeat(32));
    invalid_pack
        .files
        .insert("../escape".into(), "00".repeat(32));
    assert!(manager.validate_pack_manifest(&invalid_pack, None).is_err());
}

#[test]
fn installed_blake3_is_bound_to_the_signed_index_not_the_receipt() {
    let (root, manager, mut pack, signing) = fixture();
    add_signed_blake3(root.path(), &mut pack, &signing);
    let plan = manager.plan_vllm(&cuda_target()).unwrap();
    assert_eq!(plan.manifest, pack);
    let installed = manager.ensure_pack(&plan.manifest).unwrap();
    assert!(manager.list().unwrap()[0].valid);
    assert!(manager.installed_pack_is_valid(&pack).unwrap());
    fs::write(installed.join(&pack.entrypoint), b"#!/bin/sh\nexit 1\n").unwrap();
    assert!(!manager.installed_pack_is_valid(&pack).unwrap());
    let receipt = installed.join(INSTALL_RECORD_NAME);
    let mut record: InstalledPackRecord = read_json_bounded(&receipt, MAX_MANIFEST_BYTES).unwrap();
    record.manifest.files_blake3.as_mut().unwrap().insert(
        pack.entrypoint.clone(),
        blake3::hash(b"#!/bin/sh\nexit 1\n").to_hex().to_string(),
    );
    fs::write(receipt, serde_json::to_vec(&record).unwrap()).unwrap();
    assert!(!manager.list().unwrap()[0].valid);
    assert!(!manager.installed_pack_is_valid(&pack).unwrap());
}

#[test]
fn changing_blake3_index_bytes_invalidates_the_signature() {
    let (root, manager, mut pack, signing) = fixture();
    add_signed_blake3(root.path(), &mut pack, &signing);
    let path = root.path().join("backend-index.json");
    let mut index: BackendIndex = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    index.packs[0].files_blake3 = None;
    let bytes = serde_json::to_vec(&index).unwrap();
    let signature = fs::read_to_string(root.path().join("backend-index.json.sig")).unwrap();
    assert!(manager
        .verify_index_bytes(&bytes, &signature)
        .unwrap_err()
        .to_string()
        .contains("signature"));
}

#[test]
fn installation_checks_both_signed_digest_maps() {
    for corrupt_blake3 in [true, false] {
        let (root, manager, mut pack, signing) = fixture();
        add_signed_blake3(root.path(), &mut pack, &signing);
        let files = if corrupt_blake3 {
            pack.files_blake3.as_mut().unwrap()
        } else {
            &mut pack.files
        };
        files.insert(pack.entrypoint.clone(), "00".repeat(32));
        assert!(manager.ensure_pack(&pack).is_err());
        assert!(!manager.installed_path(&pack).unwrap().exists());
    }
}

#[cfg(unix)]
#[test]
fn blake3_does_not_allow_an_escaping_symlink_with_matching_bytes() {
    use std::os::unix::fs::symlink;
    let (root, manager, mut pack, signing) = fixture();
    add_signed_blake3(root.path(), &mut pack, &signing);
    let installed = manager.ensure_pack(&pack).unwrap();
    let external = root.path().join("external-python");
    fs::copy(installed.join(&pack.entrypoint), &external).unwrap();
    fs::remove_file(installed.join(&pack.entrypoint)).unwrap();
    symlink(external, installed.join(&pack.entrypoint)).unwrap();
    assert!(!manager.installed_pack_is_valid(&pack).unwrap());
}

#[test]
fn signed_pack_installs_atomically_and_is_reused() {
    let (_root, manager, pack, _) = fixture();
    let selected = manager.plan_vllm(&cuda_target()).unwrap();
    assert!(!selected.installed);
    let installed = manager.ensure_pack(&pack).unwrap();
    assert!(installed.join("bin/python").is_file());
    assert!(manager.installed_pack_is_valid(&pack).unwrap());
    assert_eq!(manager.ensure_pack(&pack).unwrap(), installed);
    let listed = manager.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert!(listed[0].valid);
}

#[test]
fn onnx_plan_requires_an_exact_native_profile() {
    let (_root, manager, pack, signing) = fixture();
    let pack = standard_native_pack(pack, "onnx", ONNX_CPU_PACK_PROFILE, "cpu");
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-08-25T00:00:00Z".to_string(),
        packs: vec![pack],
    };
    let bytes = serde_json::to_vec_pretty(&index).unwrap();
    let index_path = manager.config.index_path.as_ref().unwrap();
    fs::write(index_path, &bytes).unwrap();
    fs::write(
        format!("{}.sig", index_path.display()),
        sign_index(&signing, &bytes),
    )
    .unwrap();

    let plan = manager
        .plan_onnx(OnnxBackendPackProfile::Cpu, &cpu_target())
        .unwrap();
    assert_eq!(plan.selected_backend, "onnx");
    assert_eq!(plan.profile, ONNX_CPU_PACK_PROFILE);
    assert_eq!(plan.execution_mode, "native");
    assert_eq!(
        plan.manifest.adapter_abi.as_deref(),
        Some(STANDARD_NATIVE_ADAPTER_ABI)
    );
    assert!(manager
        .plan_onnx(OnnxBackendPackProfile::Cuda12, &cpu_target())
        .unwrap_err()
        .to_string()
        .contains("requires a cuda target"));
}

#[test]
fn capability_resolver_filters_model_contract_and_records_reason() {
    let (_root, manager, pack, _) = fixture();
    let mut compatible = standard_native_pack(pack.clone(), "fake-a", "cpu", "cpu");
    compatible.priority = 20;
    let mut wrong_task = standard_native_pack(pack, "fake-b", "cpu", "cpu");
    wrong_task.priority = 100;
    wrong_task.tasks = vec!["classify".to_string()];
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-09-03T00:00:00Z".to_string(),
        packs: vec![wrong_task, compatible],
    };
    let requirements = BackendPackRequirements {
        format: Some("onnx".to_string()),
        model_type: Some("causal-lm".to_string()),
        task: Some("generate".to_string()),
        execution_mode: Some(BackendExecutionMode::Native),
        capabilities: BackendPackCapabilities {
            batching: true,
            streaming: true,
            cancellation: true,
            memory_reporting: true,
            ..BackendPackCapabilities::default()
        },
        ..BackendPackRequirements::default()
    };

    let selected = manager
        .select_compatible_pack(&index, &requirements, &cpu_target())
        .unwrap();
    assert_eq!(selected.manifest.backend, "fake-a");
    assert!(selected
        .reason
        .contains("capability-based automatic selection"));
    assert!(selected.reason.contains("causal-lm"));
    assert!(selected.reason.contains("generate"));
}

#[test]
fn capability_resolver_requires_the_selected_execution_provider() {
    let (_root, manager, pack, _) = fixture();
    let mut cuda = standard_native_pack(pack.clone(), "fake-cuda", "cuda12", "cuda");
    cuda.priority = 100;
    let tensorrt = standard_native_pack(pack, "fake-tensorrt", "tensorrt10", "tensorrt");
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-09-03T00:00:00Z".to_string(),
        packs: vec![cuda, tensorrt],
    };
    let requirements = BackendPackRequirements {
        format: Some("onnx".to_string()),
        task: Some("forward".to_string()),
        execution_provider: Some("tensorrt".to_string()),
        ..BackendPackRequirements::default()
    };

    let selected = manager
        .select_compatible_pack(&index, &requirements, &tensorrt_target())
        .unwrap();
    assert_eq!(selected.manifest.backend, "fake-tensorrt");
    assert!(selected.reason.contains("provider `tensorrt`"));
}

#[test]
fn capability_resolver_orders_semantic_versions_numerically() {
    let (_root, manager, pack, _) = fixture();
    let mut older = standard_native_pack(pack.clone(), "fake", "cpu", "cpu");
    older.pack_version = "0.9.0".to_string();
    let mut newer = standard_native_pack(pack, "fake", "cpu", "cpu");
    newer.pack_version = "0.10.0".to_string();
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-09-03T00:00:00Z".to_string(),
        packs: vec![older, newer],
    };
    let requirements = BackendPackRequirements {
        backend_pin: Some("fake".to_string()),
        format: Some("onnx".to_string()),
        task: Some("forward".to_string()),
        ..BackendPackRequirements::default()
    };

    let selected = manager
        .select_compatible_pack(&index, &requirements, &cpu_target())
        .unwrap();
    assert_eq!(selected.manifest.pack_version, "0.10.0");
}

#[test]
fn explicit_backend_pin_fails_without_substitution() {
    let (_root, manager, pack, _) = fixture();
    let available = standard_native_pack(pack, "available", "cpu", "cpu");
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-09-03T00:00:00Z".to_string(),
        packs: vec![available],
    };
    let requirements = BackendPackRequirements {
        backend_pin: Some("required".to_string()),
        format: Some("onnx".to_string()),
        task: Some("forward".to_string()),
        ..BackendPackRequirements::default()
    };

    let error = manager
        .select_compatible_pack(&index, &requirements, &cpu_target())
        .unwrap_err()
        .to_string();
    assert!(error.contains("explicit backend `required`"));
    assert!(error.contains("no compatible signed pack"));
    assert!(!error.contains("selected available"));
}

#[test]
fn equally_ranked_automatic_candidates_are_ambiguous() {
    let (_root, manager, pack, _) = fixture();
    let first = standard_native_pack(pack.clone(), "fake-a", "cpu", "cpu");
    let second = standard_native_pack(pack, "fake-b", "cpu", "cpu");
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-09-03T00:00:00Z".to_string(),
        packs: vec![second, first],
    };
    let mut requirements = BackendPackRequirements {
        format: Some("onnx".to_string()),
        task: Some("forward".to_string()),
        ..BackendPackRequirements::default()
    };

    let error = manager
        .select_compatible_pack(&index, &requirements, &cpu_target())
        .unwrap_err()
        .to_string();
    assert!(error.contains("ambiguous"));
    assert!(error.contains("fake-a/cpu"));
    assert!(error.contains("fake-b/cpu"));

    requirements.backend_pin = Some("fake-b".to_string());
    let selected = manager
        .select_compatible_pack(&index, &requirements, &cpu_target())
        .unwrap();
    assert_eq!(selected.manifest.backend, "fake-b");
}

#[test]
fn scoped_allocator_requirement_rejects_unscoped_accelerator_pack() {
    let (_root, manager, pack, _) = fixture();
    let mut unscoped = standard_native_pack(pack, "fake-gpu", "cuda12", "cuda");
    unscoped.capabilities.scoped_device_allocator = false;
    unscoped.memory_behavior.allocation_scope = None;
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-09-03T00:00:00Z".to_string(),
        packs: vec![unscoped],
    };
    let requirements = BackendPackRequirements {
        backend_pin: Some("fake-gpu".to_string()),
        format: Some("onnx".to_string()),
        task: Some("generate".to_string()),
        execution_mode: Some(BackendExecutionMode::Native),
        capabilities: BackendPackCapabilities {
            memory_reporting: true,
            governed_device_allocator: true,
            scoped_device_allocator: true,
            ..BackendPackCapabilities::default()
        },
        allocation_scope: Some("kapsl-scoped-device-allocator-v1".to_string()),
        ..BackendPackRequirements::default()
    };

    let error = manager
        .select_compatible_pack(&index, &requirements, &cuda_target())
        .unwrap_err()
        .to_string();
    assert!(error.contains("scoped_device_allocator"));
}

#[test]
fn governed_allocator_requirement_rejects_missing_device_synchronization() {
    let (_root, manager, pack, _) = fixture();
    let mut unsynchronized = standard_native_pack(pack, "fake-gpu", "cuda12", "cuda");
    unsynchronized.memory_behavior.synchronize_before_free = false;
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-09-03T00:00:00Z".to_string(),
        packs: vec![unsynchronized],
    };
    let requirements = BackendPackRequirements {
        backend_pin: Some("fake-gpu".to_string()),
        format: Some("onnx".to_string()),
        task: Some("generate".to_string()),
        execution_provider: Some("cuda".to_string()),
        capabilities: BackendPackCapabilities {
            governed_device_allocator: true,
            scoped_device_allocator: true,
            ..BackendPackCapabilities::default()
        },
        allocation_scope: Some("kapsl-scoped-device-allocator-v1".to_string()),
        synchronize_before_free: true,
        ..BackendPackRequirements::default()
    };

    let error = manager
        .select_compatible_pack(&index, &requirements, &cuda_target())
        .unwrap_err()
        .to_string();
    assert!(error.contains("synchronize the device"));
}

#[test]
fn adapter_abi_is_explicit_and_fail_closed() {
    let (_root, manager, mut pack, _) = fixture();
    pack.execution_mode = BackendExecutionMode::Native;
    pack.adapter_abi = Some("vendor-private-v1".to_string());
    let error = manager
        .validate_pack_manifest(&pack, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("unsupported native adapter ABI"));

    pack.adapter_abi = Some(STANDARD_NATIVE_ADAPTER_ABI.to_string());
    pack.execution_mode = BackendExecutionMode::External;
    let error = manager
        .validate_pack_manifest(&pack, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("only native backend packs"));
}

#[test]
fn payload_must_repeat_the_signed_adapter_abi() {
    let (_root, _manager, mut signed, _) = fixture();
    signed.execution_mode = BackendExecutionMode::Native;
    signed.adapter_abi = Some(STANDARD_NATIVE_ADAPTER_ABI.to_string());
    let mut payload = BackendPayloadManifest {
        schema_version: signed.schema_version,
        backend: signed.backend.clone(),
        profile: signed.profile.clone(),
        pack_version: signed.pack_version.clone(),
        runtime_abi: signed.runtime_abi,
        adapter_abi: None,
        platform: signed.platform.clone(),
        execution_mode: signed.execution_mode,
        kv_mode: signed.kv_mode.clone(),
        entrypoint: signed.entrypoint.clone(),
    };
    assert!(validate_payload_manifest(&signed, &payload).is_err());
    payload.adapter_abi = signed.adapter_abi.clone();
    assert!(validate_payload_manifest(&signed, &payload).is_ok());
}

#[test]
fn llama_cpp_plan_requires_an_exact_native_profile() {
    let (_root, manager, mut pack, signing) = fixture();
    pack.backend = "llama-cpp".to_string();
    pack.profile = LLAMA_CPP_CPU_PACK_PROFILE.to_string();
    pack.accelerator_profile = "cpu".to_string();
    pack.minimum_cuda = None;
    pack.execution_mode = BackendExecutionMode::Native;
    pack.kv_mode = Some("native".to_string());
    let index = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-08-25T00:00:00Z".to_string(),
        packs: vec![pack],
    };
    let bytes = serde_json::to_vec_pretty(&index).unwrap();
    let index_path = manager.config.index_path.as_ref().unwrap();
    fs::write(index_path, &bytes).unwrap();
    fs::write(
        format!("{}.sig", index_path.display()),
        sign_index(&signing, &bytes),
    )
    .unwrap();

    let plan = manager
        .plan_llama_cpp(LlamaCppBackendPackProfile::Cpu, &cpu_target())
        .unwrap();
    assert_eq!(plan.selected_backend, "llama-cpp");
    assert_eq!(plan.profile, LLAMA_CPP_CPU_PACK_PROFILE);
    assert_eq!(plan.execution_mode, "native");
    assert!(manager
        .plan_llama_cpp(LlamaCppBackendPackProfile::Cuda12, &cpu_target())
        .unwrap_err()
        .to_string()
        .contains("requires a cuda target"));
}

#[test]
fn llama_cpp_pack_requires_a_signed_kv_mode() {
    let (_root, manager, mut pack, _) = fixture();
    pack.backend = "llama-cpp".to_string();
    pack.profile = LLAMA_CPP_CPU_PACK_PROFILE.to_string();
    pack.accelerator_profile = "cpu".to_string();
    pack.minimum_cuda = None;
    pack.execution_mode = BackendExecutionMode::Native;

    let error = manager
        .validate_pack_manifest(&pack, Some(&cpu_target()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("must sign an explicit kv_mode"));
}

#[test]
fn offline_index_anchor_rejects_a_conflicting_signed_release_index() {
    let (_root, manager, pack, signing) = fixture();
    let original = BackendIndex {
        schema_version: BACKEND_INDEX_SCHEMA_VERSION,
        runtime_version: manager.runtime_version().to_string(),
        generated_at: "2026-08-25T00:00:00Z".to_string(),
        packs: vec![pack],
    };
    let original_bytes = serde_json::to_vec_pretty(&original).unwrap();
    let original_signature = sign_index(&signing, &original_bytes);
    manager
        .cache_verified_index_bytes(&original_bytes, &original_signature)
        .unwrap();

    let mut conflicting = original;
    conflicting.generated_at = "2026-08-25T00:00:01Z".to_string();
    let conflicting_bytes = serde_json::to_vec_pretty(&conflicting).unwrap();
    let conflicting_signature = sign_index(&signing, &conflicting_bytes);
    let error = manager
        .cache_verified_index_bytes(&conflicting_bytes, &conflicting_signature)
        .unwrap_err()
        .to_string();
    assert!(error.contains("different signed backend index"));
}

#[test]
fn corrupted_archive_is_rejected_without_partial_install() {
    let (root, manager, pack, _) = fixture();
    let artifact = PathBuf::from(pack.artifact.strip_prefix("file://").unwrap());
    fs::write(&artifact, b"corrupt").unwrap();
    let error = manager.ensure_pack(&pack).unwrap_err().to_string();
    assert!(error.contains("size mismatch") || error.contains("checksum mismatch"));
    assert!(!manager.installed_path(&pack).unwrap().exists());
    assert!(root.path().join("cache").exists());
}

#[test]
fn invalid_index_signature_fails_closed() {
    let (_root, manager, _pack, _) = fixture();
    let signature_path = format!(
        "{}.sig",
        manager.config.index_path.as_ref().unwrap().display()
    );
    fs::write(
        signature_path,
        format!("ed25519:{}", BASE64.encode([0_u8; 64])),
    )
    .unwrap();
    assert!(manager
        .load_index()
        .unwrap_err()
        .to_string()
        .contains("signature verification failed"));
}

#[test]
fn abi_and_platform_mismatches_are_rejected() {
    let (_root, manager, mut pack, _) = fixture();
    pack.runtime_abi = 2;
    assert!(manager
        .validate_pack_manifest(&pack, Some(&cuda_target()))
        .unwrap_err()
        .to_string()
        .contains("ABI"));
    pack.runtime_abi = 1;
    pack.platform = "somewhere-else".to_string();
    assert!(manager
        .validate_pack_manifest(&pack, Some(&cuda_target()))
        .unwrap_err()
        .to_string()
        .contains("platform"));
}

#[test]
fn offline_missing_pack_has_actionable_error() {
    let (root, _manager, _pack, signing) = fixture();
    let index_path = root.path().join("backend-index.json");
    let manager = BackendManager::for_test(
        root.path().join("offline-cache"),
        "0.2.3",
        index_path,
        signing.verifying_key(),
        true,
    );
    assert!(manager
        .plan_vllm(&cuda_target())
        .unwrap_err()
        .to_string()
        .contains("offline"));
}

#[test]
fn concurrent_ensure_produces_one_complete_install() {
    let (_root, manager, pack, _) = fixture();
    let manager = Arc::new(manager);
    let pack = Arc::new(pack);
    let barrier = Arc::new(Barrier::new(3));
    let handles = (0..2)
        .map(|_| {
            let manager = manager.clone();
            let pack = pack.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                manager.ensure_pack(&pack)
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let paths = handles
        .into_iter()
        .map(|handle| handle.join().unwrap().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(paths[0], paths[1]);
    assert!(manager.installed_pack_is_valid(&pack).unwrap());
}

#[test]
fn interrupted_stage_is_removed_before_retry() {
    let (_root, manager, pack, _signing) = fixture();
    let stale = manager
        .pack_staging_root(&pack)
        .unwrap()
        .join("install-interrupted");
    fs::create_dir_all(&stale).unwrap();
    fs::write(stale.join("partial-download"), b"partial").unwrap();

    let installed = manager.ensure_pack(&pack).unwrap();
    assert!(installed.join("bin/python").is_file());
    assert!(!stale.exists());
}

#[test]
fn corrupted_cached_file_is_replaced_from_signed_pack() {
    let (_root, manager, pack, _signing) = fixture();
    let installed = manager.ensure_pack(&pack).unwrap();
    fs::write(installed.join("bin/python"), b"corrupt").unwrap();

    let repaired = manager.ensure_pack(&pack).unwrap();
    assert_eq!(
        fs::read(repaired.join("bin/python")).unwrap(),
        b"#!/bin/sh\nexit 0\n"
    );
}

#[test]
fn escaping_symlink_is_rejected() {
    assert!(!lexical_target_stays_inside(
        Path::new("/cache"),
        Path::new("/cache/bin/python"),
        Path::new("../../outside")
    ));
    assert!(lexical_target_stays_inside(
        Path::new("/cache"),
        Path::new("/cache/bin/python"),
        Path::new("../python3.12")
    ));
}

#[test]
fn numeric_versions_compare_without_lexical_errors() {
    assert!(version_at_least("13.0", "12.6"));
    assert!(version_at_least("580.10.2", "580.9"));
    assert!(!version_at_least("12.1", "12.10"));
}
