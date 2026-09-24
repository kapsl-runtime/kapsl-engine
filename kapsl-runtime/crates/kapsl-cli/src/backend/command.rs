//! `kapsl backend ensure|list|prune`: the CLI front end for the signed
//! backend-pack manager.

use crate::backend::{
    current_platform, embedded_onnx_rollback_reason, expand_run_bundles,
    generic_native_backend_packs_enabled, inspect_serving_manifest, lazy_llama_cpp_packs_enabled,
    lazy_onnx_packs_enabled, llama_cpp_pack_profile_for_manifest, onnx_backend_pack_requirements,
    onnx_pack_profile_for_manifest, preliminary_llama_cpp_memory_admission,
    preliminary_memory_admission, resolve_serving_backend, validate_model_contract,
    validate_runtime_serving_backend, BackendManager, BackendSubcommand, BackendTarget,
    LlamaCppBackendPackProfile, MemoryAdmissionStatus, OnnxBackendPackProfile,
    ResolvedServingBackend, GENERIC_NATIVE_PACKS_ENV, MANAGED_VLLM_PACK_PROFILE,
};
use kapsl_hal::device::DeviceInfo;
use std::collections::HashSet;

pub(crate) fn execute_backend_command(
    args: crate::app::BackendCommandArgs,
) -> Result<(), crate::DynError> {
    match args.command {
        crate::app::BackendSubcommand::Ensure(args) => {
            let device_info = DeviceInfo::probe();
            let target = BackendTarget::current(&device_info);
            let model_paths = expand_run_bundles(&args.model, &device_info)?;
            let manager = BackendManager::from_env(args.offline)?;
            let mut ensured = HashSet::<(String, String)>::new();
            for model in model_paths {
                let absolute = model
                    .canonicalize()
                    .map_err(|error| format!("invalid model path {}: {error}", model.display()))?;
                let manifest = inspect_serving_manifest(&absolute)?;
                validate_model_contract(&manifest)?;
                let decision = resolve_serving_backend(&manifest, device_info.has_cuda)?;
                validate_runtime_serving_backend(&manifest, decision)?;
                let memory =
                    preliminary_memory_admission(&absolute, &manifest, decision, &device_info)?;
                if memory.status == MemoryAdmissionStatus::Rejected {
                    return Err(format!(
                        "preliminary memory admission rejected `{}` before backend download: {}",
                        manifest.project_name, memory.reason
                    )
                    .into());
                }
                let uses_onnx = kapsl_core::EngineKind::resolve(&manifest).uses_onnx_session();
                let signed_onnx_route = generic_native_backend_packs_enabled()?;
                let onnx_profile = if uses_onnx && signed_onnx_route {
                    if !lazy_onnx_packs_enabled() {
                        return Err(format!(
                            "model `{}` requires a signed native backend pack, but ONNX pack installation is disabled or unsupported on {}; set {}=0 only for an explicit embedded ORT rollback",
                            manifest.project_name,
                            current_platform(),
                            GENERIC_NATIVE_PACKS_ENV
                        )
                        .into());
                    }
                    onnx_pack_profile_for_manifest(&manifest, &device_info)?
                } else {
                    None
                };
                let llama_profile = if lazy_llama_cpp_packs_enabled() {
                    llama_cpp_pack_profile_for_manifest(&manifest, &device_info)
                } else {
                    None
                };
                if decision.selected == ResolvedServingBackend::Vllm
                    && ensured.insert(("vllm".to_string(), MANAGED_VLLM_PACK_PROFILE.to_string()))
                {
                    let installed = manager.ensure_vllm(&target)?;
                    println!(
                        "Ensured backend vllm/{} at {}",
                        MANAGED_VLLM_PACK_PROFILE,
                        installed.display()
                    );
                } else if let Some(profile) = onnx_profile {
                    let mut onnx_target = target.clone();
                    onnx_target.accelerator = profile.accelerator();
                    if profile == OnnxBackendPackProfile::Cpu {
                        onnx_target.cuda_version = None;
                        onnx_target.driver_version = None;
                    }
                    let requirements = onnx_backend_pack_requirements(&manifest, profile)?;
                    let plan = manager.plan_compatible_backend(&requirements, &onnx_target)?;
                    let identity = (plan.selected_backend.clone(), plan.profile.clone());
                    if ensured.insert(identity) {
                        let installed = manager.ensure_pack(&plan.manifest)?;
                        println!(
                            "Ensured backend {}/{} at {} ({})",
                            plan.selected_backend,
                            plan.profile,
                            installed.display(),
                            plan.selection_reason
                        );
                    }
                } else if let Some(profile) = llama_profile {
                    let identity = ("llama-cpp".to_string(), profile.profile().to_string());
                    if ensured.insert(identity) {
                        let mut llama_target = target.clone();
                        llama_target.accelerator = profile.accelerator();
                        if profile == LlamaCppBackendPackProfile::Cpu {
                            llama_target.cuda_version = None;
                            llama_target.driver_version = None;
                        }
                        let plan = manager.plan_llama_cpp(profile, &llama_target)?;
                        let admission = preliminary_llama_cpp_memory_admission(
                            profile,
                            &absolute,
                            &manifest,
                            &device_info,
                            &plan.manifest,
                            None,
                        )?;
                        if admission.status == MemoryAdmissionStatus::Rejected {
                            return Err(format!(
                                "preliminary memory admission rejected `{}` before backend download: {}",
                                manifest.project_name, admission.reason
                            )
                            .into());
                        }
                        let installed = manager.ensure_pack(&plan.manifest)?;
                        println!(
                            "Ensured backend llama-cpp/{} at {}",
                            profile.profile(),
                            installed.display()
                        );
                    }
                } else if uses_onnx && !signed_onnx_route {
                    let reason = embedded_onnx_rollback_reason(&manifest)?;
                    println!(
                        "{} uses the embedded ORT rollback ({})",
                        manifest.project_name, reason
                    );
                } else if decision.selected != ResolvedServingBackend::Vllm {
                    println!(
                        "{} uses the in-process {} backend; no lazy pack is required by this runtime build",
                        manifest.project_name,
                        decision.selected.as_str()
                    );
                }
            }
            Ok(())
        }
        BackendSubcommand::List(args) => {
            let manager = BackendManager::from_env(true)?;
            let entries = manager.list()?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else if entries.is_empty() {
                println!("No backend packs are installed.");
            } else {
                println!("RUNTIME\tBACKEND\tPROFILE\tVERSION\tMODE\tSIZE\tSTATUS\tPATH");
                for entry in entries {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        entry.runtime_version,
                        entry.backend,
                        entry.profile,
                        entry.pack_version,
                        entry.execution_mode,
                        entry.installed_bytes,
                        if entry.valid { "valid" } else { "invalid" },
                        entry.path.display()
                    );
                }
            }
            Ok(())
        }
        BackendSubcommand::Prune(args) => {
            let manager = BackendManager::from_env(true)?;
            let (bytes, entries) = manager.prune(args.old_versions)?;
            println!("Pruned {entries} backend cache item(s), reclaiming {bytes} bytes.");
            Ok(())
        }
    }
}
