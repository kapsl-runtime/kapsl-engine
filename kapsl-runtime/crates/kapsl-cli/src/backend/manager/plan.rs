//! Selection of a signed pack for a model and target.

use super::*;

pub(super) fn signed_contract_contains(values: &[String], required: &str) -> bool {
    let required = required.trim();
    !required.is_empty()
        && values
            .iter()
            .any(|value| value.trim().eq_ignore_ascii_case(required))
}

pub(super) fn compare_pack_versions(left: &str, right: &str) -> std::cmp::Ordering {
    match (semver::Version::parse(left), semver::Version::parse(right)) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        (Ok(_), Err(_)) => std::cmp::Ordering::Greater,
        (Err(_), Ok(_)) => std::cmp::Ordering::Less,
        (Err(_), Err(_)) => {
            let left_numeric = numeric_version_components(left);
            let right_numeric = numeric_version_components(right);
            left_numeric
                .cmp(&right_numeric)
                .then_with(|| left.cmp(right))
        }
    }
}

impl BackendManager {
    /// Resolve one model contract against the verified signed index without
    /// downloading or loading adapter code.
    pub(crate) fn plan_compatible_backend(
        &self,
        requirements: &BackendPackRequirements,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackPlan> {
        let index = self.load_index()?;
        let selection = self.select_compatible_pack(&index, requirements, target)?;
        let pack = selection.manifest;
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: selection.reason,
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn plan_vllm(&self, target: &BackendTarget) -> ManagerResult<BackendPackPlan> {
        let index = self.load_index()?;
        let pack = self.select_pack(&index, "vllm", Some(MANAGED_VLLM_PACK_PROFILE), target)?;
        if pack.execution_mode != BackendExecutionMode::External {
            return Err(BackendManagerError::new(
                "the managed vLLM backend pack must use external execution mode",
            ));
        }
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: format!(
                "explicit vLLM profile `{}` matched the signed backend index",
                pack.profile
            ),
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn plan_onnx(
        &self,
        profile: OnnxBackendPackProfile,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackPlan> {
        if target.accelerator != profile.accelerator() {
            return Err(BackendManagerError::new(format!(
                "ONNX profile `{}` requires a {} target, not {}",
                profile.profile(),
                profile.accelerator().as_str(),
                target.accelerator.as_str()
            )));
        }
        let index = self.load_index()?;
        let pack = self.select_pack(&index, "onnx", Some(profile.profile()), target)?;
        if pack.execution_mode != BackendExecutionMode::Native {
            return Err(BackendManagerError::new(format!(
                "the ONNX backend pack {}/{} must use native execution mode",
                pack.backend, pack.profile
            )));
        }
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: format!(
                "explicit ONNX profile `{}` matched the signed backend index",
                pack.profile
            ),
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn plan_llama_cpp(
        &self,
        profile: LlamaCppBackendPackProfile,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackPlan> {
        if target.accelerator != profile.accelerator() {
            return Err(BackendManagerError::new(format!(
                "llama.cpp profile `{}` requires a {} target, not {}",
                profile.profile(),
                profile.accelerator().as_str(),
                target.accelerator.as_str()
            )));
        }
        let index = self.load_index()?;
        let pack = self.select_pack(&index, "llama-cpp", Some(profile.profile()), target)?;
        if pack.execution_mode != BackendExecutionMode::Native {
            return Err(BackendManagerError::new(format!(
                "the llama.cpp backend pack {}/{} must use native execution mode",
                pack.backend, pack.profile
            )));
        }
        let installed = self.installed_pack_is_valid(&pack).unwrap_or(false);
        Ok(BackendPackPlan {
            selected_backend: pack.backend.clone(),
            profile: pack.profile.clone(),
            selection_reason: format!(
                "explicit llama.cpp profile `{}` matched the signed backend index",
                pack.profile
            ),
            installed,
            download_required: !installed,
            download_bytes: if installed { 0 } else { pack.download_bytes },
            execution_mode: pack.execution_mode.as_str().to_string(),
            manifest: pack,
        })
    }

    pub(crate) fn select_compatible_pack(
        &self,
        index: &BackendIndex,
        requirements: &BackendPackRequirements,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackSelection> {
        let backend_pin = requirements
            .backend_pin
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let preferred_profile = requirements
            .preferred_profile
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let mut rejected = Vec::new();
        let mut candidates = Vec::new();

        for pack in &index.packs {
            if backend_pin.is_some_and(|backend| !pack.backend.eq_ignore_ascii_case(backend)) {
                continue;
            }
            if preferred_profile.is_some_and(|profile| pack.profile != profile) {
                continue;
            }

            let identity = format!("{}/{}", pack.backend, pack.profile);
            if let Err(error) = self.validate_pack_manifest(pack, Some(target)) {
                rejected.push(format!("{identity}: {error}"));
                continue;
            }
            if requirements
                .execution_mode
                .is_some_and(|mode| pack.execution_mode != mode)
            {
                rejected.push(format!(
                    "{identity}: execution mode {} does not match required {}",
                    pack.execution_mode.as_str(),
                    requirements
                        .execution_mode
                        .expect("checked execution mode")
                        .as_str()
                ));
                continue;
            }
            if requirements.adapter_abi.is_some()
                && pack.adapter_abi.as_deref() != requirements.adapter_abi.as_deref()
            {
                rejected.push(format!(
                    "{identity}: adapter ABI {:?} does not match required {:?}",
                    pack.adapter_abi, requirements.adapter_abi
                ));
                continue;
            }
            if let Some(format) = requirements.format.as_deref() {
                if !signed_contract_contains(&pack.formats, format) {
                    rejected.push(format!(
                        "{identity}: format `{format}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(model_type) = requirements.model_type.as_deref() {
                if !pack.model_types.is_empty()
                    && !signed_contract_contains(&pack.model_types, model_type)
                {
                    rejected.push(format!(
                        "{identity}: model type `{model_type}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(task) = requirements.task.as_deref() {
                if !signed_contract_contains(&pack.tasks, task) {
                    rejected.push(format!(
                        "{identity}: task `{task}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            if let Some(provider) = requirements.execution_provider.as_deref() {
                if !signed_contract_contains(
                    &pack.accelerator_requirements.execution_providers,
                    provider,
                ) {
                    rejected.push(format!(
                        "{identity}: execution provider `{provider}` is not declared by the signed pack"
                    ));
                    continue;
                }
            }
            let missing = pack
                .capabilities
                .missing_required(&requirements.capabilities);
            if !missing.is_empty() {
                rejected.push(format!(
                    "{identity}: missing required capabilities {}",
                    missing.join(", ")
                ));
                continue;
            }
            if let Some(scope) = requirements.allocation_scope.as_deref() {
                if pack.memory_behavior.allocation_scope.as_deref() != Some(scope) {
                    rejected.push(format!(
                        "{identity}: allocation scope {:?} does not match required `{scope}`",
                        pack.memory_behavior.allocation_scope
                    ));
                    continue;
                }
            }
            if requirements.synchronize_before_free && !pack.memory_behavior.synchronize_before_free
            {
                rejected.push(format!(
                    "{identity}: signed memory behavior does not synchronize the device before freeing governed allocations"
                ));
                continue;
            }
            candidates.push(pack.clone());
        }

        candidates.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| compare_pack_versions(&right.pack_version, &left.pack_version))
                .then_with(|| left.backend.cmp(&right.backend))
                .then_with(|| left.profile.cmp(&right.profile))
        });

        let Some(selected) = candidates.first() else {
            let selection = backend_pin
                .map(|backend| format!("explicit backend `{backend}`"))
                .unwrap_or_else(|| "automatic backend selection".to_string());
            return Err(BackendManagerError::new(format!(
                "{selection} found no compatible signed pack for format `{}`, model type `{}`, task `{}`, provider `{}` on {} ({}){}",
                requirements.format.as_deref().unwrap_or("any"),
                requirements.model_type.as_deref().unwrap_or("any"),
                requirements.task.as_deref().unwrap_or("any"),
                requirements.execution_provider.as_deref().unwrap_or("any"),
                target.platform,
                target.accelerator.as_str(),
                if rejected.is_empty() {
                    String::new()
                } else {
                    format!("; rejected candidates: {}", rejected.join("; "))
                }
            )));
        };

        let tied = candidates
            .iter()
            .take_while(|candidate| {
                candidate.priority == selected.priority
                    && compare_pack_versions(&candidate.pack_version, &selected.pack_version)
                        == std::cmp::Ordering::Equal
            })
            .map(|candidate| format!("{}/{}", candidate.backend, candidate.profile))
            .collect::<Vec<_>>();
        if tied.len() > 1 {
            return Err(BackendManagerError::new(format!(
                "backend selection is ambiguous between equally ranked signed packs: {}; use an explicit backend pin or assign distinct signed priorities",
                tied.join(", ")
            )));
        }

        let selection_kind = backend_pin
            .map(|backend| format!("explicit backend pin `{backend}`"))
            .unwrap_or_else(|| "capability-based automatic selection".to_string());
        Ok(BackendPackSelection {
            manifest: selected.clone(),
            reason: format!(
                "{selection_kind} matched signed {}/{} version {} for format `{}`, model type `{}`, task `{}`, provider `{}`, and {} execution",
                selected.backend,
                selected.profile,
                selected.pack_version,
                requirements.format.as_deref().unwrap_or("any"),
                requirements.model_type.as_deref().unwrap_or("any"),
                requirements.task.as_deref().unwrap_or("any"),
                requirements.execution_provider.as_deref().unwrap_or("any"),
                target.accelerator.as_str()
            ),
        })
    }

    pub(crate) fn validate_pack_for_target(
        &self,
        pack: &BackendPackManifest,
        target: &BackendTarget,
    ) -> ManagerResult<()> {
        self.validate_pack_manifest(pack, Some(target))
    }

    pub(crate) fn select_pack(
        &self,
        index: &BackendIndex,
        backend: &str,
        preferred_profile: Option<&str>,
        target: &BackendTarget,
    ) -> ManagerResult<BackendPackManifest> {
        let mut rejected = Vec::new();
        let mut candidates = Vec::new();
        for pack in index.packs.iter().filter(|pack| pack.backend == backend) {
            if preferred_profile.is_some_and(|profile| pack.profile != profile) {
                continue;
            }
            match self.validate_pack_manifest(pack, Some(target)) {
                Ok(()) => candidates.push(pack.clone()),
                Err(error) => rejected.push(format!("{}: {error}", pack.profile)),
            }
        }
        candidates.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| compare_pack_versions(&right.pack_version, &left.pack_version))
        });
        candidates.into_iter().next().ok_or_else(|| {
            BackendManagerError::new(format!(
                "no compatible signed backend pack for `{backend}` profile `{}` on {} ({}, CUDA {:?}, driver {:?}){}",
                preferred_profile.unwrap_or("auto"),
                target.platform,
                target.accelerator.as_str(),
                target.cuda_version,
                target.driver_version,
                if rejected.is_empty() {
                    String::new()
                } else {
                    format!("; rejected candidates: {}", rejected.join("; "))
                }
            ))
        })
    }
}
