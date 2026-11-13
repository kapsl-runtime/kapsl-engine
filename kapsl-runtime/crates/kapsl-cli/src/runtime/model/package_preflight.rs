//! Memory admission before a `.aimod` archive is unpacked (ENG-05).
//!
//! `PackageLoader::load` gunzips and untars the whole package into the cache
//! before the backend can state its memory plan, so a model that cannot fit
//! used to be denied only after its weights were written to disk (68 s for an
//! 18 GB package, and the same bytes again on every retry). The manifest and
//! the model entry's size are in the first tar headers; this check runs on
//! them and denies a load that cannot fit before anything is extracted.
//!
//! The check uses a lower bound only: the model file's own size must fit in the
//! CUDA devices the manifest selects. The backend's full plan (workspace, KV)
//! is larger, so a model that passes here can still be denied later, with the
//! exact numbers; a model denied here could not have been admitted. Loads that
//! target the host, a raw model file or a directory (nothing to unpack) are not
//! checked.

use super::*;

/// Budget and room of one CUDA device under the memory authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeviceRoom {
    pub(crate) budget_bytes: usize,
    pub(crate) available_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PreflightVerdict {
    Fits,
    /// Larger than the whole budget of the selected devices: no eviction helps.
    NeverFits {
        budget_bytes: usize,
    },
    /// Larger than the room left now; evicting lower-priority models may help.
    NoRoomNow {
        available_bytes: usize,
    },
}

/// Compare the model's size with the selected devices. `sharded` placements
/// (pipeline, tensor parallel) spread one copy over the devices, so their
/// rooms add up; otherwise every device needs a full copy and the best single
/// device is the upper bound on what could be granted.
pub(crate) fn preflight_verdict(
    model_bytes: u64,
    rooms: &[DeviceRoom],
    sharded: bool,
) -> PreflightVerdict {
    if rooms.is_empty() {
        return PreflightVerdict::Fits;
    }
    let combine = |field: fn(&DeviceRoom) -> usize| {
        let values = rooms.iter().map(field);
        if sharded {
            values.fold(0usize, usize::saturating_add)
        } else {
            values.max().unwrap_or(0)
        }
    };
    let budget_bytes = combine(|room| room.budget_bytes);
    let available_bytes = combine(|room| room.available_bytes);
    let needed = usize::try_from(model_bytes).unwrap_or(usize::MAX);
    if needed > budget_bytes {
        PreflightVerdict::NeverFits { budget_bytes }
    } else if needed > available_bytes {
        PreflightVerdict::NoRoomNow { available_bytes }
    } else {
        PreflightVerdict::Fits
    }
}

fn placement_is_sharded(strategy: Option<&str>, topology: &str, has_pipeline_stages: bool) -> bool {
    let spreads = |value: &str| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "pipeline" | "pipeline-parallel" | "tensor-parallel" | "tensor_parallel" | "auto"
        )
    };
    has_pipeline_stages || spreads(topology) || strategy.is_some_and(spreads)
}

/// Deny a package that cannot fit before it is extracted. `room` reports the
/// authority's budget and available bytes for an engine CUDA device id (`None`
/// when the device is not managed, which disables the check).
pub(crate) fn preflight_package_admission(
    model_id: u32,
    model_path: &Path,
    placement: &ModelLoadPlacement,
    device_info: &DeviceInfo,
    room: impl Fn(usize) -> Option<DeviceRoom>,
    sampling: CudaSamplingState,
) -> Result<(), DynError> {
    let started = Instant::now();
    // Anything unreadable falls through to the real load, which reports it.
    let Ok(Some(header)) = inspect_package_header(model_path) else {
        return Ok(());
    };
    let requirements = &header.manifest.hardware_requirements;
    let Ok(selection) = select_mesh_devices(requirements, device_info) else {
        return Ok(());
    };
    let cuda_ids: Vec<usize> = selection
        .devices
        .iter()
        .filter(|device| matches!(device.backend, kapsl_hal::device::DeviceBackend::Cuda))
        .map(|device| device.id)
        .collect();
    if cuda_ids.is_empty() {
        return Ok(());
    }
    let rooms: Option<Vec<DeviceRoom>> = cuda_ids.iter().map(|&id| room(id)).collect();
    let Some(mut rooms) = rooms else {
        return Ok(());
    };
    // Stale GPU samples: the same margin (or refusal) the load gate applies.
    for room in &mut rooms {
        room.available_bytes = sampling.clamp_available(room.available_bytes, room.budget_bytes);
    }
    let sharded = placement_is_sharded(
        requirements.strategy.as_deref(),
        &placement.topology,
        manifest_llm_pipeline_stages(&header.manifest).is_some(),
    );
    let verdict = preflight_verdict(header.model_bytes, &rooms, sharded);
    log::debug!(
        "[admission-preflight] model {model_id}: {} byte model entry vs {rooms:?} (sharded={sharded}) -> {verdict:?} in {:?}",
        header.model_bytes,
        started.elapsed()
    );
    // Evicting models cannot refresh GPU samples, so a denial that only exists
    // because of stale ones is not reclaimable.
    let fresh = sampling == CudaSamplingState::Fresh;
    let (detail, reclaimable) = match verdict {
        PreflightVerdict::Fits => return Ok(()),
        PreflightVerdict::NeverFits { budget_bytes } => (
            format!(
                "the model file alone is {} bytes but the selected CUDA device(s) have a {budget_bytes}-byte safe budget in total, so no unloading can make it fit",
                header.model_bytes
            ),
            false,
        ),
        PreflightVerdict::NoRoomNow { available_bytes } => (
            format!(
                "the model file alone is {} bytes but only {available_bytes} bytes are available on the selected CUDA device(s){}",
                header.model_bytes,
                match sampling {
                    CudaSamplingState::Fresh => String::new(),
                    CudaSamplingState::Stale { age } => format!(
                        " (GPU memory samples are {}s old, so a safety margin is held back)",
                        age.as_secs()
                    ),
                    CudaSamplingState::Failed { age } => format!(
                        " (GPU memory sampling has failed for {}s, so no new CUDA load is admitted)",
                        age.as_secs()
                    ),
                }
            ),
            fresh,
        ),
    };
    let owner = MemoryOwner::new(model_id, 0);
    let mut plan = MemoryPlan::new();
    let share = usize::try_from(header.model_bytes).unwrap_or(usize::MAX)
        / if sharded { cuda_ids.len().max(1) } else { 1 };
    for &device_id in &cuda_ids {
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Cuda { device_id },
            owner,
            MemoryAllocationClass::PersistentWeights,
            share,
        ));
    }
    let failure = MemoryAdmissionFailure::new(
        owner,
        resolve_model_priority_weight(&header.manifest, model_id),
        &plan,
        format!(
            "{detail} (checked from the archive header in {:?}; the package {} was not extracted)",
            started.elapsed(),
            model_path.display()
        ),
    );
    Err(Box::new(if reclaimable {
        failure
    } else {
        failure.unreclaimable()
    }))
}

impl ModelRuntime {
    /// Run [`preflight_package_admission`] against the live memory authority.
    pub(crate) fn preflight_package_admission(
        &self,
        model_id: u32,
        model_path: &Path,
        placement: &ModelLoadPlacement,
    ) -> Result<(), DynError> {
        let snapshot = self.resources.memory().snapshot();
        let sampling = self.resources.memory().cuda_sampling_state();
        preflight_package_admission(
            model_id,
            model_path,
            placement,
            self.device_info(),
            |device_id| {
                snapshot
                    .domain(&MemoryDomain::Cuda { device_id })
                    .map(|domain| DeviceRoom {
                        budget_bytes: domain.budget_bytes,
                        available_bytes: domain.available_bytes,
                    })
            },
            sampling,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_hal::device::{Device, DeviceBackend};
    use std::io::Write;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn cuda_host(devices: usize) -> DeviceInfo {
        DeviceInfo {
            cpu_cores: 8,
            total_memory: 64 * 1024 * 1024,
            os_type: "linux".to_string(),
            os_release: "test".to_string(),
            has_cuda: true,
            has_metal: false,
            has_rocm: false,
            has_directml: false,
            devices: (0..devices)
                .map(|id| Device {
                    id,
                    name: format!("fake-cuda-{id}"),
                    backend: DeviceBackend::Cuda,
                    memory_mb: 24 * 1024,
                    compute_units: 1,
                    pci_bus_id: None,
                    partition_id: None,
                    driver_version: None,
                    cuda_version: Some("12.6".to_string()),
                    compute_capability: Some("8.6".to_string()),
                    utilization_gpu_pct: None,
                    temperature_c: None,
                    supports_fp16: true,
                    supports_int8: true,
                })
                .collect(),
        }
    }

    /// A `.aimod` whose model entry's tar header declares `declared_bytes`
    /// while carrying only `actual_bytes` of data. Reading the header is all the
    /// preflight does; extracting it fails with an unexpected EOF, which is how
    /// the tests tell that the archive was (not) unpacked.
    fn package(dir: &Path, declared_bytes: u64, actual_bytes: usize, strategy: &str) -> PathBuf {
        let manifest = serde_json::json!({
            "project_name": "big-model",
            "framework": "gguf",
            "version": "1.0.0",
            "created_at": "2026-11-13",
            "model_file": "model.gguf",
            "hardware_requirements": {
                "preferred_provider": "cuda",
                "strategy": strategy,
            },
        });
        let manifest = serde_json::to_vec(&manifest).unwrap();
        let path = dir.join("big.aimod");
        let encoder = flate2::write::GzEncoder::new(
            std::fs::File::create(&path).unwrap(),
            flate2::Compression::fast(),
        );
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_path("metadata.json").unwrap();
        header.set_size(manifest.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append(&header, manifest.as_slice()).unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_path("model.gguf").unwrap();
        header.set_size(declared_bytes);
        header.set_mode(0o644);
        header.set_cksum();
        // Written by hand: `append` would insist on `declared_bytes` of data.
        let encoder = builder.get_mut();
        encoder.write_all(header.as_bytes()).unwrap();
        encoder.write_all(&vec![0u8; actual_bytes]).unwrap();
        encoder.flush().unwrap();
        drop(builder);
        path
    }

    fn rooms(budget: u64, available: u64) -> impl Fn(usize) -> Option<DeviceRoom> {
        move |_| {
            Some(DeviceRoom {
                budget_bytes: budget as usize,
                available_bytes: available as usize,
            })
        }
    }

    fn placement() -> ModelLoadPlacement {
        ModelLoadPlacement::new("", 1)
    }

    #[test]
    fn an_oversized_package_is_denied_from_its_header_within_five_seconds() {
        let dir = tempfile::tempdir().unwrap();
        // 40 GiB declared on a 24 GiB-budget card; the archive holds 1 MiB.
        let path = package(dir.path(), 40 * GIB, 1 << 20, "round-robin");
        let started = Instant::now();
        let error = preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(1),
            rooms(22 * GIB, 22 * GIB),
            CudaSamplingState::Fresh,
        )
        .expect_err("40 GiB cannot fit a 22 GiB budget");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        let failure = error
            .downcast_ref::<MemoryAdmissionFailure>()
            .expect("typed denial, so the API answers 409 memory_admission");
        assert!(
            !failure.is_reclaimable(),
            "no eviction can fit 40 GiB in 22 GiB"
        );
        let message = error.to_string();
        assert!(message.contains(&(40 * GIB).to_string()), "{message}");
        assert!(message.contains("not extracted"), "{message}");
        assert_eq!(
            failure.requested_bytes(),
            &[(MemoryDomain::Cuda { device_id: 0 }, (40 * GIB) as usize)]
        );
        // The archive really is not unpackable: only the preflight could deny it.
        assert!(PackageLoader::load(&path).is_err());
    }

    #[test]
    fn a_package_that_needs_models_evicted_is_denied_as_reclaimable() {
        let dir = tempfile::tempdir().unwrap();
        let path = package(dir.path(), 18 * GIB, 1 << 20, "round-robin");
        let error = preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(1),
            rooms(22 * GIB, 10 * GIB),
            CudaSamplingState::Fresh,
        )
        .expect_err("18 GiB does not fit the 10 GiB left");
        let failure = error.downcast_ref::<MemoryAdmissionFailure>().unwrap();
        assert!(
            failure.is_reclaimable(),
            "evicting a lower-priority model could fit it"
        );
    }

    #[test]
    fn a_package_that_fits_passes_and_is_left_for_the_real_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = package(dir.path(), 4 * GIB, 1 << 20, "round-robin");
        preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(1),
            rooms(22 * GIB, 20 * GIB),
            CudaSamplingState::Fresh,
        )
        .expect("4 GiB fits");
    }

    #[test]
    fn sharded_placements_add_the_devices_up() {
        let rooms_two = [DeviceRoom {
            budget_bytes: 22 * GIB as usize,
            available_bytes: 20 * GIB as usize,
        }; 2];
        let model = 30 * GIB;
        assert_eq!(
            preflight_verdict(model, &rooms_two, false),
            PreflightVerdict::NeverFits {
                budget_bytes: 22 * GIB as usize
            }
        );
        assert_eq!(
            preflight_verdict(model, &rooms_two, true),
            PreflightVerdict::Fits
        );
    }

    #[test]
    fn unknown_rooms_raw_files_and_host_loads_are_not_preflighted() {
        let dir = tempfile::tempdir().unwrap();
        let path = package(dir.path(), 40 * GIB, 1 << 20, "round-robin");
        // Device not managed by the authority: no data, no denial.
        preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(1),
            |_| None,
            CudaSamplingState::Fresh,
        )
        .unwrap();
        // A raw model file has nothing to unpack.
        let raw = dir.path().join("m.gguf");
        std::fs::write(&raw, b"GGUF").unwrap();
        preflight_package_admission(
            7,
            &raw,
            &placement(),
            &cuda_host(1),
            rooms(GIB, 0),
            CudaSamplingState::Fresh,
        )
        .unwrap();
        // No CUDA device in the inventory: host loads are left to the real admission.
        preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(0),
            rooms(GIB, 0),
            CudaSamplingState::Fresh,
        )
        .unwrap();
        // A file that is not an archive is left for the loader to report.
        let junk = dir.path().join("junk.aimod");
        std::fs::write(&junk, b"not a package").unwrap();
        preflight_package_admission(
            7,
            &junk,
            &placement(),
            &cuda_host(1),
            rooms(GIB, 0),
            CudaSamplingState::Fresh,
        )
        .unwrap();
    }

    #[test]
    fn the_priority_arbiter_skips_unreclaimable_denials() {
        let owner = MemoryOwner::new(7, 0);
        let mut plan = MemoryPlan::new();
        plan.push(MemoryClaim::runtime(
            MemoryDomain::Cuda { device_id: 0 },
            owner,
            MemoryAllocationClass::PersistentWeights,
            1,
        ));
        let failure = MemoryAdmissionFailure::new(owner, 1, &plan, "x").unreclaimable();
        assert!(!failure.is_reclaimable());
    }

    #[test]
    fn stale_and_failed_gpu_samples_deny_without_inviting_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let path = package(dir.path(), 18 * GIB, 1 << 20, "round-robin");
        let age = Duration::from_secs(120);
        // 18 GiB fits the 21 GiB left even after the 10% margin (2.2 GiB of 22),
        // but not the 19 GiB left.
        let stale = preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(1),
            rooms(22 * GIB, 21 * GIB),
            CudaSamplingState::Stale { age },
        );
        assert!(stale.is_ok(), "18 GiB <= 21 GiB - 2.2 GiB");
        let tight = preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(1),
            rooms(22 * GIB, 19 * GIB),
            CudaSamplingState::Stale { age },
        )
        .expect_err("18 GiB > 19 GiB - 2.2 GiB");
        let failure = tight.downcast_ref::<MemoryAdmissionFailure>().unwrap();
        assert!(!failure.is_reclaimable(), "evicting cannot refresh samples");
        assert!(tight.to_string().contains("120s old"), "{tight}");
        let failed = preflight_package_admission(
            7,
            &path,
            &placement(),
            &cuda_host(1),
            rooms(22 * GIB, 22 * GIB),
            CudaSamplingState::Failed {
                age: Duration::from_secs(400),
            },
        )
        .expect_err("fail closed");
        assert!(!failed
            .downcast_ref::<MemoryAdmissionFailure>()
            .unwrap()
            .is_reclaimable());
        assert!(
            failed.to_string().contains("sampling has failed"),
            "{failed}"
        );
    }
}
