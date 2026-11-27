//! Physical region telemetry, separate from the admission ledger's metrics.

use super::gpu_regions::{GpuRegionIsolation, GpuRegionSnapshot};
use prometheus::{IntGaugeVec, Opts, Registry};

pub(crate) struct GpuRegionMetrics {
    bytes: IntGaugeVec,
    owner_bytes: IntGaugeVec,
}

impl GpuRegionMetrics {
    pub(crate) fn new(registry: &Registry) -> Self {
        let bytes = IntGaugeVec::new(
            Opts::new(
                "kapsl_gpu_region_bytes",
                "GPU backing observations, not additional budget charges. Mapped bytes do not imply worker readiness; reusable bytes remain physically committed.",
            ),
            &["device", "region", "kind", "model", "replica", "participant", "generation", "state"],
        ).expect("valid GPU region metric labels");
        let owner_bytes = IntGaugeVec::new(
            Opts::new("kapsl_gpu_region_owner_bytes", "Logical allocations within GPU regions by backend, owner and allocation class; unknown participant block usage is omitted."),
            &["device", "region", "backend", "model", "replica", "class"],
        ).expect("valid GPU region owner metric labels");
        registry
            .register(Box::new(bytes.clone()))
            .expect("register GPU region metrics");
        registry
            .register(Box::new(owner_bytes.clone()))
            .expect("register GPU region owner metrics");
        Self { bytes, owner_bytes }
    }

    pub(crate) fn observe(&self, snapshots: &[GpuRegionSnapshot]) {
        self.bytes.reset();
        self.owner_bytes.reset();
        for snapshot in snapshots {
            let device = snapshot.device_id.to_string();
            let region = snapshot.id.to_string();
            let kind = snapshot.kind.to_string();
            let (model, replica, participant, generation) = match &snapshot.isolation {
                GpuRegionIsolation::Local => {
                    ("shared".into(), "shared".into(), "local", "none".into())
                }
                GpuRegionIsolation::Participant {
                    owner,
                    participant_id,
                    generation,
                    ..
                } => (
                    owner.model_id.to_string(),
                    owner.replica_id.to_string(),
                    participant_id.as_str(),
                    generation.to_string(),
                ),
            };
            let usage = &snapshot.usage;
            for (state, bytes) in [
                ("committed", Some(usage.committed_bytes)),
                ("mapped", Some(usage.mapped_bytes)),
                ("virtual-reserved", Some(usage.virtual_reserved_bytes)),
                ("logical-allocated", usage.logical_allocated_bytes),
                ("reusable", usage.reusable_bytes),
                ("largest-free-range", usage.largest_free_range_bytes),
            ] {
                if let Some(bytes) = bytes {
                    self.bytes
                        .with_label_values::<&str>(&[
                            &device,
                            &region,
                            &kind,
                            &model,
                            &replica,
                            participant,
                            &generation,
                            state,
                        ])
                        .set(i64::try_from(bytes).unwrap_or(i64::MAX));
                }
            }
            for owner in &usage.owners {
                let (model, replica) = owner
                    .owner
                    .map(|owner| (owner.model_id.to_string(), owner.replica_id.to_string()))
                    .unwrap_or_else(|| ("unattributed".into(), "unattributed".into()));
                self.owner_bytes
                    .with_label_values::<&str>(&[
                        &device,
                        &region,
                        owner.backend,
                        &model,
                        &replica,
                        &owner.class.to_string(),
                    ])
                    .set(i64::try_from(owner.logical_bytes).unwrap_or(i64::MAX));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::memory::gpu_regions::{
        GpuRegionKind, GpuRegionOwnerUsage, GpuRegionRegistry, GpuRegionSource, GpuRegionUsage,
    };
    use crate::runtime::memory::{MemoryAllocationClass, MemoryOwner};
    use std::sync::Arc;

    struct Backing(GpuRegionUsage);
    impl GpuRegionSource for Backing {
        fn region_usage(&self) -> Option<GpuRegionUsage> {
            Some(self.0.clone())
        }
    }

    fn rendered(registry: &Registry) -> String {
        prometheus::TextEncoder::new()
            .encode_to_string(&registry.gather())
            .unwrap()
    }

    #[test]
    fn exported_metrics_identify_owner_without_inventing_free_or_logical_bytes() {
        let registry = Registry::new();
        let exporter = GpuRegionMetrics::new(&registry);
        let regions = GpuRegionRegistry::new(2);
        let backing: Arc<dyn GpuRegionSource> =
            Arc::new(Backing(GpuRegionUsage::exported(512, 256, 2048)));
        regions.register(
            GpuRegionKind::Vmm,
            GpuRegionIsolation::Participant {
                owner: MemoryOwner::external_kv(5).unwrap(),
                participant_id: "worker-5".into(),
                binding_id: "kv:5:2".into(),
                generation: 7,
            },
            &backing,
        );
        exporter.observe(&regions.snapshots());
        let text = rendered(&registry);
        assert!(text.contains("participant=\"worker-5\""));
        assert!(text.contains("generation=\"7\""));
        assert!(text.contains("model=\"2147483653\""));
        assert!(text.contains("state=\"committed\"} 512"));
        assert!(text.contains("state=\"mapped\"} 256"));
        assert!(text.contains("state=\"virtual-reserved\"} 2048"));
        assert!(!text.contains("state=\"logical-allocated\""));
        assert!(!text.contains("state=\"reusable\""));
        assert!(!text.contains("kapsl_gpu_region_owner_bytes{"));
        exporter.observe(&[]);
        assert!(!rendered(&registry).contains("kapsl_gpu_region_bytes{"));
    }

    #[test]
    fn arena_metrics_distinguish_logical_owners_from_shared_backing() {
        let registry = Registry::new();
        let exporter = GpuRegionMetrics::new(&registry);
        let regions = GpuRegionRegistry::new(0);
        let backing: Arc<dyn GpuRegionSource> = Arc::new(Backing(GpuRegionUsage {
            committed_bytes: 4096,
            mapped_bytes: 4096,
            virtual_reserved_bytes: 0,
            logical_allocated_bytes: Some(1024),
            reusable_bytes: Some(3072),
            largest_free_range_bytes: Some(2048),
            owners: vec![GpuRegionOwnerUsage {
                owner: Some(MemoryOwner::new(42, 3)),
                backend: "gguf",
                class: MemoryAllocationClass::KvCache,
                logical_bytes: 1024,
            }],
        }));
        regions.register(GpuRegionKind::Arena, GpuRegionIsolation::Local, &backing);
        exporter.observe(&regions.snapshots());
        let text = rendered(&registry);
        assert!(text.contains("model=\"shared\""));
        assert!(text.contains("state=\"committed\"} 4096"));
        assert!(text.contains("state=\"reusable\"} 3072"));
        let owner_line = text
            .lines()
            .find(|line| line.starts_with("kapsl_gpu_region_owner_bytes{"))
            .unwrap();
        assert!(owner_line.contains("backend=\"gguf\""));
        assert!(owner_line.contains("model=\"42\""));
        assert!(owner_line.contains("replica=\"3\""));
        assert!(owner_line.ends_with(" 1024"));
    }
}
