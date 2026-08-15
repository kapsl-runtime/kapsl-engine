use super::*;

#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeSamples {
    pub(crate) process_memory_bytes: usize,
    pub(crate) total_system_memory_bytes: Option<usize>,
    pub(crate) gpu_utilization: f64,
    pub(crate) gpu_memory_bytes: Option<usize>,
    pub(crate) gpu_memory_total_bytes: Option<usize>,
    /// VRAM held by co-tenant processes (summed across devices), from
    /// `sample_foreign_vram`. Subtracted from `gpu_memory_bytes` in the pressure
    /// calculation so a trainer sharing the GPU throttles kapsl's *concurrency*
    /// (via the KV ceiling) rather than tripping Conserve/Emergency and
    /// truncating every request's max_new_tokens. `None` when the co-tenancy
    /// probe is disabled or unavailable, which leaves pressure behavior exactly
    /// as before.
    pub(crate) foreign_gpu_memory_bytes: Option<usize>,
    pub(crate) collected_at_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimePressureState {
    Normal = 0,
    Conserve = 1,
    Emergency = 2,
}

impl RuntimePressureState {
    pub(crate) fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Conserve,
            2 => Self::Emergency,
            _ => Self::Normal,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Conserve => "conserve",
            Self::Emergency => "emergency",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimePressureConfig {
    pub(crate) memory_conserve_ratio: f64,
    pub(crate) memory_emergency_ratio: f64,
    pub(crate) gpu_util_conserve_ratio: f64,
    pub(crate) gpu_util_emergency_ratio: f64,
    pub(crate) gpu_mem_conserve_ratio: f64,
    pub(crate) gpu_mem_emergency_ratio: f64,
    pub(crate) conserve_max_new_tokens: Option<u32>,
    pub(crate) emergency_max_new_tokens: Option<u32>,
}

impl RuntimePressureConfig {
    pub(crate) fn from_env() -> Self {
        let memory_conserve_pct = optional_env_var(PRESSURE_MEMORY_CONSERVE_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(80.0)
            .clamp(0.0, 100.0);
        let memory_emergency_pct = optional_env_var(PRESSURE_MEMORY_EMERGENCY_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(90.0)
            .clamp(memory_conserve_pct, 100.0);
        let gpu_util_conserve_pct = optional_env_var(PRESSURE_GPU_UTIL_CONSERVE_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(85.0)
            .clamp(0.0, 100.0);
        let gpu_util_emergency_pct = optional_env_var(PRESSURE_GPU_UTIL_EMERGENCY_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(95.0)
            .clamp(gpu_util_conserve_pct, 100.0);
        let gpu_mem_conserve_pct = optional_env_var(PRESSURE_GPU_MEM_CONSERVE_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(85.0)
            .clamp(0.0, 100.0);
        let gpu_mem_emergency_pct = optional_env_var(PRESSURE_GPU_MEM_EMERGENCY_PCT_ENV)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(95.0)
            .clamp(gpu_mem_conserve_pct, 100.0);
        let conserve_max_new_tokens = optional_env_var(PRESSURE_CONSERVE_MAX_TOKENS_ENV)
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 0);
        let emergency_max_new_tokens = optional_env_var(PRESSURE_EMERGENCY_MAX_TOKENS_ENV)
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 0)
            .or(Some(128));

        Self {
            memory_conserve_ratio: memory_conserve_pct / 100.0,
            memory_emergency_ratio: memory_emergency_pct / 100.0,
            gpu_util_conserve_ratio: gpu_util_conserve_pct / 100.0,
            gpu_util_emergency_ratio: gpu_util_emergency_pct / 100.0,
            gpu_mem_conserve_ratio: gpu_mem_conserve_pct / 100.0,
            gpu_mem_emergency_ratio: gpu_mem_emergency_pct / 100.0,
            conserve_max_new_tokens,
            emergency_max_new_tokens,
        }
    }

    pub(crate) fn max_new_tokens_cap(&self, state: RuntimePressureState) -> Option<u32> {
        match state {
            RuntimePressureState::Normal => None,
            RuntimePressureState::Conserve => self.conserve_max_new_tokens,
            RuntimePressureState::Emergency => self.emergency_max_new_tokens,
        }
    }
}

/// Memory pressure is derived solely from the authority snapshot. Raw sampler
/// fields remain available for the system API, while only non-memory GPU
/// utilization is consumed directly here.
pub(crate) fn evaluate_authority_pressure_state(
    memory: &MemorySnapshot,
    gpu_utilization: f64,
    config: &RuntimePressureConfig,
) -> RuntimePressureState {
    let host_ratio = memory.max_host_ratio();
    let device_ratio = memory.max_device_ratio();
    let gpu_util_ratio = gpu_utilization.clamp(0.0, 1.0);

    if host_ratio.is_some_and(|ratio| ratio >= config.memory_emergency_ratio)
        || device_ratio.is_some_and(|ratio| ratio >= config.gpu_mem_emergency_ratio)
        || gpu_util_ratio >= config.gpu_util_emergency_ratio
    {
        return RuntimePressureState::Emergency;
    }
    if host_ratio.is_some_and(|ratio| ratio >= config.memory_conserve_ratio)
        || device_ratio.is_some_and(|ratio| ratio >= config.gpu_mem_conserve_ratio)
        || gpu_util_ratio >= config.gpu_util_conserve_ratio
    {
        return RuntimePressureState::Conserve;
    }
    RuntimePressureState::Normal
}

#[derive(Debug, Clone)]
pub(crate) struct ThroughputSample {
    pub(crate) last_total: u64,
    pub(crate) last_timestamp: Instant,
    pub(crate) throughput: f64,
}

/// Maximum number of recent latency observations retained per model.
pub(crate) const LATENCY_WINDOW_CAPACITY: usize = 256;

/// Bounded ring of recent end-to-end inference latencies (milliseconds) for a
/// single model. Surfaced as avg/p99 on `GET /api/models` so the enterprise
/// autoscaler can scale workers on latency SLOs, not just queue depth.
#[derive(Debug, Clone, Default)]
pub(crate) struct LatencyWindow {
    samples: std::collections::VecDeque<f64>,
}

impl LatencyWindow {
    pub(crate) fn record(&mut self, latency_ms: f64) {
        if !latency_ms.is_finite() || latency_ms < 0.0 {
            return;
        }
        if self.samples.len() >= LATENCY_WINDOW_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(latency_ms);
    }

    pub(crate) fn average_ms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    pub(crate) fn p99_ms(&self) -> f64 {
        self.percentile_ms(0.99)
    }

    /// Nearest-rank percentile over the current window; 0.0 when empty.
    pub(crate) fn percentile_ms(&self, quantile: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = (quantile.clamp(0.0, 1.0) * sorted.len() as f64)
            .ceil()
            .max(1.0) as usize;
        sorted[rank.min(sorted.len()) - 1]
    }
}

pub(crate) fn update_throughput(
    samples: &mut HashMap<u32, ThroughputSample>,
    model_id: u32,
    total_inferences: u64,
    now: Instant,
) -> f64 {
    if let Some(entry) = samples.get_mut(&model_id) {
        let elapsed = now.duration_since(entry.last_timestamp).as_secs_f64();
        let delta = total_inferences.saturating_sub(entry.last_total);
        let throughput = if elapsed > 0.0 {
            delta as f64 / elapsed
        } else {
            entry.throughput
        };
        entry.last_total = total_inferences;
        entry.last_timestamp = now;
        entry.throughput = throughput;
        throughput
    } else {
        samples.insert(
            model_id,
            ThroughputSample {
                last_total: total_inferences,
                last_timestamp: now,
                throughput: 0.0,
            },
        );
        0.0
    }
}

pub(crate) fn sample_nvidia_smi() -> Option<(f64, usize, usize)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    aggregate_gpu_samples(&stdout, device_vram_cap_bytes)
}

/// Parse `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total` CSV
/// (one line per GPU, in device-index order) into average utilization plus
/// summed used/total VRAM bytes. `cap_for_device` supplies the cooperative
/// software-vGPU cap per device index: each GPU's reported *total* is clamped to
/// its cap so back-pressure and the enterprise admission guard reason about the
/// virtual slice, not the physical card. The cap never inflates the reported
/// total (it is a `min`), and a missing cap leaves the device unchanged. Split
/// out from the command invocation so the clamp is unit-testable without a GPU.
fn aggregate_gpu_samples(
    stdout: &str,
    cap_for_device: impl Fn(usize) -> Option<usize>,
) -> Option<(f64, usize, usize)> {
    let mut total_util = 0.0;
    let mut total_mem_bytes: usize = 0;
    let mut total_mem_capacity_bytes: usize = 0;
    let mut count = 0.0;

    for (device_index, line) in stdout.lines().enumerate() {
        let mut parts = line.split(',');
        let util_str = parts.next().map(|s| s.trim());
        let mem_str = parts.next().map(|s| s.trim());
        let mem_total_str = parts.next().map(|s| s.trim());
        let (util_str, mem_str, mem_total_str) = match (util_str, mem_str, mem_total_str) {
            (Some(util_str), Some(mem_str), Some(mem_total_str)) => {
                (util_str, mem_str, mem_total_str)
            }
            _ => continue,
        };

        if let (Ok(util), Ok(mem_mb), Ok(mem_total_mb)) = (
            util_str.parse::<f64>(),
            mem_str.parse::<f64>(),
            mem_total_str.parse::<f64>(),
        ) {
            let mem_total_bytes = (mem_total_mb * 1024.0 * 1024.0) as usize;
            let capped_total = cap_for_device(device_index)
                .map_or(mem_total_bytes, |cap| mem_total_bytes.min(cap));
            total_util += util;
            total_mem_bytes = total_mem_bytes.saturating_add((mem_mb * 1024.0 * 1024.0) as usize);
            total_mem_capacity_bytes = total_mem_capacity_bytes.saturating_add(capped_total);
            count += 1.0;
        }
    }

    if count == 0.0 {
        return None;
    }

    let avg_util = (total_util / count) / 100.0;
    Some((avg_util, total_mem_bytes, total_mem_capacity_bytes))
}

/// Bytes of device VRAM held by compute processes that are NOT this engine,
/// keyed by physical device index. Powers the co-tenancy ceiling: when another
/// process (e.g. a training job) shares the GPU, its footprint is subtracted
/// from the KV pool's effective ceiling so kapsl throttles concurrency instead
/// of OOMing the neighbor.
///
/// Shells out to `nvidia-smi` twice: once to map each GPU's UUID to its device
/// index (the compute-apps query reports UUID, not index), then to enumerate
/// running compute apps. Our own PID is excluded. Returns an empty map on any
/// failure so callers treat "no signal" as "no co-tenant" and default,
/// single-tenant behavior is unchanged.
///
/// Caveat: under CUDA MPS every client's memory is attributed to the MPS server
/// PID, so per-process filtering collapses. In that mode callers should fall
/// back to device-wide `memory.used` minus kapsl's own known block bytes.
pub(crate) fn sample_foreign_vram() -> HashMap<usize, usize> {
    let uuid_to_index = query_gpu_uuid_index();
    if uuid_to_index.is_empty() {
        return HashMap::new();
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let stdout = match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        _ => return HashMap::new(),
    };
    // Under CUDA MPS every client's memory is attributed to the MPS server
    // process, so the per-PID split below cannot separate our own bytes from a
    // neighbor's — the guard is effectively blind. Warn once so an operator does
    // not mistake the over-subtracted ceiling (or an un-throttled neighbor) for
    // a bug; MPS isolation needs MPS resource limits or MIG instead.
    if compute_apps_show_mps(&stdout) {
        static MPS_WARNED: std::sync::Once = std::sync::Once::new();
        MPS_WARNED.call_once(|| {
            log::warn!(
                "co-tenancy guard: CUDA MPS server detected in nvidia-smi compute-apps; \
                 per-process VRAM attribution collapses to the MPS server PID, so the \
                 foreign-VRAM ceiling is unreliable under MPS (use MPS resource limits or \
                 MIG for isolation in this mode)"
            );
        });
    }
    aggregate_foreign_vram(&stdout, &uuid_to_index, std::process::id())
}

/// Whether any `nvidia-smi --query-compute-apps` row is the CUDA MPS server,
/// which signals that per-process VRAM attribution has collapsed. Split from the
/// command invocation so it is unit-testable without a GPU. Expects the
/// `pid,process_name,used_memory,gpu_uuid` column layout.
fn compute_apps_show_mps(stdout: &str) -> bool {
    stdout.lines().any(|line| {
        line.split(',')
            .nth(1)
            .is_some_and(|name| name.trim().contains("nvidia-cuda-mps-server"))
    })
}

/// Map each GPU's UUID to its device index. The compute-apps query reports a
/// UUID rather than the index the rest of the runtime keys on, so this bridges
/// the two. Empty on any failure.
fn query_gpu_uuid_index() -> HashMap<String, usize> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=uuid,index", "--format=csv,noheader,nounits"])
        .output();
    let stdout = match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        _ => return HashMap::new(),
    };
    parse_uuid_index(&stdout)
}

/// Parse `nvidia-smi --query-gpu=uuid,index` CSV into a UUID→index map. Split
/// from the command invocation so it is unit-testable without a GPU.
fn parse_uuid_index(stdout: &str) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for line in stdout.lines() {
        let mut parts = line.split(',');
        let (Some(uuid), Some(index)) = (parts.next(), parts.next()) else {
            continue;
        };
        if let Ok(index) = index.trim().parse::<usize>() {
            map.insert(uuid.trim().to_string(), index);
        }
    }
    map
}

/// Sum per-process VRAM by device index, excluding `own_pid`. Split out from the
/// `nvidia-smi` invocation so it is unit-testable without a GPU. Rows whose UUID
/// is not in `uuid_to_index` are skipped rather than misattributed to device 0.
/// Expects the `pid,process_name,used_memory,gpu_uuid` column layout; the
/// process name is used only for MPS detection (see `compute_apps_show_mps`).
fn aggregate_foreign_vram(
    stdout: &str,
    uuid_to_index: &HashMap<String, usize>,
    own_pid: u32,
) -> HashMap<usize, usize> {
    let mut foreign: HashMap<usize, usize> = HashMap::new();
    for line in stdout.lines() {
        let mut parts = line.split(',');
        let (Some(pid_str), Some(_name), Some(used_str), Some(uuid_str)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let (Ok(pid), Ok(used_mb)) = (
            pid_str.trim().parse::<u32>(),
            used_str.trim().parse::<f64>(),
        ) else {
            continue;
        };
        if pid == own_pid {
            continue;
        }
        let Some(&device) = uuid_to_index.get(uuid_str.trim()) else {
            continue;
        };
        let used_bytes = (used_mb * 1024.0 * 1024.0) as usize;
        let entry = foreign.entry(device).or_insert(0);
        *entry = entry.saturating_add(used_bytes);
    }
    foreign
}

/// Prometheus gauges and transition logs for the co-tenancy KV ceiling, fed
/// with the `CeilingSample`s returned by `refresh_ceilings` each monitor tick.
/// Without this the shrink-fast/recover-slow behavior is invisible from outside
/// the process: an operator sees throughput drop (or the autoscaler hold back)
/// with nothing to correlate it against. Only constructed when the co-tenancy
/// guard is enabled, so the default configuration exports no new series.
pub(crate) struct CotenancyCeilingExporter {
    /// Smoothed live ceiling — the value that actually drives per-engine caps.
    ceiling_bytes: prometheus::IntGaugeVec,
    /// Instantaneous target; the smoothed gauge lags it during recovery, and
    /// that visible gap is the grow-slow hysteresis.
    ceiling_target_bytes: prometheus::IntGaugeVec,
    foreign_vram_bytes: prometheus::IntGaugeVec,
    squeeze_active: prometheus::IntGaugeVec,
    /// Last squeeze state per device, so the squeeze/recover log lines fire on
    /// transitions only — a steady trainer produces one WARN, not one per tick.
    squeezed_devices: HashMap<usize, bool>,
}

impl CotenancyCeilingExporter {
    pub(crate) fn new(registry: &prometheus::Registry) -> Self {
        let ceiling_bytes = prometheus::IntGaugeVec::new(
            prometheus::Opts::new(
                "kapsl_cotenancy_kv_ceiling_bytes",
                "Live smoothed foreign-aware KV ceiling per device (bytes); drives per-engine \
                 KV caps. Shrinks immediately under a co-tenant, recovers a quarter of the \
                 remaining gap per tick.",
            ),
            &["device"],
        )
        .unwrap();
        let ceiling_target_bytes = prometheus::IntGaugeVec::new(
            prometheus::Opts::new(
                "kapsl_cotenancy_kv_ceiling_target_bytes",
                "Instantaneous foreign-aware KV ceiling per device (bytes): declared budget \
                 minus co-tenant VRAM minus safety reserve. The smoothed ceiling lags this \
                 while recovering.",
            ),
            &["device"],
        )
        .unwrap();
        let foreign_vram_bytes = prometheus::IntGaugeVec::new(
            prometheus::Opts::new(
                "kapsl_cotenancy_foreign_vram_bytes",
                "VRAM held by co-tenant (non-kapsl) compute processes per device (bytes).",
            ),
            &["device"],
        )
        .unwrap();
        let squeeze_active = prometheus::IntGaugeVec::new(
            prometheus::Opts::new(
                "kapsl_cotenancy_squeeze_active",
                "1 while a co-tenant is squeezing this device's KV ceiling (queue-depth \
                 scale-ups suppressed), else 0.",
            ),
            &["device"],
        )
        .unwrap();
        registry
            .register(Box::new(ceiling_bytes.clone()))
            .expect("Failed to register kapsl_cotenancy_kv_ceiling_bytes");
        registry
            .register(Box::new(ceiling_target_bytes.clone()))
            .expect("Failed to register kapsl_cotenancy_kv_ceiling_target_bytes");
        registry
            .register(Box::new(foreign_vram_bytes.clone()))
            .expect("Failed to register kapsl_cotenancy_foreign_vram_bytes");
        registry
            .register(Box::new(squeeze_active.clone()))
            .expect("Failed to register kapsl_cotenancy_squeeze_active");
        Self {
            ceiling_bytes,
            ceiling_target_bytes,
            foreign_vram_bytes,
            squeeze_active,
            squeezed_devices: HashMap::new(),
        }
    }

    pub(crate) fn observe(&mut self, samples: &[CeilingSample]) {
        for sample in samples {
            let device = sample.device_id.to_string();
            self.ceiling_bytes
                .with_label_values(&[device.as_str()])
                .set(sample.smoothed_bytes as i64);
            self.ceiling_target_bytes
                .with_label_values(&[device.as_str()])
                .set(sample.target_bytes as i64);
            self.foreign_vram_bytes
                .with_label_values(&[device.as_str()])
                .set(sample.foreign_bytes as i64);
            self.squeeze_active
                .with_label_values(&[device.as_str()])
                .set(sample.squeezed as i64);

            // Every ceiling step at debug so the shrink-fast/recover-slow ramp
            // can be traced tick-by-tick; the operator-facing lines below fire
            // only on squeeze transitions.
            if sample.smoothed_bytes != sample.previous_bytes {
                log::debug!(
                    "co-tenancy guard: device {} KV ceiling {}B -> {}B (target {}B, foreign {}B)",
                    sample.device_id,
                    sample.previous_bytes,
                    sample.smoothed_bytes,
                    sample.target_bytes,
                    sample.foreign_bytes,
                );
            }

            let was_squeezed = self
                .squeezed_devices
                .insert(sample.device_id, sample.squeezed)
                .unwrap_or(false);
            if sample.squeezed && !was_squeezed {
                log::warn!(
                    "co-tenancy guard: device {} KV ceiling squeezed to {}B by {}B of foreign \
                     VRAM (instantaneous target {}B); admission is backing off and queue-depth \
                     scale-ups are suppressed",
                    sample.device_id,
                    sample.smoothed_bytes,
                    sample.foreign_bytes,
                    sample.target_bytes,
                );
            } else if !sample.squeezed && was_squeezed {
                log::info!(
                    "co-tenancy guard: device {} KV ceiling recovered to {}B (target {}B); \
                     scale-up suppression released",
                    sample.device_id,
                    sample.smoothed_bytes,
                    sample.target_bytes,
                );
            }
        }
    }
}

/// One normalized metric family for the authority snapshot. Every series uses
/// the same domain/model/replica/class/state labels, so host, CUDA, and provider
/// memory can be compared without backend-specific joins.
pub(crate) struct MemorySnapshotExporter {
    bytes: prometheus::IntGaugeVec,
}

impl MemorySnapshotExporter {
    pub(crate) fn new(registry: &prometheus::Registry) -> Self {
        let bytes = prometheus::IntGaugeVec::new(
            prometheus::Opts::new(
                "kapsl_memory_authority_bytes",
                "Unified memory authority bytes by domain, model, replica, allocation class, and accounting state.",
            ),
            &["domain", "model", "replica", "class", "state"],
        )
        .expect("valid memory authority metric labels");
        registry
            .register(Box::new(bytes.clone()))
            .expect("Failed to register kapsl_memory_authority_bytes");
        Self { bytes }
    }

    pub(crate) fn observe(&self, snapshot: &MemorySnapshot) {
        self.bytes.reset();
        let set =
            |domain: &str, model: &str, replica: &str, class: &str, state: &str, bytes: usize| {
                self.bytes
                    .with_label_values(&[domain, model, replica, class, state])
                    .set(i64::try_from(bytes).unwrap_or(i64::MAX));
            };
        for row in &snapshot.rows {
            let domain = row.domain.to_string();
            let model = row.owner.model_id.to_string();
            let replica = row.owner.replica_id.to_string();
            let class = row.class.to_string();
            for (state, bytes) in [
                ("planned", row.planned_bytes),
                ("reserved", row.reserved_bytes),
                ("committed", row.committed_bytes),
                ("observed", row.observed_bytes),
            ] {
                set(&domain, &model, &replica, &class, state, bytes);
            }
        }
        for domain in &snapshot.domains {
            let label = domain.domain.to_string();
            for (state, bytes) in [
                ("planned", domain.planned_bytes),
                ("reserved", domain.reserved_bytes),
                ("committed", domain.committed_bytes),
                ("observed", domain.observed_bytes),
                ("budget", domain.budget_bytes),
                ("available", domain.available_bytes),
            ] {
                set(&label, "all", "all", "all", state, bytes);
            }
        }
    }
}

#[cfg(test)]
mod latency_window_tests {
    use super::LatencyWindow;
    use super::LATENCY_WINDOW_CAPACITY;

    #[test]
    fn empty_window_reports_zero() {
        let window = LatencyWindow::default();
        assert_eq!(window.average_ms(), 0.0);
        assert_eq!(window.p99_ms(), 0.0);
    }

    #[test]
    fn computes_average_and_nearest_rank_p99() {
        let mut window = LatencyWindow::default();
        for ms in 1..=100 {
            window.record(ms as f64);
        }
        assert!((window.average_ms() - 50.5).abs() < 1e-9);
        // Nearest-rank p99 of 1..=100 is rank ceil(0.99*100)=99 -> value 99.
        assert_eq!(window.p99_ms(), 99.0);
        assert_eq!(window.percentile_ms(0.5), 50.0);
        assert_eq!(window.percentile_ms(1.0), 100.0);
    }

    #[test]
    fn discards_invalid_samples() {
        let mut window = LatencyWindow::default();
        window.record(f64::NAN);
        window.record(-5.0);
        window.record(f64::INFINITY);
        assert_eq!(window.average_ms(), 0.0);
        window.record(10.0);
        assert_eq!(window.average_ms(), 10.0);
    }

    #[test]
    fn evicts_oldest_beyond_capacity() {
        let mut window = LatencyWindow::default();
        for _ in 0..LATENCY_WINDOW_CAPACITY {
            window.record(1.0);
        }
        // Newer, larger samples push out the old 1.0 values.
        for _ in 0..LATENCY_WINDOW_CAPACITY {
            window.record(9.0);
        }
        assert_eq!(window.average_ms(), 9.0);
    }
}

#[cfg(test)]
mod vram_clamp_tests {
    use super::aggregate_gpu_samples;

    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: usize = 1024 * 1024 * 1024;

    // One GPU reporting 4 GiB used of a 24 GiB physical card.
    const ONE_GPU_CSV: &str = "30, 4096, 24576";

    #[test]
    fn no_cap_reports_physical_total() {
        let (util, used, total) = aggregate_gpu_samples(ONE_GPU_CSV, |_| None).unwrap();
        assert!((util - 0.30).abs() < 1e-9);
        assert_eq!(used, (4096.0 * MIB) as usize);
        // Physical 24 GiB is reported unchanged when no cap is configured.
        assert_eq!(total, (24576.0 * MIB) as usize);
    }

    #[test]
    fn cap_clamps_reported_total_to_the_slice() {
        // An 8 GiB software-vGPU slice on the same 24 GiB card.
        let (_, _, total) = aggregate_gpu_samples(ONE_GPU_CSV, |_| Some(8 * GIB)).unwrap();
        assert_eq!(total, 8 * GIB);
    }

    #[test]
    fn cap_larger_than_card_never_inflates_total() {
        // A cap above the physical card is ignored (clamp is a min).
        let (_, _, total) = aggregate_gpu_samples(ONE_GPU_CSV, |_| Some(64 * GIB)).unwrap();
        assert_eq!(total, (24576.0 * MIB) as usize);
    }

    #[test]
    fn per_device_caps_are_applied_by_index() {
        // Two GPUs; only device 1 is capped to 8 GiB.
        let csv = "10, 2048, 24576\n20, 2048, 24576";
        let (_, _, total) = aggregate_gpu_samples(csv, |id| (id == 1).then_some(8 * GIB)).unwrap();
        // device 0 physical (24 GiB) + device 1 capped (8 GiB).
        assert_eq!(total, (24576.0 * MIB) as usize + 8 * GIB);
    }
}

#[cfg(test)]
mod foreign_vram_tests {
    use super::{aggregate_foreign_vram, compute_apps_show_mps, parse_uuid_index};
    use std::collections::HashMap;

    const MIB: usize = 1024 * 1024;

    fn uuid_index(pairs: &[(&str, usize)]) -> HashMap<String, usize> {
        pairs.iter().map(|(u, i)| ((*u).to_string(), *i)).collect()
    }

    #[test]
    fn parses_uuid_index_csv() {
        let csv = "GPU-aaa, 0\nGPU-bbb, 1";
        let map = parse_uuid_index(csv);
        assert_eq!(map.get("GPU-aaa").copied(), Some(0));
        assert_eq!(map.get("GPU-bbb").copied(), Some(1));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn sums_foreign_and_excludes_own_pid() {
        // Our PID is 4242. A trainer (pid 99) holds 6 GiB; our own 2 GiB row is
        // ignored so the ceiling only backs off for the neighbor's footprint.
        // Column layout is pid,process_name,used_memory,gpu_uuid.
        let csv = "99, python, 6144, GPU-aaa\n4242, kapsl, 2048, GPU-aaa";
        let foreign = aggregate_foreign_vram(csv, &uuid_index(&[("GPU-aaa", 0)]), 4242);
        assert_eq!(foreign.get(&0).copied(), Some(6144 * MIB));
    }

    #[test]
    fn attributes_by_device_and_sums_multiple_foreign_procs() {
        let csv = "10, a, 1024, GPU-aaa\n11, b, 2048, GPU-aaa\n12, c, 4096, GPU-bbb";
        let foreign =
            aggregate_foreign_vram(csv, &uuid_index(&[("GPU-aaa", 0), ("GPU-bbb", 1)]), 4242);
        assert_eq!(foreign.get(&0).copied(), Some(3072 * MIB));
        assert_eq!(foreign.get(&1).copied(), Some(4096 * MIB));
    }

    #[test]
    fn skips_rows_with_unknown_uuid_rather_than_misattributing() {
        // A GPU not in the index map (e.g. a MIG child or a device kapsl doesn't
        // manage) must not be folded into device 0.
        let csv = "10, trainer, 4096, GPU-unknown";
        let foreign = aggregate_foreign_vram(csv, &uuid_index(&[("GPU-aaa", 0)]), 4242);
        assert!(foreign.is_empty());
    }

    #[test]
    fn empty_when_only_our_pid_present() {
        let csv = "4242, kapsl, 8192, GPU-aaa";
        let foreign = aggregate_foreign_vram(csv, &uuid_index(&[("GPU-aaa", 0)]), 4242);
        assert!(foreign.is_empty());
    }

    #[test]
    fn detects_mps_server_row_and_ignores_plain_processes() {
        let plain = "99, python, 6144, GPU-aaa\n4242, kapsl, 2048, GPU-aaa";
        assert!(!compute_apps_show_mps(plain));

        let with_mps = "99, python, 6144, GPU-aaa\n77, nvidia-cuda-mps-server, 10240, GPU-aaa";
        assert!(compute_apps_show_mps(with_mps));
    }
}

#[cfg(test)]
mod cotenancy_exporter_tests {
    use super::CotenancyCeilingExporter;
    use crate::runtime::memory::CeilingSample;

    const GIB: usize = 1024 * 1024 * 1024;

    fn gauge_value(registry: &prometheus::Registry, name: &str, device: &str) -> Option<i64> {
        registry
            .gather()
            .iter()
            .find(|family| family.name() == name)?
            .get_metric()
            .iter()
            .find(|metric| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == "device" && label.value() == device)
            })
            .map(|metric| metric.get_gauge().value() as i64)
    }

    #[test]
    fn exports_ceiling_gauges_per_device_and_tracks_recovery() {
        let registry = prometheus::Registry::new();
        let mut exporter = CotenancyCeilingExporter::new(&registry);

        // A 12 GiB trainer squeezes device 0 to a 24 GiB ceiling.
        exporter.observe(&[CeilingSample {
            device_id: 0,
            foreign_bytes: 12 * GIB,
            target_bytes: 24 * GIB,
            smoothed_bytes: 24 * GIB,
            previous_bytes: 40 * GIB,
            squeezed: true,
        }]);
        let ceiling = |name: &str| gauge_value(&registry, name, "0");
        assert_eq!(
            ceiling("kapsl_cotenancy_kv_ceiling_bytes"),
            Some((24 * GIB) as i64)
        );
        assert_eq!(
            ceiling("kapsl_cotenancy_kv_ceiling_target_bytes"),
            Some((24 * GIB) as i64)
        );
        assert_eq!(
            ceiling("kapsl_cotenancy_foreign_vram_bytes"),
            Some((12 * GIB) as i64)
        );
        assert_eq!(ceiling("kapsl_cotenancy_squeeze_active"), Some(1));

        // Trainer exits: the smoothed gauge lags the target (grow-slow), which
        // is exactly the gap a dashboard should show, and the squeeze clears.
        exporter.observe(&[CeilingSample {
            device_id: 0,
            foreign_bytes: 0,
            target_bytes: 36 * GIB,
            smoothed_bytes: 27 * GIB,
            previous_bytes: 24 * GIB,
            squeezed: false,
        }]);
        assert_eq!(
            ceiling("kapsl_cotenancy_kv_ceiling_bytes"),
            Some((27 * GIB) as i64)
        );
        assert_eq!(
            ceiling("kapsl_cotenancy_kv_ceiling_target_bytes"),
            Some((36 * GIB) as i64)
        );
        assert_eq!(ceiling("kapsl_cotenancy_foreign_vram_bytes"), Some(0));
        assert_eq!(ceiling("kapsl_cotenancy_squeeze_active"), Some(0));
    }
}
