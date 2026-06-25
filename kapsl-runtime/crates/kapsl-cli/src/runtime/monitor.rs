use super::*;

#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeSamples {
    pub(crate) process_memory_bytes: usize,
    pub(crate) total_system_memory_bytes: Option<usize>,
    pub(crate) gpu_utilization: f64,
    pub(crate) gpu_memory_bytes: Option<usize>,
    pub(crate) gpu_memory_total_bytes: Option<usize>,
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

pub(crate) fn evaluate_runtime_pressure_state(
    samples: &RuntimeSamples,
    config: &RuntimePressureConfig,
) -> RuntimePressureState {
    let process_memory_ratio = samples
        .total_system_memory_bytes
        .filter(|total| *total > 0)
        .map(|total| (samples.process_memory_bytes as f64 / total as f64).clamp(0.0, 1.0));

    let gpu_util_ratio = samples.gpu_utilization.clamp(0.0, 1.0);
    let gpu_mem_ratio = match (samples.gpu_memory_bytes, samples.gpu_memory_total_bytes) {
        (Some(used), Some(total)) if total > 0 => {
            Some((used as f64 / total as f64).clamp(0.0, 1.0))
        }
        _ => None,
    };

    let emergency = process_memory_ratio
        .is_some_and(|ratio| ratio >= config.memory_emergency_ratio)
        || gpu_util_ratio >= config.gpu_util_emergency_ratio
        || gpu_mem_ratio.is_some_and(|ratio| ratio >= config.gpu_mem_emergency_ratio);
    if emergency {
        return RuntimePressureState::Emergency;
    }

    let conserve = process_memory_ratio.is_some_and(|ratio| ratio >= config.memory_conserve_ratio)
        || gpu_util_ratio >= config.gpu_util_conserve_ratio
        || gpu_mem_ratio.is_some_and(|ratio| ratio >= config.gpu_mem_conserve_ratio);
    if conserve {
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

pub(crate) type InterModelRoutes = HashMap<String, Vec<String>>;

#[derive(Debug, Clone)]
pub(crate) struct InterModelRelayState {
    pub(crate) routes: InterModelRoutes,
    pub(crate) min_interval: Duration,
    pub(crate) last_relay_at: Arc<Mutex<HashMap<u32, Instant>>>,
}

impl InterModelRelayState {
    pub(crate) fn from_env() -> Self {
        let routes_raw =
            optional_env_var_alias(INTER_MODEL_ROUTES_ENV, LEGACY_INTER_MODEL_ROUTES_ENV)
                .unwrap_or_default();
        let routes = parse_inter_model_routes(&routes_raw);
        let min_interval_ms = optional_env_var_alias(
            INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV,
            LEGACY_INTER_MODEL_RELAY_MIN_INTERVAL_MS_ENV,
        )
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_INTER_MODEL_RELAY_MIN_INTERVAL_MS)
        .max(100);

        Self {
            routes,
            min_interval: Duration::from_millis(min_interval_ms),
            last_relay_at: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn has_routes(&self) -> bool {
        !self.routes.is_empty()
    }

    pub(crate) fn targets_for(&self, source_model_name: &str) -> Option<&[String]> {
        self.routes.get(source_model_name).map(Vec::as_slice)
    }

    pub(crate) fn should_emit(&self, source_model_id: u32) -> bool {
        let now = Instant::now();
        let mut last = self.last_relay_at.lock();
        if let Some(previous) = last.get(&source_model_id).copied() {
            if now.duration_since(previous) < self.min_interval {
                return false;
            }
        }
        last.insert(source_model_id, now);
        true
    }
}

pub(crate) fn parse_inter_model_routes(raw: &str) -> InterModelRoutes {
    let mut routes = HashMap::<String, Vec<String>>::new();
    for rule in raw.split([';', '\n']) {
        let trimmed = rule.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((source_raw, target_raw)) =
            trimmed.split_once('=').or_else(|| trimmed.split_once("->"))
        else {
            continue;
        };
        let source = source_raw.trim();
        if source.is_empty() {
            continue;
        }
        for target in target_raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let entry = routes.entry(source.to_string()).or_default();
            if !entry.iter().any(|existing| existing == target) {
                entry.push(target.to_string());
            }
        }
    }
    routes
}

pub(crate) fn relay_prompt_from_output(
    source_model_name: &str,
    output: &BinaryTensorPacket,
) -> Option<String> {
    if output.dtype != TensorDtype::Utf8 {
        return None;
    }
    let text = std::str::from_utf8(&output.data).ok()?.trim();
    if text.is_empty() {
        return None;
    }
    Some(format!("Report from {}:\n{}", source_model_name, text))
}

pub(crate) fn resolve_target_base_model_id(
    model_registry: &ModelRegistry,
    target_model_name: &str,
) -> Option<u32> {
    let mut candidates = model_registry
        .list()
        .into_iter()
        .filter(|model| model.name == target_model_name && model.status == ModelStatus::Active)
        .collect::<Vec<_>>();
    candidates.sort_by_key(|model| model.id);
    candidates.first().map(|model| model.base_model_id)
}

pub(crate) fn maybe_publish_inter_model_relays(
    relay_state: &InterModelRelayState,
    source_model_id: u32,
    source_model_name: &str,
    request_is_relay: bool,
    output: &BinaryTensorPacket,
    replica_pools: &ReplicaPools,
    model_registry: &ModelRegistry,
) {
    if request_is_relay {
        return;
    }

    let Some(targets) = relay_state.targets_for(source_model_name) else {
        return;
    };
    if targets.is_empty() || !relay_state.should_emit(source_model_id) {
        return;
    }

    let Some(prompt) = relay_prompt_from_output(source_model_name, output) else {
        return;
    };

    for target_model_name in targets {
        if target_model_name == source_model_name {
            continue;
        }

        let Some(target_base_model_id) =
            resolve_target_base_model_id(model_registry, target_model_name)
        else {
            log::warn!(
                "Inter-model relay target not found: source={} target={}",
                source_model_name,
                target_model_name
            );
            continue;
        };

        let target_pool = {
            let pools = replica_pools.read();
            pools.get(&target_base_model_id).cloned()
        };
        let Some(target_pool) = target_pool else {
            log::warn!(
                "Inter-model relay pool missing: source={} target={} base_model_id={}",
                source_model_name,
                target_model_name,
                target_base_model_id
            );
            continue;
        };

        let data = prompt.clone().into_bytes();
        let relay_request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, data.len() as i64],
                dtype: TensorDtype::Utf8,
                data,
            },
            additional_inputs: Vec::new(),
            session_id: Some(format!(
                "{}{}->{}",
                INTER_MODEL_RELAY_SESSION_PREFIX, source_model_id, target_base_model_id
            )),
            metadata: Some(kapsl_engine_api::RequestMetadata {
                request_id: Some(format!(
                    "relay-{}-{}-{}",
                    source_model_id,
                    target_base_model_id,
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis()
                )),
                priority: Some(1),
                ..kapsl_engine_api::RequestMetadata::default()
            }),
            cancellation: None,
        };

        let source_model_name = source_model_name.to_string();
        let target_model_name = target_model_name.clone();
        tokio::spawn(async move {
            if let Err(error) = target_pool
                .infer(&relay_request, kapsl_scheduler::Priority::Throughput, false)
                .await
            {
                log::warn!(
                    "Inter-model relay failed: source={} target={} error={}",
                    source_model_name,
                    target_model_name,
                    error
                );
            }
        });
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
            total_mem_bytes =
                total_mem_bytes.saturating_add((mem_mb * 1024.0 * 1024.0) as usize);
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
    use super::{
        aggregate_gpu_samples, evaluate_runtime_pressure_state, RuntimePressureConfig,
        RuntimePressureState, RuntimeSamples,
    };

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

    #[test]
    fn pressure_ratio_is_computed_against_the_cap() {
        // 7 GiB used on a 24 GiB card is calm (29%) physically, but against an
        // 8 GiB slice it is 88% — the clamped total drives back-pressure into
        // emergency, which is the whole point of the cooperative clamp.
        let csv = "10, 7168, 24576";
        let config = RuntimePressureConfig {
            memory_conserve_ratio: 0.7,
            memory_emergency_ratio: 0.9,
            gpu_util_conserve_ratio: 0.8,
            gpu_util_emergency_ratio: 0.95,
            gpu_mem_conserve_ratio: 0.8,
            gpu_mem_emergency_ratio: 0.85,
            conserve_max_new_tokens: Some(256),
            emergency_max_new_tokens: Some(128),
        };

        let (_, used, physical_total) = aggregate_gpu_samples(csv, |_| None).unwrap();
        let physical = RuntimeSamples {
            process_memory_bytes: 0,
            total_system_memory_bytes: Some(64 * GIB),
            gpu_utilization: 0.1,
            gpu_memory_bytes: Some(used),
            gpu_memory_total_bytes: Some(physical_total),
            collected_at_ms: 0,
        };
        assert_eq!(
            evaluate_runtime_pressure_state(&physical, &config),
            RuntimePressureState::Normal
        );

        let (_, used, capped_total) = aggregate_gpu_samples(csv, |_| Some(8 * GIB)).unwrap();
        let capped = RuntimeSamples {
            gpu_memory_bytes: Some(used),
            gpu_memory_total_bytes: Some(capped_total),
            ..physical
        };
        assert_eq!(
            evaluate_runtime_pressure_state(&capped, &config),
            RuntimePressureState::Emergency
        );
    }
}
