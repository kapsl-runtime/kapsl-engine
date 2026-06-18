use super::*;

pub(crate) struct ManagedRuntimeSpec {
    pub(crate) name: String,
    pub(crate) base_url: String,
    pub(crate) profile: RuntimeGroupProfile,
    pub(crate) auth_token: Option<String>,
    pub(crate) memory_budget_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ControlScalingPolicy {
    pub(crate) min_replicas: u32,
    pub(crate) max_replicas: u32,
    pub(crate) target_queue_depth: usize,
    pub(crate) scale_down_threshold: usize,
    pub(crate) cooldown_seconds: u64,
}

impl RuntimeGroupProfile {
    pub(crate) fn default_scaling_policy(self) -> ControlScalingPolicy {
        match self {
            Self::Latency => ControlScalingPolicy {
                min_replicas: 1,
                max_replicas: 2,
                target_queue_depth: 2,
                scale_down_threshold: 1,
                cooldown_seconds: 180,
            },
            Self::Balanced => ControlScalingPolicy {
                min_replicas: 1,
                max_replicas: 4,
                target_queue_depth: 5,
                scale_down_threshold: 2,
                cooldown_seconds: 300,
            },
            Self::Throughput => ControlScalingPolicy {
                min_replicas: 1,
                max_replicas: 6,
                target_queue_depth: 10,
                scale_down_threshold: 3,
                cooldown_seconds: 600,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ControlHealthResponse {
    pub(crate) status: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ControlModelSummary {
    pub(crate) id: u32,
    #[serde(default)]
    pub(crate) base_model_id: u32,
    #[serde(default)]
    pub(crate) queue_depth: (usize, usize),
}

#[derive(Debug, Deserialize)]
pub(crate) struct ControlSystemStatsResponse {
    pub(crate) process_memory_bytes: usize,
    #[serde(default)]
    pub(crate) gpu_utilization: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeObservation {
    pub(crate) healthy: bool,
    pub(crate) health_status: String,
    pub(crate) total_queue_depth: usize,
    pub(crate) model_count: usize,
    pub(crate) avg_queue_depth: f64,
    pub(crate) gpu_utilization: f64,
    pub(crate) process_memory_bytes: u64,
    pub(crate) memory_utilization: Option<f64>,
    pub(crate) pressure_score: f64,
    pub(crate) overload: bool,
    pub(crate) hot: bool,
    pub(crate) model_ids: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct RuntimeControlState {
    pub(crate) weight: f64,
    pub(crate) unhealthy_until: Option<Instant>,
    pub(crate) overload_duration: Duration,
    pub(crate) hot_duration: Duration,
    pub(crate) last_error: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ControlRuntimeSnapshot {
    pub(crate) name: String,
    pub(crate) base_url: String,
    pub(crate) profile: RuntimeGroupProfile,
    pub(crate) weight: f64,
    pub(crate) eligible: bool,
    pub(crate) healthy: bool,
    pub(crate) cooling_down: bool,
    pub(crate) pressure_score: Option<f64>,
    pub(crate) avg_queue_depth: Option<f64>,
    pub(crate) total_queue_depth: Option<usize>,
    pub(crate) model_count: Option<usize>,
    pub(crate) gpu_utilization: Option<f64>,
    pub(crate) memory_utilization: Option<f64>,
    pub(crate) process_memory_bytes: Option<u64>,
    pub(crate) message: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ControlWeightsFile {
    pub(crate) generated_at_ms: u64,
    pub(crate) interval_seconds: u64,
    pub(crate) runtimes: Vec<ControlRuntimeSnapshot>,
}

pub(crate) fn parse_named_pair(raw: &str, flag_name: &str) -> Result<(String, String), DynError> {
    let Some((name, value)) = raw.split_once('=') else {
        return Err(dyn_error_from_message(format!(
            "{} expects NAME=VALUE, received '{}'",
            flag_name, raw
        )));
    };
    let name = name.trim();
    let value = value.trim();
    if name.is_empty() || value.is_empty() {
        return Err(dyn_error_from_message(format!(
            "{} expects non-empty NAME and VALUE, received '{}'",
            flag_name, raw
        )));
    }
    Ok((name.to_string(), value.to_string()))
}

pub(crate) fn parse_named_overrides(
    values: &[String],
    flag_name: &str,
) -> Result<HashMap<String, String>, DynError> {
    let mut map = HashMap::with_capacity(values.len());
    for raw in values {
        let (name, value) = parse_named_pair(raw, flag_name)?;
        map.insert(name, value);
    }
    Ok(map)
}

pub(crate) fn parse_runtime_group_profile(raw: &str) -> Result<RuntimeGroupProfile, DynError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "latency" => Ok(RuntimeGroupProfile::Latency),
        "balanced" => Ok(RuntimeGroupProfile::Balanced),
        "throughput" => Ok(RuntimeGroupProfile::Throughput),
        other => Err(dyn_error_from_message(format!(
            "Invalid runtime profile '{}'. Expected one of: latency, balanced, throughput",
            other
        ))),
    }
}

pub(crate) fn format_authorization_header(token: Option<&str>) -> Option<String> {
    let raw = token?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some((scheme, _)) = raw.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            return Some(raw.to_string());
        }
    }
    Some(format!("Bearer {}", raw))
}

pub(crate) fn parse_control_runtime_specs(
    args: &ControlCommandArgs,
) -> Result<Vec<ManagedRuntimeSpec>, DynError> {
    let profile_overrides = parse_named_overrides(&args.runtime_profiles, "--runtime-profile")?;
    let token_overrides = parse_named_overrides(&args.runtime_tokens, "--runtime-token")?;
    let memory_overrides =
        parse_named_overrides(&args.memory_budget_bytes, "--memory-budget-bytes")?;
    let shared_token = format_authorization_header(args.auth_token.as_deref());

    let mut seen = HashSet::new();
    let mut specs = Vec::with_capacity(args.runtimes.len());

    for raw in &args.runtimes {
        let (name, base_url_raw) = parse_named_pair(raw, "--runtime")?;
        if !seen.insert(name.clone()) {
            return Err(dyn_error_from_message(format!(
                "Duplicate runtime name '{}' in --runtime list",
                name
            )));
        }

        if !base_url_raw.starts_with("http://") && !base_url_raw.starts_with("https://") {
            return Err(dyn_error_from_message(format!(
                "Runtime '{}' URL must start with http:// or https:// (got '{}')",
                name, base_url_raw
            )));
        }
        let base_url = base_url_raw.trim_end_matches('/').to_string();

        let profile = match profile_overrides.get(&name) {
            Some(raw_profile) => parse_runtime_group_profile(raw_profile)?,
            None => RuntimeGroupProfile::Balanced,
        };

        let auth_token = match token_overrides.get(&name) {
            Some(raw_token) => format_authorization_header(Some(raw_token)),
            None => shared_token.clone(),
        };

        let memory_budget_bytes = match memory_overrides.get(&name) {
            Some(raw_budget) => Some(raw_budget.parse::<u64>().map_err(|error| {
                dyn_error_from_message(format!(
                    "Invalid --memory-budget-bytes for '{}': {}",
                    name, error
                ))
            })?),
            None => None,
        };

        specs.push(ManagedRuntimeSpec {
            name,
            base_url,
            profile,
            auth_token,
            memory_budget_bytes,
        });
    }

    Ok(specs)
}

pub(crate) fn control_http_get_json<T: DeserializeOwned>(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
) -> Result<T, String> {
    let mut request = agent.get(url);
    if let Some(auth_header) = format_authorization_header(token) {
        request = request.header("Authorization", &auth_header);
    }

    let mut response = request
        .call()
        .map_err(|error| format!("GET {} failed: {}", url, format_remote_http_error(error)))?;
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("Failed to read response body from {}: {}", url, error))?;
    serde_json::from_str::<T>(&body)
        .map_err(|error| format!("Failed to decode JSON response from {}: {}", url, error))
}

pub(crate) fn control_http_post_json(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    payload: &serde_json::Value,
) -> Result<(), String> {
    let mut request = agent.post(url).header("Content-Type", "application/json");
    if let Some(auth_header) = format_authorization_header(token) {
        request = request.header("Authorization", &auth_header);
    }

    let body = serde_json::to_string(payload)
        .map_err(|error| format!("Failed to serialize control payload for {}: {}", url, error))?;
    request
        .send(body)
        .map_err(|error| format!("POST {} failed: {}", url, format_remote_http_error(error)))?;
    Ok(())
}

pub(crate) fn collect_runtime_observation(
    agent: &ureq::Agent,
    runtime: &ManagedRuntimeSpec,
    queue_target: usize,
    hot_gpu_threshold: f64,
    hot_memory_threshold: f64,
) -> Result<RuntimeObservation, String> {
    let health_url = format!("{}/api/health", runtime.base_url);
    let models_url = format!("{}/api/models", runtime.base_url);
    let stats_url = format!("{}/api/system/stats", runtime.base_url);

    let health: ControlHealthResponse =
        control_http_get_json(agent, &health_url, runtime.auth_token.as_deref())?;
    let models: Vec<ControlModelSummary> =
        control_http_get_json(agent, &models_url, runtime.auth_token.as_deref())?;
    let stats: ControlSystemStatsResponse =
        control_http_get_json(agent, &stats_url, runtime.auth_token.as_deref())?;

    let total_queue_depth = models
        .iter()
        .map(|model| model.queue_depth.0.saturating_add(model.queue_depth.1))
        .sum::<usize>();
    let model_count = models.len();
    let avg_queue_depth = if model_count > 0 {
        total_queue_depth as f64 / model_count as f64
    } else {
        0.0
    };
    let queue_norm = if queue_target == 0 {
        0.0
    } else {
        (avg_queue_depth / queue_target as f64).clamp(0.0, 1.0)
    };
    let gpu_utilization = stats.gpu_utilization.clamp(0.0, 1.0);
    let memory_utilization = runtime.memory_budget_bytes.and_then(|budget| {
        if budget == 0 {
            None
        } else {
            Some((stats.process_memory_bytes as f64 / budget as f64).clamp(0.0, 1.0))
        }
    });
    let pressure_score =
        (0.5 * queue_norm) + (0.3 * gpu_utilization) + (0.2 * memory_utilization.unwrap_or(0.0));

    let overload = avg_queue_depth > (queue_target as f64 * 2.0);
    let hot = gpu_utilization >= hot_gpu_threshold
        || memory_utilization.is_some_and(|ratio| ratio >= hot_memory_threshold);

    let mut model_ids: Vec<u32> = models
        .iter()
        .map(|model| {
            if model.base_model_id > 0 {
                model.base_model_id
            } else {
                model.id
            }
        })
        .collect();
    model_ids.sort_unstable();
    model_ids.dedup();

    Ok(RuntimeObservation {
        healthy: health.status.eq_ignore_ascii_case("healthy"),
        health_status: health.status,
        total_queue_depth,
        model_count,
        avg_queue_depth,
        gpu_utilization,
        process_memory_bytes: stats.process_memory_bytes as u64,
        memory_utilization,
        pressure_score,
        overload,
        hot,
        model_ids,
    })
}

pub(crate) fn normalize_control_weights(
    runtimes: &[ManagedRuntimeSpec],
    states: &mut HashMap<String, RuntimeControlState>,
    observations: &HashMap<String, RuntimeObservation>,
    now: Instant,
    weight_floor: f64,
) {
    let mut eligible_names = Vec::new();
    for runtime in runtimes {
        let state = states
            .entry(runtime.name.clone())
            .or_insert_with(RuntimeControlState::default);
        let cooling_down = state.unhealthy_until.is_some_and(|until| until > now);
        let eligible = observations
            .get(&runtime.name)
            .is_some_and(|observation| observation.healthy)
            && !cooling_down;
        if eligible {
            state.weight = state.weight.max(weight_floor);
            eligible_names.push(runtime.name.clone());
        } else {
            state.weight = 0.0;
        }
    }

    if eligible_names.is_empty() {
        return;
    }

    let total_weight = eligible_names
        .iter()
        .filter_map(|name| states.get(name).map(|state| state.weight))
        .sum::<f64>();

    if total_weight <= f64::EPSILON {
        let equal_weight = 1.0 / eligible_names.len() as f64;
        for name in eligible_names {
            if let Some(state) = states.get_mut(&name) {
                state.weight = equal_weight;
            }
        }
        return;
    }

    for name in eligible_names {
        if let Some(state) = states.get_mut(&name) {
            state.weight /= total_weight;
        }
    }
}

pub(crate) fn apply_scaling_policy_to_runtime_models(
    agent: &ureq::Agent,
    runtime: &ManagedRuntimeSpec,
    model_ids: &[u32],
    policy: ControlScalingPolicy,
    applied: &mut HashMap<(String, u32), ControlScalingPolicy>,
) -> Result<usize, String> {
    let payload = serde_json::to_value(policy)
        .map_err(|error| format!("Failed to encode scaling payload: {}", error))?;
    let mut updated = 0usize;

    for model_id in model_ids {
        let key = (runtime.name.clone(), *model_id);
        if applied
            .get(&key)
            .is_some_and(|previous| *previous == policy)
        {
            continue;
        }

        let url = format!("{}/api/models/{}/scaling", runtime.base_url, model_id);
        control_http_post_json(agent, &url, runtime.auth_token.as_deref(), &payload)?;
        applied.insert(key, policy);
        updated = updated.saturating_add(1);
    }

    Ok(updated)
}

pub(crate) fn persist_control_weights(
    path: &Path,
    payload: &ControlWeightsFile,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "Failed to create control weights directory {}: {}",
                    parent.display(),
                    error
                )
            })?;
        }
    }
    let serialized = serde_json::to_string_pretty(payload)
        .map_err(|error| format!("Failed to serialize control weights snapshot: {}", error))?;
    fs::write(path, serialized).map_err(|error| {
        format!(
            "Failed to write control weights snapshot {}: {}",
            path.display(),
            error
        )
    })
}

pub(crate) fn execute_control_command(args: ControlCommandArgs) -> Result<(), DynError> {
    if args.queue_target == 0 {
        return Err(dyn_error_from_message(
            "--queue-target must be greater than 0",
        ));
    }

    let runtimes = parse_control_runtime_specs(&args)?;
    if runtimes.is_empty() {
        return Err(dyn_error_from_message(
            "At least one --runtime NAME=URL must be provided",
        ));
    }

    let _ = env_logger::try_init();

    let interval = Duration::from_secs(args.interval_seconds.max(1));
    let timeout = Duration::from_millis(args.timeout_ms.max(1));
    let unhealthy_hold = Duration::from_secs(args.unhealthy_hold_seconds.max(1));
    let overload_window = Duration::from_secs(args.overload_window_seconds.max(1));
    let hot_window = Duration::from_secs(args.hot_window_seconds.max(1));
    let high_pressure = args.high_pressure_score.clamp(0.0, 1.0);
    let low_pressure = args.low_pressure_score.clamp(0.0, high_pressure);
    let hot_gpu = args.hot_gpu_utilization.clamp(0.0, 1.0);
    let hot_memory = args.hot_memory_utilization.clamp(0.0, 1.0);
    let weight_step = args.weight_step.clamp(0.0, 1.0);
    let weight_floor = args.weight_floor.clamp(0.0, 1.0);
    let overload_shift = args.overload_shift_fraction.clamp(0.0, 1.0);
    let control_agent_config = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .timeout_per_call(Some(timeout))
        .build();
    let control_agent: ureq::Agent = control_agent_config.into();

    let mut states = HashMap::new();
    let initial_weight = 1.0 / runtimes.len() as f64;
    for runtime in &runtimes {
        states.insert(
            runtime.name.clone(),
            RuntimeControlState {
                weight: initial_weight,
                ..RuntimeControlState::default()
            },
        );
    }
    let mut applied_scaling_policies: HashMap<(String, u32), ControlScalingPolicy> = HashMap::new();

    log::info!(
        "Control loop started (runtimes={}, interval={}s, dry_run={}, weights_file={})",
        runtimes.len(),
        interval.as_secs(),
        args.dry_run,
        args.weights_file.display()
    );
    for runtime in &runtimes {
        log::info!(
            "  - {} => {} profile={} memory_budget_bytes={:?}",
            runtime.name,
            runtime.base_url,
            match runtime.profile {
                RuntimeGroupProfile::Latency => "latency",
                RuntimeGroupProfile::Balanced => "balanced",
                RuntimeGroupProfile::Throughput => "throughput",
            },
            runtime.memory_budget_bytes
        );
    }

    loop {
        let cycle_started = Instant::now();
        let now = Instant::now();
        let mut observations: HashMap<String, RuntimeObservation> = HashMap::new();
        let mut poll_errors: HashMap<String, String> = HashMap::new();

        for runtime in &runtimes {
            match collect_runtime_observation(
                &control_agent,
                runtime,
                args.queue_target,
                hot_gpu,
                hot_memory,
            ) {
                Ok(observation) => {
                    observations.insert(runtime.name.clone(), observation);
                }
                Err(error) => {
                    poll_errors.insert(runtime.name.clone(), error);
                }
            }
        }

        for runtime in &runtimes {
            let state = states
                .entry(runtime.name.clone())
                .or_insert_with(RuntimeControlState::default);

            if let Some(error) = poll_errors.get(&runtime.name) {
                state.unhealthy_until = Some(now + unhealthy_hold);
                state.overload_duration = Duration::from_secs(0);
                state.hot_duration = Duration::from_secs(0);
                state.last_error = Some(error.clone());
                state.weight = 0.0;
                continue;
            }

            let Some(observation) = observations.get(&runtime.name) else {
                state.unhealthy_until = Some(now + unhealthy_hold);
                state.overload_duration = Duration::from_secs(0);
                state.hot_duration = Duration::from_secs(0);
                state.last_error = Some("Observation missing after poll".to_string());
                state.weight = 0.0;
                continue;
            };

            if !observation.healthy {
                state.unhealthy_until = Some(now + unhealthy_hold);
                state.overload_duration = Duration::from_secs(0);
                state.hot_duration = Duration::from_secs(0);
                state.last_error =
                    Some(format!("Health status is '{}'", observation.health_status));
                state.weight = 0.0;
                continue;
            }

            if state.unhealthy_until.is_some_and(|until| until > now) {
                state.weight = 0.0;
                continue;
            }

            state.unhealthy_until = None;
            state.last_error = None;

            if observation.overload {
                state.overload_duration = state.overload_duration.saturating_add(interval);
            } else {
                state.overload_duration = Duration::from_secs(0);
            }
            if observation.hot {
                state.hot_duration = state.hot_duration.saturating_add(interval);
            } else {
                state.hot_duration = Duration::from_secs(0);
            }

            if observation.pressure_score > high_pressure {
                state.weight = (state.weight - weight_step).max(weight_floor);
            } else if observation.pressure_score < low_pressure {
                state.weight += weight_step;
            }

            if state.overload_duration >= overload_window || state.hot_duration >= hot_window {
                state.weight = (state.weight - overload_shift).max(weight_floor);
            }

            if !args.dry_run {
                let desired_policy = runtime.profile.default_scaling_policy();
                if let Err(error) = apply_scaling_policy_to_runtime_models(
                    &control_agent,
                    runtime,
                    &observation.model_ids,
                    desired_policy,
                    &mut applied_scaling_policies,
                ) {
                    state.last_error = Some(error);
                }
            }
        }

        normalize_control_weights(&runtimes, &mut states, &observations, now, weight_floor);

        let snapshots: Vec<ControlRuntimeSnapshot> = runtimes
            .iter()
            .map(|runtime| {
                let state = states
                    .get(&runtime.name)
                    .cloned()
                    .unwrap_or_else(RuntimeControlState::default);
                let observation = observations.get(&runtime.name);
                let cooling_down = state.unhealthy_until.is_some_and(|until| until > now);
                let eligible = observation.is_some_and(|obs| obs.healthy) && !cooling_down;
                ControlRuntimeSnapshot {
                    name: runtime.name.clone(),
                    base_url: runtime.base_url.clone(),
                    profile: runtime.profile,
                    weight: state.weight,
                    eligible,
                    healthy: observation.is_some_and(|obs| obs.healthy),
                    cooling_down,
                    pressure_score: observation.map(|obs| obs.pressure_score),
                    avg_queue_depth: observation.map(|obs| obs.avg_queue_depth),
                    total_queue_depth: observation.map(|obs| obs.total_queue_depth),
                    model_count: observation.map(|obs| obs.model_count),
                    gpu_utilization: observation.map(|obs| obs.gpu_utilization),
                    memory_utilization: observation.and_then(|obs| obs.memory_utilization),
                    process_memory_bytes: observation.map(|obs| obs.process_memory_bytes),
                    message: state.last_error.clone(),
                }
            })
            .collect();

        let output = ControlWeightsFile {
            generated_at_ms: now_unix_seconds().saturating_mul(1000),
            interval_seconds: interval.as_secs(),
            runtimes: snapshots,
        };
        if let Err(error) = persist_control_weights(&args.weights_file, &output) {
            log::warn!("{}", error);
        }

        let summary = output
            .runtimes
            .iter()
            .map(|runtime| {
                let score = runtime
                    .pressure_score
                    .map(|value| format!("{:.2}", value))
                    .unwrap_or_else(|| "-".to_string());
                format!(
                    "{}:w={:.2},score={},healthy={},cooldown={}",
                    runtime.name, runtime.weight, score, runtime.healthy, runtime.cooling_down
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        log::info!("Control tick {}", summary);

        let elapsed = cycle_started.elapsed();
        if elapsed < interval {
            std::thread::sleep(interval - elapsed);
        }
    }
}
