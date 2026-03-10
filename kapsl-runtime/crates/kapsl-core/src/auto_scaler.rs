use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const SCALE_UP_HYSTERESIS: Duration = Duration::from_secs(20);
const MAX_SCALE_UP_STEP: u32 = 1;

/// Configuration for auto-scaling behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    /// Minimum number of replicas (default: 1)
    pub min_replicas: u32,
    /// Maximum number of replicas (default: number of GPUs)
    pub max_replicas: u32,
    /// Target average queue depth to trigger scale-up (default: 5)
    pub target_queue_depth: usize,
    /// Scale down when queue depth stays below this for cooldown period (default: 2)
    pub scale_down_threshold: usize,
    /// Wait time in seconds before scaling down to avoid thrashing (default: 300)
    pub cooldown_seconds: u64,
}

impl Default for ScalingPolicy {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 4,
            target_queue_depth: 5,
            scale_down_threshold: 2,
            cooldown_seconds: 300,
        }
    }
}

impl ScalingPolicy {
    /// Create a new scaling policy with custom values
    pub fn new(
        min_replicas: u32,
        max_replicas: u32,
        target_queue_depth: usize,
        scale_down_threshold: usize,
        cooldown_seconds: u64,
    ) -> Self {
        Self {
            min_replicas,
            max_replicas,
            target_queue_depth,
            scale_down_threshold,
            cooldown_seconds,
        }
    }

    /// Validate policy fields before accepting updates from external callers.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_replicas < 1 {
            return Err("min_replicas must be at least 1".to_string());
        }
        if self.max_replicas < self.min_replicas {
            return Err("max_replicas must be greater than or equal to min_replicas".to_string());
        }
        if self.target_queue_depth == 0 {
            return Err("target_queue_depth must be greater than 0".to_string());
        }
        if self.cooldown_seconds == 0 {
            return Err("cooldown_seconds must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Tracks scaling state for a specific model
#[derive(Debug)]
struct ModelScalingState {
    last_scale_up: Option<Instant>,
    last_scale_down: Option<Instant>,
    high_load_duration: Duration,
    low_load_duration: Duration,
}

impl Default for ModelScalingState {
    fn default() -> Self {
        Self {
            last_scale_up: None,
            last_scale_down: None,
            high_load_duration: Duration::from_secs(0),
            low_load_duration: Duration::from_secs(0),
        }
    }
}

/// Auto-scaler that makes scaling decisions based on metrics
pub struct AutoScaler {
    policies: HashMap<u32, ScalingPolicy>,
    states: HashMap<u32, ModelScalingState>,
}

impl AutoScaler {
    /// Create a new AutoScaler
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            states: HashMap::new(),
        }
    }

    /// Register a scaling policy for a specific base model
    pub fn register_policy(&mut self, base_model_id: u32, policy: ScalingPolicy) {
        self.policies.insert(base_model_id, policy);
        self.states
            .insert(base_model_id, ModelScalingState::default());
    }

    /// Get the scaling policy for a model (or default)
    pub fn get_policy(&self, base_model_id: u32) -> ScalingPolicy {
        self.policies
            .get(&base_model_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Determine if a model should scale up
    ///
    /// # Arguments
    /// * `base_model_id` - The base model ID to check
    /// * `current_replicas` - Current number of active replicas
    /// * `healthy_replicas` - Number of healthy replicas in the pool
    /// * `queue_depth` - Current total queue depth across all replicas
    /// * `elapsed_since_last_check` - Time elapsed since last scaling check
    /// * `metrics_available` - Whether queue/health metrics are available and trustworthy
    ///
    /// # Returns
    /// `Some(target_replicas)` if scaling up is needed, `None` otherwise
    pub fn should_scale_up(
        &mut self,
        base_model_id: u32,
        current_replicas: u32,
        healthy_replicas: u32,
        queue_depth: usize,
        elapsed_since_last_check: Duration,
        metrics_available: bool,
    ) -> Option<u32> {
        let policy = self.get_policy(base_model_id);
        let state = self.states.entry(base_model_id).or_default();

        if policy.target_queue_depth == 0 {
            return None;
        }

        // Fail-safe: don't scale if metrics are unavailable or pool is degraded.
        if !metrics_available || current_replicas == 0 || healthy_replicas < current_replicas {
            state.high_load_duration = Duration::from_secs(0);
            return None;
        }

        // Don't scale beyond configured max_replicas.
        if current_replicas >= policy.max_replicas {
            state.high_load_duration = Duration::from_secs(0);
            return None;
        }

        // Calculate average queue depth per replica
        let avg_queue_depth = if current_replicas > 0 {
            queue_depth as f64 / current_replicas as f64
        } else {
            queue_depth as f64
        };

        // Scale up if average queue depth exceeds target
        if avg_queue_depth > policy.target_queue_depth as f64 {
            // Anti-flap: require sustained high load before scaling.
            state.high_load_duration += elapsed_since_last_check;
            state.low_load_duration = Duration::from_secs(0);
            if state.high_load_duration < SCALE_UP_HYSTERESIS {
                return None;
            }

            // Calculate how many replicas we need
            let needed_replicas = queue_depth
                .saturating_add(policy.target_queue_depth.saturating_sub(1))
                / policy.target_queue_depth;
            let needed_replicas = needed_replicas.min(policy.max_replicas as usize) as u32;

            if needed_replicas > current_replicas {
                // Protection: scale up in bounded steps instead of jumping to full target.
                let stepped_target = current_replicas
                    .saturating_add(MAX_SCALE_UP_STEP)
                    .min(needed_replicas)
                    .min(policy.max_replicas);
                if stepped_target > current_replicas {
                    state.last_scale_up = Some(Instant::now());
                    state.high_load_duration = Duration::from_secs(0);
                    return Some(stepped_target);
                }
            }
        } else {
            state.high_load_duration = Duration::from_secs(0);
        }

        None
    }

    /// Determine if a model should scale down
    ///
    /// # Arguments
    /// * `base_model_id` - The base model ID to check
    /// * `current_replicas` - Current number of active replicas
    /// * `healthy_replicas` - Number of healthy replicas in the pool
    /// * `queue_depth` - Current total queue depth across all replicas
    /// * `elapsed_since_last_check` - Time elapsed since last scaling check
    /// * `metrics_available` - Whether queue/health metrics are available and trustworthy
    ///
    /// # Returns
    /// `Some(target_replicas)` if scaling down is needed, `None` otherwise
    pub fn should_scale_down(
        &mut self,
        base_model_id: u32,
        current_replicas: u32,
        healthy_replicas: u32,
        queue_depth: usize,
        elapsed_since_last_check: Duration,
        metrics_available: bool,
    ) -> Option<u32> {
        let policy = self.get_policy(base_model_id);
        let state = self.states.entry(base_model_id).or_default();

        // Fail-safe: don't scale when we cannot trust metrics or pool is degraded.
        if !metrics_available || healthy_replicas < current_replicas {
            state.low_load_duration = Duration::from_secs(0);
            return None;
        }

        // Don't scale below min_replicas
        if current_replicas <= policy.min_replicas {
            return None;
        }

        // Check if load is low
        if queue_depth <= policy.scale_down_threshold {
            // Accumulate low load duration
            state.low_load_duration += elapsed_since_last_check;

            // Check if we've been in low load state long enough
            let cooldown = Duration::from_secs(policy.cooldown_seconds);
            if state.low_load_duration >= cooldown {
                // Check cooldown since last scale-down
                let can_scale_down = state
                    .last_scale_down
                    .map(|last| last.elapsed() >= cooldown)
                    .unwrap_or(true);
                let recently_scaled_up = state
                    .last_scale_up
                    .map(|last| last.elapsed() < cooldown)
                    .unwrap_or(false);

                if can_scale_down && !recently_scaled_up {
                    // Scale down by 1 replica at a time
                    let target = current_replicas - 1;
                    state.last_scale_down = Some(Instant::now());
                    state.low_load_duration = Duration::from_secs(0);
                    return Some(target.max(policy.min_replicas));
                }
            }
        } else {
            // Reset low load duration if load increases
            if let Some(state) = self.states.get_mut(&base_model_id) {
                state.low_load_duration = Duration::from_secs(0);
            }
        }

        None
    }

    /// Get the next unique replica ID for a base model
    pub fn get_next_replica_id(&self, _base_model_id: u32, existing_replicas: &[u32]) -> u32 {
        // Find the maximum existing replica ID and add 1
        existing_replicas
            .iter()
            .max()
            .map(|&max| max + 1)
            .unwrap_or(1)
    }
}

impl Default for AutoScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[path = "auto_scaler_tests.rs"]
mod auto_scaler_tests;
