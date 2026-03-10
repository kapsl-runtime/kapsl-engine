use crate::priority::Priority;

/// Metadata for determining request priority
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    pub priority: u8,                    // 0 = high, 1+ = low (explicit override)
    pub sla_deadline: Option<u64>,       // Deadline in ms from now
    pub batch_size: usize,               // Number of items in batch
    pub input_size_bytes: Option<usize>, // Size of input tensor
    pub estimated_flops: Option<u64>,    // Estimated FLOPs required
}

impl Default for RequestMetadata {
    fn default() -> Self {
        Self {
            priority: 1,
            sla_deadline: None,
            batch_size: 1,
            input_size_bytes: None,
            estimated_flops: None,
        }
    }
}

// Threshold constants for workload-based priority determination
// These can be tuned based on your specific models and hardware
const LIGHT_TASK_SIZE_BYTES: usize = 500_000; // 500KB - small images, short text
const LIGHT_TASK_FLOPS: u64 = 1_000_000_000; // 1 GFLOP
const COMBINED_WORK_THRESHOLD: usize = 2_000_000; // 2MB total (size × batch)
const SLA_LATENCY_CRITICAL_MS: u64 = 250; // 250ms deadline
const SMALL_BATCH_SIZE: usize = 4; // Batches ≤ 4 items

/// Determines priority based on request metadata using multiple strategies:
///
/// **Priority Strategies (in order of precedence):**
/// 1. Explicit priority override (metadata.priority == 0)
/// 2. SLA deadline (< 250ms → high priority)
/// 3. Workload-based (light tasks → high priority)
/// 4. Small batch size (≤ 4 items → high priority)
///
/// **Workload Determination:**
/// - Input size: < 500KB → light
/// - Estimated FLOPs: < 1 GFLOP → light
/// - Combined: (size × batch) < 2MB → light
pub fn determine_priority(metadata: &RequestMetadata) -> Priority {
    // 1. Explicit priority override (highest precedence)
    if metadata.priority == 0 {
        return Priority::LatencyCritical;
    }

    // 2. SLA deadline - strict latency requirements
    if let Some(deadline_ms) = metadata.sla_deadline {
        if deadline_ms < SLA_LATENCY_CRITICAL_MS {
            return Priority::LatencyCritical;
        }
    }

    // 3. Workload-based priority - light tasks get high priority
    if is_light_workload(metadata) {
        return Priority::LatencyCritical;
    }

    // 4. Small batch consideration
    if metadata.batch_size <= SMALL_BATCH_SIZE {
        return Priority::LatencyCritical;
    }

    // Default: heavy workloads go to throughput queue for batching
    Priority::Throughput
}

/// Determines if the request represents a light workload
///
/// Uses multiple heuristics to classify the task:
/// - Input tensor size (e.g., small images, short text)
/// - Estimated computation (FLOPs)
/// - Combined metric (total data to process)
fn is_light_workload(metadata: &RequestMetadata) -> bool {
    // Strategy 1: Input size-based
    if let Some(size) = metadata.input_size_bytes {
        if size <= LIGHT_TASK_SIZE_BYTES {
            return true; // Small input → likely quick inference
        }
    }

    // Strategy 2: Estimated FLOPs (if available)
    if let Some(flops) = metadata.estimated_flops {
        if flops <= LIGHT_TASK_FLOPS {
            return true; // Low compute → quick task
        }
    }

    // Strategy 3: Combined heuristic (input size × batch size)
    if let Some(size) = metadata.input_size_bytes {
        let total_work = size * metadata.batch_size;
        if total_work <= COMBINED_WORK_THRESHOLD {
            return true; // Total work is manageable
        }
    }

    false
}

#[cfg(test)]
#[path = "request_metadata_tests.rs"]
mod request_metadata_tests;
