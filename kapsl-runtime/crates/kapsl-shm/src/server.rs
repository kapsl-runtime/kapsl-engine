use crate::allocator::{
    ShmAllocatorSnapshot, ShmClassBudget, ShmPoolAllocator, TieredShmAllocator,
};
use crate::memory::{ShmManager, TensorHeader};
use crate::ring_buffer::LockFreeRingBuffer;
use async_trait::async_trait;
use kapsl_engine_api::{BinaryTensorPacket, EngineModelInfo, InferenceRequest, TensorDtype};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use kapsl_transport::{RequestMetadata, ResponseMetadata, TransportError, TransportServer};
use prometheus::{IntCounter, IntGauge, Opts, Registry};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::time::{sleep, Duration};

const DEFAULT_TENSOR_SLOT_CLASS_BUDGETS: [ShmClassBudget; 5] = [
    ShmClassBudget {
        slot_size: 256 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 1024 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 4 * 1024 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 16 * 1024 * 1024,
        weight: 1,
    },
    ShmClassBudget {
        slot_size: 64 * 1024 * 1024,
        weight: 1,
    },
];
const DEFAULT_TENSOR_SLOT_LEASE_TTL_SECS: u64 = 30;
const ERROR_LEN_PREFIX_BYTES: usize = std::mem::size_of::<u64>();
const SHM_METRICS_REFRESH_SECS: u64 = 1;
const MODEL_AWARE_MAX_CLASSES: usize = 6;
const MODEL_AWARE_MIN_SLOT_BYTES: usize = 64 * 1024;
const MODEL_AWARE_MAX_SLOT_BYTES: usize = 64 * 1024 * 1024;
const MODEL_AWARE_DYNAMIC_BATCH_FALLBACK: usize = 1;
const MODEL_AWARE_DYNAMIC_DIM_FALLBACK: usize = 256;
const MODEL_AWARE_MAX_ESTIMATED_TENSOR_BYTES: usize = 128 * 1024 * 1024;
const MODEL_AWARE_MAX_MODEL_WEIGHT: u32 = 1_000;

/// Request entry in the shared memory queue
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShmRequest {
    metadata: RequestMetadata,
    tensor_offset: u64,
    tensor_size: u64,
}

/// Response entry in the shared memory queue
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShmResponse {
    metadata: ResponseMetadata,
    result_offset: u64,
    result_size: u64,
    error_offset: u64, // 0 if no error
}

#[derive(Clone)]
struct ShmPoolMetrics {
    pool_in_use: IntGauge,
    pool_exhausted_total: IntCounter,
    pool_oldest_lease_ms: IntGauge,
}

impl ShmPoolMetrics {
    fn register(registry: &Arc<Registry>) -> Result<Self, prometheus::Error> {
        let pool_in_use = IntGauge::with_opts(Opts::new(
            "kapsl_shm_pool_in_use",
            "Number of currently leased SHM tensor slots",
        ))?;
        let pool_exhausted_total = IntCounter::with_opts(Opts::new(
            "kapsl_shm_pool_exhausted_total",
            "Total number of SHM tensor pool allocation failures",
        ))?;
        let pool_oldest_lease_ms = IntGauge::with_opts(Opts::new(
            "kapsl_shm_pool_oldest_lease_ms",
            "Age in milliseconds of the oldest active SHM slot lease",
        ))?;

        registry.register(Box::new(pool_in_use.clone()))?;
        registry.register(Box::new(pool_exhausted_total.clone()))?;
        registry.register(Box::new(pool_oldest_lease_ms.clone()))?;

        Ok(Self {
            pool_in_use,
            pool_exhausted_total,
            pool_oldest_lease_ms,
        })
    }

    fn update_from_snapshot(&self, snapshot: ShmAllocatorSnapshot) {
        self.pool_in_use
            .set(snapshot.in_use_slots.min(i64::MAX as usize) as i64);
        self.pool_oldest_lease_ms
            .set(snapshot.oldest_lease_ms.min(i64::MAX as u64) as i64);
    }

    fn on_exhausted(&self) {
        self.pool_exhausted_total.inc();
    }
}

fn dtype_size_bytes(dtype: Option<&str>) -> usize {
    match dtype.map(|v| v.to_ascii_lowercase()) {
        Some(v) if v == "float64" || v == "fp64" || v == "int64" || v == "i64" => 8,
        Some(v) if v == "float16" || v == "fp16" => 2,
        Some(v) if v == "uint8" || v == "u8" || v == "string" || v == "utf8" => 1,
        _ => 4, // default to fp32/i32 when unknown
    }
}

fn estimate_shape_elements(shape: &[i64]) -> Option<usize> {
    if shape.is_empty() {
        return Some(1);
    }
    let mut elements = 1usize;
    for (index, dim) in shape.iter().copied().enumerate() {
        let resolved = if dim > 0 {
            dim as usize
        } else if index == 0 {
            MODEL_AWARE_DYNAMIC_BATCH_FALLBACK
        } else {
            MODEL_AWARE_DYNAMIC_DIM_FALLBACK
        };
        elements = elements.checked_mul(resolved)?;
    }
    Some(elements)
}

fn estimate_tensor_bytes(shape: &[i64], dtype: Option<&str>) -> Option<usize> {
    let elements = estimate_shape_elements(shape)?;
    let elem_size = dtype_size_bytes(dtype);
    let payload_bytes = elements.checked_mul(elem_size)?;
    let total = payload_bytes.checked_add(std::mem::size_of::<TensorHeader>())?;
    Some(total.min(MODEL_AWARE_MAX_ESTIMATED_TENSOR_BYTES))
}

fn bucket_slot_size(bytes: usize, pool_bytes: usize) -> usize {
    let pool_cap = pool_bytes.max(1);
    let min_slot = MODEL_AWARE_MIN_SLOT_BYTES.min(pool_cap);
    let max_slot = MODEL_AWARE_MAX_SLOT_BYTES.min(pool_cap).max(min_slot);
    let clamped = bytes.clamp(min_slot, max_slot);
    clamped
        .checked_next_power_of_two()
        .unwrap_or(max_slot)
        .clamp(min_slot, max_slot)
}

fn add_model_tensor_buckets(
    shapes: &[Vec<i64>],
    dtypes: &[String],
    pool_bytes: usize,
    model_weight: u32,
    buckets: &mut HashMap<usize, u32>,
) {
    for (index, shape) in shapes.iter().enumerate() {
        let dtype = dtypes.get(index).map(String::as_str);
        if let Some(bytes) = estimate_tensor_bytes(shape, dtype) {
            let slot_size = bucket_slot_size(bytes, pool_bytes);
            let entry = buckets.entry(slot_size).or_insert(0);
            *entry = entry.saturating_add(model_weight);
        }
    }
}

fn add_model_info_buckets(
    model_info: &EngineModelInfo,
    pool_bytes: usize,
    model_weight: u32,
    buckets: &mut HashMap<usize, u32>,
) {
    add_model_tensor_buckets(
        &model_info.input_shapes,
        &model_info.input_dtypes,
        pool_bytes,
        model_weight,
        buckets,
    );
    add_model_tensor_buckets(
        &model_info.output_shapes,
        &model_info.output_dtypes,
        pool_bytes,
        model_weight,
        buckets,
    );
}

fn model_peak_weight(model_info: &EngineModelInfo) -> u32 {
    model_info
        .peak_concurrency
        .unwrap_or(1)
        .clamp(1, MODEL_AWARE_MAX_MODEL_WEIGHT)
}

fn derive_model_aware_class_budgets(
    schedulers: &HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    pool_bytes: usize,
) -> Vec<ShmClassBudget> {
    let mut buckets: HashMap<usize, u32> = HashMap::new();
    // Keep a tiny class for errors/control payloads.
    let control_slot = bucket_slot_size(ERROR_LEN_PREFIX_BYTES + 4096, pool_bytes);
    buckets.insert(control_slot, 1);

    for scheduler in schedulers.values() {
        if let Some(model_info) = scheduler.model_info() {
            let model_weight = model_peak_weight(&model_info);
            add_model_info_buckets(&model_info, pool_bytes, model_weight, &mut buckets);
        }
    }

    // Only control bucket available means no usable model metadata.
    if buckets.len() <= 1 {
        return Vec::new();
    }

    let mut weighted: Vec<(usize, u32)> = buckets.into_iter().collect();
    weighted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    weighted.truncate(MODEL_AWARE_MAX_CLASSES);
    weighted.sort_unstable_by_key(|(slot_size, _)| *slot_size);

    weighted
        .into_iter()
        .map(|(slot_size, weight)| ShmClassBudget {
            slot_size,
            weight: weight.max(1),
        })
        .collect()
}

/// Shared memory server implementing TransportServer
pub struct ShmServer {
    shm_name: String,
    shm_size: usize,
    schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    metrics_registry: Option<Arc<Registry>>,
}

impl ShmServer {
    pub fn new(
        shm_name: &str,
        shm_size: usize,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
    ) -> Self {
        Self::new_with_registry(shm_name, shm_size, schedulers, None)
    }

    pub fn new_with_registry(
        shm_name: &str,
        shm_size: usize,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
        metrics_registry: Option<Arc<Registry>>,
    ) -> Self {
        Self {
            shm_name: shm_name.to_string(),
            shm_size,
            schedulers,
            metrics_registry,
        }
    }

    /// Check if shared memory is available on this platform
    pub fn is_available() -> bool {
        cfg!(unix) || cfg!(windows)
    }

    async fn run_internal(&self) -> Result<(), TransportError> {
        // Create shared memory
        let shm = Arc::new(
            ShmManager::create(&self.shm_name, self.shm_size)
                .map_err(|e| TransportError::ServerError(e.to_string()))?,
        );

        let tensor_pool_bytes = shm.max_tensor_size();
        if tensor_pool_bytes <= std::mem::size_of::<TensorHeader>() {
            return Err(TransportError::ServerError(format!(
                "SHM tensor pool too small: {} bytes",
                tensor_pool_bytes
            )));
        }

        let model_budgets = derive_model_aware_class_budgets(&self.schedulers, tensor_pool_bytes);
        let (class_budgets, budget_source): (Vec<ShmClassBudget>, &'static str) =
            if model_budgets.is_empty() {
                (DEFAULT_TENSOR_SLOT_CLASS_BUDGETS.to_vec(), "default")
            } else {
                (model_budgets, "model-metadata")
            };

        let tensor_allocator = Arc::new(TieredShmAllocator::new_with_class_budgets(
            shm.tensor_pool_offset(),
            tensor_pool_bytes,
            &class_budgets,
            std::time::Duration::from_secs(DEFAULT_TENSOR_SLOT_LEASE_TTL_SECS),
        ));
        let schedulers = Arc::new(self.schedulers.clone());
        let shm_pool_metrics = self.metrics_registry.as_ref().and_then(|registry| {
            match ShmPoolMetrics::register(registry) {
                Ok(metrics) => Some(Arc::new(metrics)),
                Err(e) => {
                    log::warn!("Failed to register SHM pool metrics: {}", e);
                    None
                }
            }
        });
        if let Some(metrics) = shm_pool_metrics.as_ref() {
            metrics.update_from_snapshot(tensor_allocator.snapshot());
        }
        log::info!(
            "SHM tensor pool configured: base={} source={} classes=[{}] ttl_s={}",
            shm.tensor_pool_offset(),
            budget_source,
            tensor_allocator.layout_summary(),
            DEFAULT_TENSOR_SLOT_LEASE_TTL_SECS
        );

        // Initialize request and response queues
        let req_queue_offset = shm.request_queue_offset();
        let resp_queue_offset = shm.response_queue_offset();

        log::info!("Request queue offset: {}", req_queue_offset);
        log::info!("Response queue offset: {}", resp_queue_offset);

        unsafe {
            // Initialize the queues in shared memory (only once)
            LockFreeRingBuffer::<ShmRequest>::new(
                shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                1024,
            );
            LockFreeRingBuffer::<ShmResponse>::new(
                shm.as_ptr().add(resp_queue_offset) as *mut ShmResponse,
                1024,
            );
        }

        log::info!("Shared memory server running on '{}'", self.shm_name);
        log::info!("Shared memory server listening on /{}", self.shm_name);
        log::info!("Starting request polling loop...");

        let mut poll_count = 0;
        let mut last_metrics_refresh = Instant::now();
        // Main server loop
        loop {
            if let Some(metrics) = shm_pool_metrics.as_ref() {
                if last_metrics_refresh.elapsed()
                    >= std::time::Duration::from_secs(SHM_METRICS_REFRESH_SECS)
                {
                    metrics.update_from_snapshot(tensor_allocator.snapshot());
                    last_metrics_refresh = Instant::now();
                }
            }

            // Poll request queue
            let request_opt = unsafe {
                let req_queue: LockFreeRingBuffer<ShmRequest> = LockFreeRingBuffer::connect(
                    shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                    1024,
                );
                req_queue.pop()
            };

            if poll_count % 10000 == 0 {
                log::debug!(
                    "Polled {} times, request: {:?}",
                    poll_count,
                    request_opt.is_some()
                );
            }
            poll_count += 1;

            if let Some(request) = request_opt {
                log::debug!("Received SHM request: {:?}", request);

                let schedulers = schedulers.clone();
                let shm = shm.clone();
                let tensor_allocator = tensor_allocator.clone();
                let shm_pool_metrics = shm_pool_metrics.clone();

                // Spawn task to handle request
                tokio::spawn(async move {
                    let start = Instant::now();

                    // Read tensor from shared memory
                    let tensor = unsafe {
                        read_tensor_from_shm(shm.as_ptr(), request.tensor_offset as usize)
                    };

                    // Process inference
                    if let Some(scheduler) = schedulers.get(&request.metadata.model_id) {
                        let priority = if request.metadata.priority == 0 {
                            Priority::LatencyCritical
                        } else {
                            Priority::Throughput
                        };
                        let request_obj = InferenceRequest {
                            input: tensor,
                            additional_inputs: Vec::new(),
                            session_id: None, // SHM currently doesn't support session ID
                            metadata: None,
                            cancellation: None,
                        };
                        let result = scheduler
                            .infer(&request_obj, priority, request.metadata.force_cpu)
                            .await;

                        let latency_ns = start.elapsed().as_nanos() as u64;

                        match result {
                            Ok(output) => {
                                let result_size =
                                    std::mem::size_of::<TensorHeader>() + output.data.len();
                                let response = if let Some(result_offset) = allocate_pool_slot(
                                    tensor_allocator.as_ref(),
                                    result_size,
                                    shm_pool_metrics.as_deref(),
                                ) {
                                    unsafe {
                                        write_tensor_to_shm(shm.as_ptr(), result_offset, &output);
                                    }
                                    ShmResponse {
                                        metadata: ResponseMetadata::success(
                                            request.metadata.request_id,
                                            latency_ns,
                                        ),
                                        result_offset: result_offset as u64,
                                        result_size: result_size as u64,
                                        error_offset: 0,
                                    }
                                } else {
                                    let msg = format!(
                                        "SHM tensor pool exhausted (required={} bytes, largest_slot={} bytes, layout={})",
                                        result_size,
                                        tensor_allocator.largest_slot_size(),
                                        tensor_allocator.layout_summary()
                                    );
                                    log::warn!("{}", msg);
                                    let error_offset = write_error_to_shm(
                                        shm.as_ptr(),
                                        tensor_allocator.as_ref(),
                                        shm_pool_metrics.as_deref(),
                                        &msg,
                                    )
                                    .unwrap_or(0);
                                    ShmResponse {
                                        metadata: ResponseMetadata::error(
                                            request.metadata.request_id,
                                            latency_ns,
                                        ),
                                        result_offset: 0,
                                        result_size: 0,
                                        error_offset: error_offset as u64,
                                    }
                                };

                                unsafe {
                                    push_response_and_notify(shm.as_ref(), response);
                                }
                            }
                            Err(e) => {
                                let error_offset = write_error_to_shm(
                                    shm.as_ptr(),
                                    tensor_allocator.as_ref(),
                                    shm_pool_metrics.as_deref(),
                                    &e.to_string(),
                                )
                                .unwrap_or(0);

                                let response = ShmResponse {
                                    metadata: ResponseMetadata::error(
                                        request.metadata.request_id,
                                        latency_ns,
                                    ),
                                    result_offset: 0,
                                    result_size: 0,
                                    error_offset: error_offset as u64,
                                };

                                unsafe {
                                    push_response_and_notify(shm.as_ref(), response);
                                }
                            }
                        }
                    }
                });
            } else {
                // No requests, yield CPU briefly
                sleep(Duration::from_micros(10)).await;
            }
        }
    }
}

#[async_trait]
impl TransportServer for ShmServer {
    async fn run(&self) -> Result<(), TransportError> {
        self.run_internal().await
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        // Shared memory will be cleaned up when the process exits
        Ok(())
    }

    fn transport_type(&self) -> &'static str {
        "shared_memory"
    }
}

/// Read tensor from shared memory
unsafe fn read_tensor_from_shm(base: *mut u8, offset: usize) -> BinaryTensorPacket {
    let header_ptr = base.add(offset) as *const TensorHeader;
    let header = std::ptr::read(header_ptr);

    // Read shape
    let shape: Vec<i64> = header.shape[0..header.ndim as usize].to_vec();

    // Read dtype (simple mapping)
    let dtype = match header.dtype {
        0 => TensorDtype::Float32,
        1 => TensorDtype::Float64,
        2 => TensorDtype::Int32,
        3 => TensorDtype::Int64,
        _ => TensorDtype::Float32,
    };

    // Read data
    let data_ptr = base.add(offset + std::mem::size_of::<TensorHeader>());
    let data = std::slice::from_raw_parts(data_ptr, header.data_size as usize).to_vec();

    BinaryTensorPacket { shape, dtype, data }
}

/// Write tensor to shared memory
unsafe fn write_tensor_to_shm(base: *mut u8, offset: usize, tensor: &BinaryTensorPacket) {
    // Write header
    let mut shape_array = [0i64; 8];
    for (i, &s) in tensor.shape.iter().enumerate() {
        shape_array[i] = s;
    }

    let dtype_byte = match tensor.dtype {
        TensorDtype::Float32 => 0,
        TensorDtype::Float64 => 1,
        TensorDtype::Int32 => 2,
        TensorDtype::Int64 => 3,
        _ => 0,
    };

    let header = TensorHeader {
        ndim: tensor.shape.len() as u32,
        dtype: dtype_byte,
        _padding: [0; 3],
        shape: shape_array,
        data_size: tensor.data.len() as u64,
    };

    let header_ptr = base.add(offset) as *mut TensorHeader;
    std::ptr::write(header_ptr, header);

    // Write data
    let data_ptr = base.add(offset + std::mem::size_of::<TensorHeader>());
    std::ptr::copy_nonoverlapping(tensor.data.as_ptr(), data_ptr, tensor.data.len());
}

unsafe fn push_response_and_notify(shm: &ShmManager, response: ShmResponse) {
    let resp_queue: LockFreeRingBuffer<ShmResponse> = LockFreeRingBuffer::connect(
        shm.as_ptr().add(shm.response_queue_offset()) as *mut ShmResponse,
        1024,
    );
    let _ = resp_queue.push(response);

    // Notify via pipe
    let write_fd = shm.notify_write_fd();
    if write_fd >= 0 {
        let byte: u8 = 1;
        libc::write(write_fd, &byte as *const u8 as *const libc::c_void, 1);
    }
}

fn allocate_pool_slot(
    allocator: &(impl ShmPoolAllocator + ?Sized),
    required_size: usize,
    metrics: Option<&ShmPoolMetrics>,
) -> Option<usize> {
    let offset = allocator.try_allocate(required_size);
    if offset.is_none() {
        if let Some(m) = metrics {
            m.on_exhausted();
        }
    }
    if let Some(m) = metrics {
        m.update_from_snapshot(allocator.snapshot());
    }
    offset
}

/// Write error message to shared memory.
/// Layout: `[u64 error_len][error bytes]`.
fn write_error_to_shm(
    base: *mut u8,
    allocator: &(impl ShmPoolAllocator + ?Sized),
    metrics: Option<&ShmPoolMetrics>,
    error: &str,
) -> Option<usize> {
    let bytes = error.as_bytes();
    let total_size = ERROR_LEN_PREFIX_BYTES + bytes.len();
    let offset = allocate_pool_slot(allocator, total_size, metrics)?;
    unsafe {
        let len_ptr = base.add(offset) as *mut u64;
        std::ptr::write(len_ptr, bytes.len() as u64);
        let ptr = base.add(offset + ERROR_LEN_PREFIX_BYTES);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    }
    Some(offset)
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod server_tests;
