//! Isolated worker process construction and remote engine transport.

use super::*;
use kapsl_backends::OnnxRuntimeTuning;
#[cfg(unix)]
use kapsl_transport::protocol::{
    blocking as wire, CodecError, StreamResponse, DEFAULT_MAX_FRAME_PAYLOAD_BYTES, OP_INFER_STREAM,
};

fn is_explicit_worker_gpu_boundary(name: &str, value: &str) -> bool {
    let is_positive_mb = || value.trim().parse::<usize>().is_ok_and(|mb| mb > 0);
    let is_cuda_cap = || parse_cuda_memory_limit(value).is_some();

    if name == ISOLATED_WORKER_GPU_POOL_ENV {
        return matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        );
    }
    if name == KAPSL_GPU_MEMORY_LIMIT_MB_ENV {
        return is_positive_mb();
    }
    if name == CUDA_DEVICE_MEMORY_LIMIT_ENV {
        return is_cuda_cap();
    }
    name.strip_prefix(&format!("{CUDA_DEVICE_MEMORY_LIMIT_ENV}_"))
        .is_some_and(|device_id| !device_id.is_empty() && is_cuda_cap())
}

fn isolated_worker_gpu_pool_allowed() -> bool {
    std::env::vars().any(|(name, value)| is_explicit_worker_gpu_boundary(&name, &value))
}

/// Everything needed to (re)spawn an isolated worker child for a model, so the
/// supervisor can restart a dead worker without re-deriving arguments.
#[derive(Clone)]
pub(crate) struct WorkerSpec {
    model_id: u32,
    model_path: PathBuf,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: String,
    tp_degree: usize,
    onnx_tuning: Option<OnnxRuntimeTuning>,
    offline: bool,
}

pub(crate) struct WorkerProcess {
    socket_path: String,
    child: Mutex<Child>,
    /// Spec used to respawn this worker on death.
    spec: WorkerSpec,
    /// Set when the worker is intentionally torn down (kill/Drop) so the
    /// supervisor stops and never resurrects a deliberately-stopped worker.
    shutdown: AtomicBool,
    /// Number of automatic restarts performed (bounds the restart budget).
    restarts: AtomicU32,
}

impl WorkerProcess {
    pub(crate) fn try_wait(&self) -> Option<std::process::ExitStatus> {
        self.child.lock().try_wait().ok().flatten()
    }

    pub(crate) fn kill(&self) {
        // Mark shutdown first so the supervisor won't try to restart it.
        self.shutdown.store(true, Ordering::Relaxed);
        let mut child = self.child.lock();
        if let Ok(None) = child.try_wait() {
            let _ = child.kill();
        }
    }

    /// Respawn the child process bound to the same socket path. The previous
    /// (dead) child handle is replaced. Unix-only; no-op error elsewhere.
    #[cfg(unix)]
    pub(crate) fn restart_child(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if Path::new(&self.socket_path).exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
        let child = build_worker_command(&self.spec, &self.socket_path)?.spawn()?;
        *self.child.lock() = child;
        Ok(())
    }

    #[cfg(not(unix))]
    pub(crate) fn restart_child(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Err("Isolated workers are only supported on unix platforms".into())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Supervise an isolated worker: restart it (bounded retries with backoff) if it
/// dies unexpectedly, so an isolated model recovers from a crash instead of
/// staying down. Exits when the worker is intentionally shut down. Returns the
/// same `Arc` for convenience.
pub(crate) fn start_worker_with_supervisor(worker: Arc<WorkerProcess>) -> Arc<WorkerProcess> {
    const CHECK_INTERVAL: Duration = Duration::from_secs(2);
    const RESTART_BACKOFF: Duration = Duration::from_secs(2);
    const MAX_RESTARTS: u32 = 5;
    const READY_TIMEOUT: Duration = Duration::from_secs(30);

    let w = worker.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(CHECK_INTERVAL).await;
            if w.shutdown.load(Ordering::Relaxed) {
                break;
            }
            // Still alive — nothing to do.
            let Some(status) = w.try_wait() else {
                continue;
            };
            if w.restarts.load(Ordering::Relaxed) >= MAX_RESTARTS {
                log::error!(
                    "[worker-supervisor] model {} exited ({}); exceeded {} restarts, giving up",
                    w.spec.model_id,
                    status,
                    MAX_RESTARTS
                );
                break;
            }
            let attempt = w.restarts.fetch_add(1, Ordering::Relaxed) + 1;
            log::warn!(
                "[worker-supervisor] model {} exited ({}); restarting (attempt {}/{})",
                w.spec.model_id,
                status,
                attempt,
                MAX_RESTARTS
            );
            tokio::time::sleep(RESTART_BACKOFF).await;
            if w.shutdown.load(Ordering::Relaxed) {
                break;
            }
            match w.restart_child() {
                Ok(()) => match wait_for_worker_ready_async(&w, READY_TIMEOUT).await {
                    Ok(()) => log::info!(
                        "[worker-supervisor] model {} restarted successfully",
                        w.spec.model_id
                    ),
                    Err(e) => log::error!(
                        "[worker-supervisor] model {} restarted but not ready: {}",
                        w.spec.model_id,
                        e
                    ),
                },
                Err(e) => log::error!(
                    "[worker-supervisor] model {} restart spawn failed: {}",
                    w.spec.model_id,
                    e
                ),
            }
        }
    });
    worker
}

#[cfg(unix)]
pub(crate) fn socket_ready(socket_path: &str) -> bool {
    if !Path::new(socket_path).exists() {
        return false;
    }
    UnixStream::connect(socket_path).is_ok()
}

#[cfg(not(unix))]
pub(crate) fn socket_ready(_socket_path: &str) -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
/// Build the `Command` that launches an isolated worker child for `spec`, bound
/// to `socket_path`. Shared by initial spawn and supervisor restart.
#[cfg(unix)]
pub(crate) fn build_worker_command(
    spec: &WorkerSpec,
    socket_path: &str,
) -> Result<Command, Box<dyn std::error::Error + Send + Sync>> {
    let exe = std::env::current_exe()?;
    let mut command = Command::new(exe);
    command
        .arg("--worker")
        .arg("--worker-model-id")
        .arg(spec.model_id.to_string())
        .arg("--model")
        .arg(&spec.model_path)
        .arg("--socket")
        .arg(socket_path)
        .arg("--transport")
        .arg("socket")
        .arg("--batch-size")
        .arg(spec.batch_size.to_string())
        .arg("--scheduler-queue-size")
        .arg(spec.scheduler_queue_size.to_string())
        .arg("--scheduler-max-micro-batch")
        .arg(spec.scheduler_max_micro_batch.to_string())
        .arg("--scheduler-queue-delay-ms")
        .arg(spec.scheduler_queue_delay_ms.to_string())
        .arg("--topology")
        .arg(&spec.topology)
        .arg("--tp-degree")
        .arg(spec.tp_degree.to_string())
        .env(LLM_ISOLATE_PROCESS_ENV, "0");
    if spec.offline {
        command.arg("--offline");
    }
    if !isolated_worker_gpu_pool_allowed() {
        // DeviceMemoryManager still runs in the child for planned-vs-actual
        // accounting, but no process-local arena may reserve the same global
        // pool capacity as its siblings.
        command.env(GPU_DEVICE_POOL_DISABLED_ENV, "1");
    }
    if let Some(onnx_tuning) = &spec.onnx_tuning {
        if let Some(value) = onnx_tuning.memory_pattern {
            command.arg("--onnx-memory-pattern").arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.disable_cpu_mem_arena {
            command
                .arg("--onnx-disable-cpu-mem-arena")
                .arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.session_buckets {
            command.arg("--onnx-session-buckets").arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.bucket_dim_granularity {
            command
                .arg("--onnx-bucket-dim-granularity")
                .arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.bucket_max_dims {
            command.arg("--onnx-bucket-max-dims").arg(value.to_string());
        }
        if let Some(value) = onnx_tuning.peak_concurrency_hint {
            command
                .arg("--onnx-peak-concurrency-hint")
                .arg(value.to_string());
        }
    }
    Ok(command)
}

// Keep the worker command's launch settings explicit at this process boundary;
// they are immediately captured in `WorkerSpec` for supervision and restart.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_worker_process(
    model_id: u32,
    model_path: &Path,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: Option<&OnnxRuntimeTuning>,
) -> Result<WorkerProcess, Box<dyn std::error::Error + Send + Sync>> {
    let spec = WorkerSpec {
        model_id,
        model_path: model_path.to_path_buf(),
        batch_size,
        scheduler_queue_size,
        scheduler_max_micro_batch,
        scheduler_queue_delay_ms,
        topology: topology.to_string(),
        tp_degree,
        onnx_tuning: onnx_tuning.cloned(),
        offline: backend_packs_are_offline(),
    };

    #[cfg(not(unix))]
    {
        let _ = &spec;
        return Err("Isolated workers are only supported on unix platforms".into());
    }

    #[cfg(unix)]
    {
        let socket_path = format!("/tmp/kapsl-worker-{}-{}.sock", model_id, std::process::id());
        if Path::new(&socket_path).exists() {
            std::fs::remove_file(&socket_path)?;
        }
        let child = build_worker_command(&spec, &socket_path)?.spawn()?;
        Ok(WorkerProcess {
            socket_path,
            child: Mutex::new(child),
            spec,
            shutdown: AtomicBool::new(false),
            restarts: AtomicU32::new(0),
        })
    }
}

pub(crate) async fn wait_for_worker_ready_async(
    worker: &WorkerProcess,
    timeout: Duration,
) -> Result<(), EngineError> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = worker.try_wait() {
            return Err(EngineError::backend(format!(
                "Worker exited before ready: {}",
                status
            )));
        }
        if socket_ready(&worker.socket_path) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(EngineError::backend(
                "Timed out waiting for worker socket".to_string(),
            ));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

pub(crate) struct RemoteEngine {
    model_id: u32,
    socket_path: String,
    worker: Arc<WorkerProcess>,
}

fn remote_batching_policy() -> BatchingPolicy {
    BatchingPolicy::delegated().with_priority_support()
}

impl RemoteEngine {
    pub(crate) fn new(model_id: u32, worker: Arc<WorkerProcess>) -> Self {
        Self {
            model_id,
            socket_path: worker.socket_path.clone(),
            worker,
        }
    }

    #[cfg(unix)]
    pub(crate) fn connect(&self) -> Result<UnixStream, EngineError> {
        if let Some(status) = self.worker.try_wait() {
            return Err(EngineError::backend(format!(
                "Worker process exited: {}",
                status
            )));
        }
        UnixStream::connect(&self.socket_path)
            .map_err(|e| EngineError::backend(format!("IPC connect failed: {}", e)))
    }
}

#[cfg(unix)]
fn remote_codec_error(error: CodecError) -> EngineError {
    match error {
        CodecError::Remote(message) => EngineError::backend(format!("Remote error: {message}")),
        other => EngineError::backend(format!("IPC protocol failed: {other}")),
    }
}

#[cfg(unix)]
fn infer_remote_connection<S>(
    conn: &mut S,
    model_id: u32,
    request: &InferenceRequest,
) -> Result<BinaryTensorPacket, EngineError>
where
    S: Read + Write + ?Sized,
{
    wire::infer_request_over_stream(conn, model_id, request).map_err(remote_codec_error)
}

#[cfg(unix)]
fn forward_remote_stream<S>(
    conn: &mut S,
    model_id: u32,
    request: &InferenceRequest,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<BinaryTensorPacket, EngineError>>,
) -> Result<(), EngineError>
where
    S: Read + Write + ?Sized,
{
    wire::write_request_value(conn, model_id, OP_INFER_STREAM, request)
        .map_err(remote_codec_error)?;

    loop {
        match wire::read_stream_packet(conn, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
            .map_err(remote_codec_error)?
        {
            StreamResponse::Chunk(packet) => {
                if tx.send(Ok(packet)).is_err() {
                    return Ok(());
                }
            }
            StreamResponse::End => return Ok(()),
        }
    }
}

#[async_trait::async_trait]
impl Engine for RemoteEngine {
    async fn load(&mut self, _model_path: &Path) -> Result<(), EngineError> {
        Ok(())
    }

    fn infer(&self, request: &InferenceRequest) -> Result<BinaryTensorPacket, EngineError> {
        #[cfg(not(unix))]
        {
            let _ = request;
            return Err(EngineError::backend(
                "IPC isolation is only supported on unix platforms".to_string(),
            ));
        }

        #[cfg(unix)]
        {
            let mut conn = self.connect()?;
            infer_remote_connection(&mut conn, self.model_id, request)
        }
    }

    fn self_batches(&self) -> bool {
        true
    }

    fn batching_policy(&self) -> BatchingPolicy {
        remote_batching_policy()
    }

    fn infer_stream(
        &self,
        request: &InferenceRequest,
    ) -> std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<BinaryTensorPacket, EngineError>> + Send>,
    > {
        #[cfg(not(unix))]
        {
            let _ = request;
            let stream = stream::once(async {
                Err(EngineError::backend(
                    "IPC isolation is only supported on unix platforms".to_string(),
                ))
            });
            return Box::pin(stream);
        }

        #[cfg(unix)]
        {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let socket_path = self.socket_path.clone();
            let model_id = self.model_id;
            let request = request.clone();
            let worker = self.worker.clone();

            std::thread::spawn(move || {
                let mut conn = match UnixStream::connect(&socket_path) {
                    Ok(conn) => conn,
                    Err(e) => {
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "IPC connect failed: {}",
                            e
                        ))));
                        return;
                    }
                };

                if let Some(status) = worker.try_wait() {
                    let _ = tx.send(Err(EngineError::backend(format!(
                        "Worker process exited: {}",
                        status
                    ))));
                    return;
                }

                if let Err(error) = forward_remote_stream(&mut conn, model_id, &request, &tx) {
                    let _ = tx.send(Err(error));
                }
            });

            let stream = stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|item| (item, rx))
            });
            Box::pin(stream)
        }
    }

    fn unload(&mut self) {
        // Shared worker lifecycle is owned by Arc<WorkerProcess>.
    }

    fn metrics(&self) -> EngineMetrics {
        #[cfg(not(unix))]
        {
            EngineMetrics::default()
        }

        #[cfg(unix)]
        {
            let pid = self.worker.child.lock().id();
            let pid = Pid::from_u32(pid);

            let mut system = System::new();
            system.refresh_process(pid);
            let memory_usage = system
                .process(pid)
                .map(|p| p.memory() as usize)
                .unwrap_or(0);

            EngineMetrics {
                memory_usage,
                ..EngineMetrics::default()
            }
        }
    }

    fn health_check(&self) -> Result<(), EngineError> {
        #[cfg(not(unix))]
        {
            return Err(EngineError::backend(
                "IPC isolation is only supported on unix platforms".to_string(),
            ));
        }

        #[cfg(unix)]
        {
            if let Some(status) = self.worker.try_wait() {
                return Err(EngineError::backend(format!(
                    "Worker process exited: {}",
                    status
                )));
            }
            UnixStream::connect(&self.socket_path)
                .map(|_| ())
                .map_err(|e| EngineError::backend(format!("IPC health check failed: {}", e)))
        }
    }
}

#[cfg(test)]
mod remote_engine_tests {
    use super::*;
    use kapsl_engine_api::BatchingMode;
    #[cfg(unix)]
    use kapsl_engine_api::{RequestMetadata, TensorDtype};
    #[cfg(unix)]
    use kapsl_transport::protocol::{
        blocking as test_wire, DEFAULT_MAX_FRAME_PAYLOAD_BYTES, STATUS_OK, STATUS_STREAM_CHUNK,
        STATUS_STREAM_END,
    };

    #[cfg(unix)]
    fn test_packet(value: f32) -> BinaryTensorPacket {
        BinaryTensorPacket {
            shape: vec![1],
            dtype: TensorDtype::Float32,
            data: value.to_le_bytes().to_vec(),
        }
    }

    #[test]
    fn remote_engine_delegates_batching_and_forwards_priority() {
        let policy = remote_batching_policy();

        assert_eq!(policy.mode, BatchingMode::Delegated);
        assert_eq!(policy.max_requests, 1);
        assert!(policy.supports_priority);
    }

    #[test]
    fn isolated_worker_pool_requires_an_explicit_device_boundary() {
        assert!(!is_explicit_worker_gpu_boundary(
            "KAPSL_GPU_DEVICE_POOL_BYTES",
            "8g"
        ));
        assert!(is_explicit_worker_gpu_boundary(
            CUDA_DEVICE_MEMORY_LIMIT_ENV,
            "8g"
        ));
        assert!(is_explicit_worker_gpu_boundary(
            "CUDA_DEVICE_MEMORY_LIMIT_0",
            "8589934592"
        ));
        assert!(is_explicit_worker_gpu_boundary(
            KAPSL_GPU_MEMORY_LIMIT_MB_ENV,
            "8192"
        ));
        assert!(is_explicit_worker_gpu_boundary(
            ISOLATED_WORKER_GPU_POOL_ENV,
            "true"
        ));
        assert!(!is_explicit_worker_gpu_boundary(
            ISOLATED_WORKER_GPU_POOL_ENV,
            "false"
        ));
        #[cfg(feature = "gpu-device-pool")]
        {
            assert!(!is_explicit_worker_gpu_boundary(
                GPU_DEVICE_POOL_MODE_ENV,
                "auto"
            ));
            assert!(!is_explicit_worker_gpu_boundary(
                GPU_DEVICE_POOL_UNPOOLED_RESERVE_BYTES_ENV,
                "2g"
            ));
        }
    }

    #[cfg(unix)]
    #[test]
    fn unary_remote_connection_preserves_the_complete_request() {
        let (mut client, mut server) = UnixStream::pair().expect("create Unix stream pair");
        let server_thread = std::thread::spawn(move || {
            let frame = test_wire::read_request_frame(&mut server, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .expect("read request frame");
            assert_eq!(frame.header.model_id, 7);
            let request = frame
                .decode_inference_request()
                .expect("decode inference request");
            assert_eq!(request.session_id.as_deref(), Some("trace-session"));
            let metadata = request.metadata.expect("request metadata");
            assert_eq!(metadata.priority, Some(0));
            assert_eq!(metadata.auth_token.as_deref(), Some("secret"));

            test_wire::write_response_value(&mut server, STATUS_OK, &request.input)
                .expect("write response");
        });
        let request = InferenceRequest::new(test_packet(1.0))
            .with_session_id("trace-session")
            .with_metadata(RequestMetadata {
                priority: Some(0),
                auth_token: Some("secret".to_string()),
                ..RequestMetadata::default()
            });

        let output = infer_remote_connection(&mut client, 7, &request).expect("remote inference");
        assert_eq!(output.data, request.input.data);
        server_thread.join().expect("server thread");
    }

    #[cfg(unix)]
    #[test]
    fn streaming_remote_connection_forwards_chunks_until_end() {
        let (mut client, mut server) = UnixStream::pair().expect("create Unix stream pair");
        let server_thread = std::thread::spawn(move || {
            let frame = test_wire::read_request_frame(&mut server, DEFAULT_MAX_FRAME_PAYLOAD_BYTES)
                .expect("read stream request");
            assert_eq!(frame.header.model_id, 9);
            assert_eq!(frame.header.op_code, OP_INFER_STREAM);
            frame
                .decode_inference_request()
                .expect("decode stream request");

            test_wire::write_response_value(&mut server, STATUS_STREAM_CHUNK, &test_packet(2.0))
                .expect("write first chunk");
            test_wire::write_response_value(&mut server, STATUS_STREAM_CHUNK, &test_packet(3.0))
                .expect("write second chunk");
            test_wire::write_response_bytes(&mut server, STATUS_STREAM_END, &[])
                .expect("write stream end");
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        forward_remote_stream(
            &mut client,
            9,
            &InferenceRequest::new(test_packet(1.0)),
            &tx,
        )
        .expect("forward stream");
        drop(tx);
        let first = rx
            .blocking_recv()
            .expect("first chunk")
            .expect("first result");
        let second = rx
            .blocking_recv()
            .expect("second chunk")
            .expect("second result");
        assert_eq!(first.data, test_packet(2.0).data);
        assert_eq!(second.data, test_packet(3.0).data);
        assert!(rx.blocking_recv().is_none());
        server_thread.join().expect("server thread");
    }
}
