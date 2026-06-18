use super::*;

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
    onnx_tuning: OnnxRuntimeTuning,
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
    let onnx_tuning = &spec.onnx_tuning;
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
    Ok(command)
}

pub(crate) fn spawn_worker_process(
    model_id: u32,
    model_path: &Path,
    batch_size: usize,
    scheduler_queue_size: usize,
    scheduler_max_micro_batch: usize,
    scheduler_queue_delay_ms: u64,
    topology: &str,
    tp_degree: usize,
    onnx_tuning: &OnnxRuntimeTuning,
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
        onnx_tuning: onnx_tuning.clone(),
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

pub(crate) fn wait_for_worker_ready(
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
        std::thread::sleep(Duration::from_millis(100));
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

    #[cfg(unix)]
    pub(crate) fn read_response_header(
        &self,
        conn: &mut UnixStream,
    ) -> Result<ResponseHeader, EngineError> {
        let mut header_buf = [0u8; 8];
        conn.read_exact(&mut header_buf)
            .map_err(|e| EngineError::backend(format!("IPC read header failed: {}", e)))?;
        let status = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
        let payload_size = u32::from_le_bytes(header_buf[4..8].try_into().unwrap());
        Ok(ResponseHeader {
            status,
            payload_size,
        })
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
            let payload = bincode::serialize(request)
                .map_err(|e| EngineError::backend(format!("IPC serialize failed: {}", e)))?;

            let header = RequestHeader {
                model_id: self.model_id,
                op_code: OP_INFER,
                payload_size: payload.len() as u32,
            };

            conn.write_all(&header.model_id.to_le_bytes())
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;
            conn.write_all(&header.op_code.to_le_bytes())
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;
            conn.write_all(&header.payload_size.to_le_bytes())
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;
            conn.write_all(&payload)
                .map_err(|e| EngineError::backend(format!("IPC write failed: {}", e)))?;

            let resp = self.read_response_header(&mut conn)?;
            let mut payload = vec![0u8; resp.payload_size as usize];
            conn.read_exact(&mut payload)
                .map_err(|e| EngineError::backend(format!("IPC read failed: {}", e)))?;

            if resp.status == STATUS_OK {
                bincode::deserialize::<BinaryTensorPacket>(&payload)
                    .map_err(|e| EngineError::backend(format!("IPC decode failed: {}", e)))
            } else {
                let msg = String::from_utf8_lossy(&payload);
                Err(EngineError::backend(format!(
                    "Remote error (status {}): {}",
                    resp.status, msg
                )))
            }
        }
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

                let payload = match bincode::serialize(&request) {
                    Ok(payload) => payload,
                    Err(e) => {
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "IPC serialize failed: {}",
                            e
                        ))));
                        return;
                    }
                };

                let header = RequestHeader {
                    model_id,
                    op_code: OP_INFER_STREAM,
                    payload_size: payload.len() as u32,
                };

                if conn.write_all(&header.model_id.to_le_bytes()).is_err()
                    || conn.write_all(&header.op_code.to_le_bytes()).is_err()
                    || conn.write_all(&header.payload_size.to_le_bytes()).is_err()
                    || conn.write_all(&payload).is_err()
                {
                    let _ = tx.send(Err(EngineError::backend("IPC write failed".to_string())));
                    return;
                }

                loop {
                    let mut header_buf = [0u8; 8];
                    if conn.read_exact(&mut header_buf).is_err() {
                        let _ = tx.send(Err(EngineError::backend("IPC read failed".to_string())));
                        return;
                    }
                    let status = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
                    let payload_size = u32::from_le_bytes(header_buf[4..8].try_into().unwrap());

                    if status == STATUS_STREAM_END {
                        break;
                    }

                    let mut payload = vec![0u8; payload_size as usize];
                    if conn.read_exact(&mut payload).is_err() {
                        let _ = tx.send(Err(EngineError::backend("IPC read failed".to_string())));
                        return;
                    }

                    if status == STATUS_STREAM_CHUNK {
                        match bincode::deserialize::<BinaryTensorPacket>(&payload) {
                            Ok(packet) => {
                                if tx.send(Ok(packet)).is_err() {
                                    return;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(EngineError::backend(format!(
                                    "IPC decode failed: {}",
                                    e
                                ))));
                                return;
                            }
                        }
                    } else if status == STATUS_ERR {
                        let msg = String::from_utf8_lossy(&payload);
                        let _ =
                            tx.send(Err(EngineError::backend(format!("Remote error: {}", msg))));
                        return;
                    } else {
                        let _ = tx.send(Err(EngineError::backend(format!(
                            "Unexpected IPC status: {}",
                            status
                        ))));
                        return;
                    }
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
