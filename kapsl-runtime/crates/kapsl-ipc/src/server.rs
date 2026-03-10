use crate::protocol::{
    HybridRequest, HybridResponse, RequestHeader, ResponseHeader, OP_HYBRID_INFER, OP_INFER,
    OP_INFER_STREAM, STATUS_ERR, STATUS_OK, STATUS_STREAM_CHUNK, STATUS_STREAM_END,
};
use async_trait::async_trait;
use bincode;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, NamedTensor, TensorDtype};
use kapsl_scheduler::{Priority, ReplicaScheduler};
use kapsl_transport::{ResponseMetadata, TransportError, TransportServer};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ServerOptions;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

use kapsl_shm::memory::{ShmManager, TensorHeader};

pub type SchedulerLookup =
    Arc<dyn Fn(u32) -> Option<Arc<dyn ReplicaScheduler + Send + Sync>> + Send + Sync>;

#[derive(Debug, Deserialize)]
struct LegacyInferenceRequestV1 {
    input: BinaryTensorPacket,
    #[serde(default)]
    additional_inputs: Vec<NamedTensor>,
    #[serde(default)]
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LegacyInferenceRequestV0 {
    input: BinaryTensorPacket,
}

fn decode_inference_request(payload: &[u8]) -> Result<InferenceRequest, String> {
    match bincode::deserialize::<InferenceRequest>(payload) {
        Ok(request) => Ok(request),
        Err(primary_err) => {
            if let Ok(legacy) = bincode::deserialize::<LegacyInferenceRequestV1>(payload) {
                return Ok(InferenceRequest {
                    input: legacy.input,
                    additional_inputs: legacy.additional_inputs,
                    session_id: legacy.session_id,
                    metadata: None,
                    cancellation: None,
                });
            }
            if let Ok(legacy) = bincode::deserialize::<LegacyInferenceRequestV0>(payload) {
                return Ok(InferenceRequest {
                    input: legacy.input,
                    additional_inputs: Vec::new(),
                    session_id: None,
                    metadata: None,
                    cancellation: None,
                });
            }
            Err(format!("Deserialization error: {}", primary_err))
        }
    }
}

pub struct IpcServer {
    socket_path: String,
    scheduler_lookup: SchedulerLookup,
    shm_manager: Option<Arc<ShmManager>>,
}

impl IpcServer {
    pub fn new(
        socket_path: &str,
        schedulers: HashMap<u32, Arc<dyn ReplicaScheduler + Send + Sync>>,
        shm_manager: Option<Arc<ShmManager>>,
    ) -> Self {
        let schedulers = Arc::new(schedulers);
        let scheduler_lookup: SchedulerLookup =
            Arc::new(move |model_id| schedulers.get(&model_id).cloned());
        Self::new_with_lookup(socket_path, scheduler_lookup, shm_manager)
    }

    pub fn new_with_lookup(
        socket_path: &str,
        scheduler_lookup: SchedulerLookup,
        shm_manager: Option<Arc<ShmManager>>,
    ) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            scheduler_lookup,
            shm_manager,
        }
    }

    async fn run_internal(&self) -> std::io::Result<()> {
        let scheduler_lookup = self.scheduler_lookup.clone();

        #[cfg(unix)]
        {
            if std::path::Path::new(&self.socket_path).exists() {
                // Avoid clobbering a live socket from another runtime: if we can connect,
                // it is in-use and we should refuse to start.
                if UnixStream::connect(&self.socket_path).await.is_ok() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::AddrInUse,
                        format!(
                            "IPC socket path {} is already in use. Is another kapsl runtime running? Use --socket to choose a different path.",
                            self.socket_path
                        ),
                    ));
                }

                // Stale socket (or leftover file) from a previous crash.
                std::fs::remove_file(&self.socket_path)?;
            }

            let listener = UnixListener::bind(&self.socket_path)?;
            std::fs::set_permissions(&self.socket_path, std::fs::Permissions::from_mode(0o600))?;
            log::info!("IPC Server listening on {}", self.socket_path);

            loop {
                let (stream, _) = listener.accept().await?;
                let scheduler_lookup = scheduler_lookup.clone();
                let shm_manager = self.shm_manager.clone();

                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, scheduler_lookup, shm_manager).await {
                        log::error!("Connection error: {}", e);
                    }
                });
            }
        }

        #[cfg(windows)]
        {
            loop {
                let server = ServerOptions::new().create(&self.socket_path)?;

                server.connect().await?;
                let scheduler_lookup = scheduler_lookup.clone();
                let shm_manager = self.shm_manager.clone();

                tokio::spawn(async move {
                    if let Err(e) = handle_connection(server, scheduler_lookup, shm_manager).await {
                        log::error!("Connection error: {}", e);
                    }
                });
            }
        }
    }
}

#[async_trait]
impl TransportServer for IpcServer {
    async fn run(&self) -> Result<(), TransportError> {
        self.run_internal().await.map_err(TransportError::Io)
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        // Clean up socket file on shutdown
        #[cfg(unix)]
        {
            if std::path::Path::new(&self.socket_path).exists() {
                std::fs::remove_file(&self.socket_path).map_err(TransportError::Io)?;
            }
        }
        Ok(())
    }

    fn transport_type(&self) -> &'static str {
        "socket"
    }
}

pub(crate) async fn handle_connection<T>(
    mut connection: T,
    scheduler_lookup: SchedulerLookup,
    shm_manager: Option<Arc<ShmManager>>,
) -> std::io::Result<()>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        // Read header as raw bytes (not bincode)
        let mut model_id_buf = [0u8; 4];
        if connection.read_exact(&mut model_id_buf).await.is_err() {
            return Ok(()); // Connection closed
        }
        let mut op_code_buf = [0u8; 4];
        connection.read_exact(&mut op_code_buf).await?;
        let mut payload_size_buf = [0u8; 4];
        connection.read_exact(&mut payload_size_buf).await?;

        let header = RequestHeader {
            model_id: u32::from_le_bytes(model_id_buf),
            op_code: u32::from_le_bytes(op_code_buf),
            payload_size: u32::from_le_bytes(payload_size_buf),
        };

        // Read payload
        let mut payload = vec![0u8; header.payload_size as usize];
        connection.read_exact(&mut payload).await?;

        match header.op_code {
            OP_INFER_STREAM => {
                // Deserialize request
                let request: InferenceRequest = match decode_inference_request(&payload) {
                    Ok(req) => req,
                    Err(error_msg) => {
                        let resp_header = ResponseHeader {
                            status: STATUS_ERR,
                            payload_size: error_msg.len() as u32,
                        };
                        connection
                            .write_all(&resp_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&resp_header.payload_size.to_le_bytes())
                            .await?;
                        connection.write_all(error_msg.as_bytes()).await?;
                        continue;
                    }
                };

                // Get scheduler for model
                let scheduler = match scheduler_lookup(header.model_id) {
                    Some(s) => s,
                    None => {
                        let error_msg = format!("Model {} not found", header.model_id);
                        let resp_header = ResponseHeader {
                            status: STATUS_ERR,
                            payload_size: error_msg.len() as u32,
                        };
                        connection
                            .write_all(&resp_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&resp_header.payload_size.to_le_bytes())
                            .await?;
                        connection.write_all(error_msg.as_bytes()).await?;
                        continue;
                    }
                };

                // Execute streaming inference
                let stream_result = scheduler
                    .infer_stream(request, Priority::LatencyCritical, false)
                    .await;

                use futures::StreamExt;
                match stream_result {
                    Ok(mut inference_stream) => {
                        while let Some(result) = inference_stream.next().await {
                            match result {
                                Ok(packet) => {
                                    // Serialize packet
                                    let response_bytes = match bincode::serialize(&packet) {
                                        Ok(b) => b,
                                        Err(e) => {
                                            log::error!("Serialization error: {}", e);
                                            break;
                                        }
                                    };

                                    // Send chunk header
                                    let response_header = ResponseHeader {
                                        status: STATUS_STREAM_CHUNK,
                                        payload_size: response_bytes.len() as u32,
                                    };

                                    connection
                                        .write_all(&response_header.status.to_le_bytes())
                                        .await?;
                                    connection
                                        .write_all(&response_header.payload_size.to_le_bytes())
                                        .await?;
                                    connection.write_all(&response_bytes).await?;
                                    connection.flush().await?;
                                }
                                Err(e) => {
                                    // Send error frame and stop
                                    let error_msg = e.to_string();
                                    let response_bytes = error_msg.as_bytes();
                                    let response_header = ResponseHeader {
                                        status: STATUS_ERR,
                                        payload_size: response_bytes.len() as u32,
                                    };
                                    connection
                                        .write_all(&response_header.status.to_le_bytes())
                                        .await?;
                                    connection
                                        .write_all(&response_header.payload_size.to_le_bytes())
                                        .await?;
                                    connection.write_all(response_bytes).await?;
                                    connection.flush().await?;
                                    break;
                                }
                            }
                        }

                        // Send End of Stream frame
                        let response_header = ResponseHeader {
                            status: STATUS_STREAM_END,
                            payload_size: 0,
                        };
                        connection
                            .write_all(&response_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&response_header.payload_size.to_le_bytes())
                            .await?;
                        connection.flush().await?;
                    }
                    Err(e) => {
                        // Send error frame for initial failure
                        let error_msg = e.to_string();
                        let response_bytes = error_msg.as_bytes();
                        let response_header = ResponseHeader {
                            status: STATUS_ERR,
                            payload_size: response_bytes.len() as u32,
                        };
                        connection
                            .write_all(&response_header.status.to_le_bytes())
                            .await?;
                        connection
                            .write_all(&response_header.payload_size.to_le_bytes())
                            .await?;
                        connection.write_all(response_bytes).await?;
                        connection.flush().await?;
                    }
                }
            }
            OP_INFER => {
                // Find scheduler for model_id
                if let Some(scheduler) = scheduler_lookup(header.model_id) {
                    // Deserialize payload to InferenceRequest
                    let request: InferenceRequest = match decode_inference_request(&payload) {
                        Ok(req) => req,
                        Err(error_msg) => {
                            let resp_header = ResponseHeader {
                                status: STATUS_ERR,
                                payload_size: error_msg.len() as u32,
                            };
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(error_msg.as_bytes()).await?;
                            continue;
                        }
                    };

                    // Process
                    // Default to Throughput priority and allow GPU (force_cpu = false)
                    let result = scheduler.infer(&request, Priority::Throughput, false).await;

                    match result {
                        Ok(output) => {
                            let output_bytes =
                                bincode::serialize(&output).map_err(std::io::Error::other)?;

                            let resp_header = ResponseHeader {
                                status: STATUS_OK,
                                payload_size: output_bytes.len() as u32,
                            };

                            // Write header as raw bytes (not bincode)
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(&output_bytes).await?;
                        }
                        Err(e) => {
                            let error_msg = e.to_string();
                            let resp_header = ResponseHeader {
                                status: STATUS_ERR,
                                payload_size: error_msg.len() as u32,
                            };
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(error_msg.as_bytes()).await?;
                        }
                    }
                } else {
                    // Model not found
                    let error_msg = format!("Model {} not found", header.model_id);
                    let resp_header = ResponseHeader {
                        status: STATUS_ERR,
                        payload_size: error_msg.len() as u32,
                    };
                    connection
                        .write_all(&resp_header.status.to_le_bytes())
                        .await?;
                    connection
                        .write_all(&resp_header.payload_size.to_le_bytes())
                        .await?;
                    connection.write_all(error_msg.as_bytes()).await?;
                }
            }
            OP_HYBRID_INFER => {
                // Payload already read at line 131-132, just deserialize it
                // Deserialize HybridRequest
                let hybrid_req: HybridRequest = bincode::deserialize(&payload)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                if let Some(shm_manager) = &shm_manager {
                    let base_ptr = shm_manager.as_ptr();

                    // Read TensorHeader from SHM
                    let header_ptr = unsafe {
                        base_ptr.add(hybrid_req.shm_offset as usize) as *const TensorHeader
                    };
                    let tensor_header = unsafe { &*header_ptr };

                    // Read tensor data
                    let data_ptr = unsafe {
                        base_ptr.add(
                            hybrid_req.shm_offset as usize + std::mem::size_of::<TensorHeader>(),
                        )
                    };
                    let data_slice = unsafe {
                        std::slice::from_raw_parts(data_ptr, tensor_header.data_size as usize)
                    };

                    // Build InferenceRequest
                    let shape = tensor_header.shape[0..tensor_header.ndim as usize].to_vec();
                    let dtype = match tensor_header.dtype {
                        0 => TensorDtype::Float32,
                        1 => TensorDtype::Float64,
                        2 => TensorDtype::Int32,
                        3 => TensorDtype::Int64,
                        _ => TensorDtype::Float32,
                    };

                    let packet = BinaryTensorPacket {
                        shape,
                        dtype,
                        data: data_slice.to_vec(),
                    };

                    let request = InferenceRequest {
                        input: packet,
                        additional_inputs: Vec::new(),
                        session_id: None,
                        metadata: None,
                        cancellation: None,
                    };

                    // Perform inference
                    let result =
                        if let Some(scheduler) = scheduler_lookup(hybrid_req.metadata.model_id) {
                            scheduler
                                .infer(
                                    &request,
                                    Priority::Throughput,
                                    hybrid_req.metadata.force_cpu,
                                )
                                .await
                        } else {
                            Err(kapsl_engine_api::EngineError::ModelNotLoaded)
                        };

                    match result {
                        Ok(output) => {
                            // Serialize output to BinaryTensorPacket
                            let packet = BinaryTensorPacket {
                                shape: output.shape,
                                dtype: output.dtype,
                                data: output.data,
                            };

                            // Calculate required size
                            let output_size =
                                std::mem::size_of::<TensorHeader>() + packet.data.len();

                            // Allocate output slot with bounds checking
                            // Use smaller slots (1MB) and more of them (400 slots from 512MB to 912MB)
                            static SERVER_SLOT_COUNTER: std::sync::atomic::AtomicUsize =
                                std::sync::atomic::AtomicUsize::new(0);
                            let slot = SERVER_SLOT_COUNTER
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let output_offset = 512 * 1024 * 1024 + (slot % 400) * 1_000_000; // 1MB slots, 400 slots

                            // Bounds check
                            let shm_size = shm_manager.size();
                            if output_offset + output_size > shm_size {
                                let error_msg = format!("Output would exceed SHM bounds: offset={}, size={}, shm_size={}",
                                    output_offset, output_size, shm_size);
                                let resp_header = ResponseHeader {
                                    status: STATUS_ERR,
                                    payload_size: error_msg.len() as u32,
                                };
                                connection
                                    .write_all(&resp_header.status.to_le_bytes())
                                    .await?;
                                connection
                                    .write_all(&resp_header.payload_size.to_le_bytes())
                                    .await?;
                                connection.write_all(error_msg.as_bytes()).await?;
                                continue;
                            }

                            // Write result to SHM
                            // Re-acquire base_ptr to avoid holding !Send raw pointer across await
                            let base_ptr = shm_manager.as_ptr();
                            unsafe {
                                // Write header
                                let out_header = TensorHeader {
                                    ndim: packet.shape.len() as u32,
                                    dtype: match packet.dtype {
                                        TensorDtype::Float32 => 0,
                                        TensorDtype::Float64 => 1,
                                        TensorDtype::Int32 => 2,
                                        TensorDtype::Int64 => 3,
                                        _ => 0,
                                    },
                                    _padding: [0; 3],
                                    shape: {
                                        let mut arr = [0i64; 8];
                                        for (i, &v) in packet.shape.iter().enumerate() {
                                            arr[i] = v;
                                        }
                                        arr
                                    },
                                    data_size: packet.data.len() as u64,
                                };

                                let hdr_ptr = base_ptr.add(output_offset) as *mut TensorHeader;
                                std::ptr::write(hdr_ptr, out_header);

                                let data_ptr = base_ptr
                                    .add(output_offset + std::mem::size_of::<TensorHeader>());
                                std::ptr::copy_nonoverlapping(
                                    packet.data.as_ptr(),
                                    data_ptr,
                                    packet.data.len(),
                                );
                            }

                            // Send HybridResponse
                            let resp = HybridResponse {
                                metadata: ResponseMetadata {
                                    request_id: hybrid_req.metadata.request_id,
                                    status: STATUS_OK as u8,
                                    _padding: [0; 7],
                                    latency_ns: 0,
                                },
                                shm_offset: output_offset as u64,
                                shm_size: (std::mem::size_of::<TensorHeader>() + packet.data.len())
                                    as u64,
                            };

                            let resp_bytes =
                                bincode::serialize(&resp).map_err(std::io::Error::other)?;

                            let resp_header = ResponseHeader {
                                status: STATUS_OK,
                                payload_size: resp_bytes.len() as u32,
                            };

                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(&resp_bytes).await?;
                        }
                        Err(e) => {
                            let error_msg = e.to_string();
                            let resp_header = ResponseHeader {
                                status: STATUS_ERR,
                                payload_size: error_msg.len() as u32,
                            };
                            connection
                                .write_all(&resp_header.status.to_le_bytes())
                                .await?;
                            connection
                                .write_all(&resp_header.payload_size.to_le_bytes())
                                .await?;
                            connection.write_all(error_msg.as_bytes()).await?;
                        }
                    }
                } else {
                    let error_msg = "SHM Manager not configured".to_string();
                    let resp_header = ResponseHeader {
                        status: STATUS_ERR,
                        payload_size: error_msg.len() as u32,
                    };
                    connection
                        .write_all(&resp_header.status.to_le_bytes())
                        .await?;
                    connection
                        .write_all(&resp_header.payload_size.to_le_bytes())
                        .await?;
                    connection.write_all(error_msg.as_bytes()).await?;
                }
            }
            _ => {
                // Unsupported op
                let resp_header = ResponseHeader {
                    status: STATUS_ERR,
                    payload_size: 0,
                };
                // Write header as raw bytes (not bincode)
                connection
                    .write_all(&resp_header.status.to_le_bytes())
                    .await?;
                connection
                    .write_all(&resp_header.payload_size.to_le_bytes())
                    .await?;
            }
        }
    }
}
