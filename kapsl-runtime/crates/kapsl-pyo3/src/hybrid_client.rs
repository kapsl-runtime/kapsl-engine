use kapsl_engine_api::TensorDtype;
use kapsl_ipc::protocol::{
    HybridRequest, HybridResponse, RequestHeader, OP_HYBRID_INFER, STATUS_OK,
};
use kapsl_shm::allocator::{ShmPoolAllocator, TieredShmAllocator};
use kapsl_shm::memory::{ShmManager, TensorHeader};
use kapsl_transport::RequestMetadata;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(windows)]
use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient, PipeMode};
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

#[pyclass]
pub struct KapslHybridClient {
    shm: Arc<ShmManager>,
    allocator: TieredShmAllocator,
    socket_path: PathBuf,
    #[cfg(unix)]
    stream: Arc<Mutex<Option<UnixStream>>>,
    #[cfg(windows)]
    stream: Arc<Mutex<Option<NamedPipeClient>>>,
    rt: Runtime,
    request_id_counter: u64,
}

#[pymethods]
#[allow(clippy::useless_conversion)]
impl KapslHybridClient {
    #[new]
    fn new(shm_name: String, socket_path: String) -> PyResult<Self> {
        let shm = ShmManager::connect(&shm_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(e.to_string()))?;

        let shm_arc = Arc::new(shm);

        // Initialize simple allocator
        let tensor_pool_offset = shm_arc.tensor_pool_offset();
        let max_tensor_size = shm_arc.max_tensor_size();

        let allocator = TieredShmAllocator::new_with_default_classes(
            tensor_pool_offset,
            max_tensor_size,
            Duration::from_secs(30),
        );

        let rt = Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            shm: shm_arc,
            allocator,
            socket_path: PathBuf::from(socket_path),
            stream: Arc::new(Mutex::new(None)),
            rt,
            request_id_counter: 1,
        })
    }

    fn infer(&mut self, shape: Vec<i64>, dtype: String, data: Vec<u8>) -> PyResult<Vec<u8>> {
        let request_id = self.request_id_counter;
        self.request_id_counter += 1;

        // 1. Allocate SHM slot
        let tensor_size = std::mem::size_of::<TensorHeader>() + data.len();
        let tensor_offset = self.allocator.try_allocate(tensor_size).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "SHM tensor pool exhausted (required={} bytes, largest_slot={} bytes, layout={})",
                tensor_size,
                self.allocator.largest_slot_size(),
                self.allocator.layout_summary(),
            ))
        })?;
        let _request_lease = RequestSlotLease::new(&self.allocator, tensor_offset);

        // 2. Write tensor to SHM
        unsafe {
            let dtype = TensorDtype::from_str(&dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            write_tensor_to_shm(self.shm.as_ptr(), tensor_offset, &shape, &dtype, &data);
        }

        // 3. Prepare Hybrid Request
        let metadata = RequestMetadata::new(request_id, 0, 0, false);

        let request = HybridRequest {
            metadata,
            shm_offset: tensor_offset as u64,
            shm_size: tensor_size as u64,
        };

        // 4. Send Request via IPC and await Response
        let stream_mutex = self.stream.clone();
        let socket_path = self.socket_path.clone();

        let response = self
            .rt
            .block_on(async move {
                let mut guard = stream_mutex.lock().await;
                if guard.is_none() {
                    #[cfg(unix)]
                    let stream = UnixStream::connect(&socket_path).await?;
                    #[cfg(windows)]
                    let stream = ClientOptions::new()
                        .pipe_mode(PipeMode::Byte)
                        .open(&socket_path)
                        .map_err(|e| {
                            std::io::Error::new(
                                std::io::ErrorKind::ConnectionRefused,
                                format!(
                                    "Failed to open named pipe '{}': {}",
                                    socket_path.display(),
                                    e
                                ),
                            )
                        })?;

                    *guard = Some(stream);
                }
                let stream = guard.as_mut().unwrap();

                // Serialize payload
                let payload = bincode::serialize(&request).map_err(std::io::Error::other)?;

                // Send Header
                let header = RequestHeader {
                    model_id: 0, // Default model ID 0
                    op_code: OP_HYBRID_INFER,
                    payload_size: payload.len() as u32,
                };

                stream.write_all(&header.model_id.to_le_bytes()).await?;
                stream.write_all(&header.op_code.to_le_bytes()).await?;
                stream.write_all(&header.payload_size.to_le_bytes()).await?;

                // Send Payload
                stream.write_all(&payload).await?;

                // Read Response Header
                let mut status_buf = [0u8; 4];
                stream.read_exact(&mut status_buf).await?;
                let status = u32::from_le_bytes(status_buf);

                let mut size_buf = [0u8; 4];
                stream.read_exact(&mut size_buf).await?;
                let payload_size = u32::from_le_bytes(size_buf);

                // Read Payload
                let mut resp_payload = vec![0u8; payload_size as usize];
                stream.read_exact(&mut resp_payload).await?;

                if status != STATUS_OK {
                    let error_msg = String::from_utf8_lossy(&resp_payload).to_string();
                    return Err(std::io::Error::other(error_msg));
                }

                let response: HybridResponse = bincode::deserialize(&resp_payload)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                Ok(response)
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // 5. Read Result from SHM
        let result_offset = response.shm_offset as usize;
        let header_size = std::mem::size_of::<TensorHeader>();

        unsafe {
            let header_ptr = self.shm.as_ptr().add(result_offset) as *const TensorHeader;
            let header = &*header_ptr;

            let byte_ptr = self.shm.as_ptr().add(result_offset + header_size) as *const u8;
            let byte_len = header.data_size as usize;
            let byte_data = std::slice::from_raw_parts(byte_ptr, byte_len);
            Ok(byte_data.to_vec())
        }
    }
}

struct RequestSlotLease<'a> {
    allocator: &'a TieredShmAllocator,
    offset: usize,
}

impl<'a> RequestSlotLease<'a> {
    fn new(allocator: &'a TieredShmAllocator, offset: usize) -> Self {
        Self { allocator, offset }
    }
}

impl Drop for RequestSlotLease<'_> {
    fn drop(&mut self) {
        let _ = self.allocator.release(self.offset);
    }
}

// Helper function
unsafe fn write_tensor_to_shm(
    base: *mut u8,
    offset: usize,
    shape: &[i64],
    dtype: &TensorDtype,
    data: &[u8],
) {
    let mut shape_array = [0i64; 8];
    for (i, &s) in shape.iter().enumerate() {
        shape_array[i] = s;
    }

    let dtype_byte = match dtype {
        TensorDtype::Float32 => 0,
        TensorDtype::Float64 => 1,
        TensorDtype::Int32 => 2,
        TensorDtype::Int64 => 3,
        _ => 0,
    };

    let header = TensorHeader {
        ndim: shape.len() as u32,
        dtype: dtype_byte,
        _padding: [0; 3],
        shape: shape_array,
        data_size: data.len() as u64,
    };

    let header_ptr = base.add(offset) as *mut TensorHeader;
    std::ptr::write(header_ptr, header);

    let data_ptr = base.add(offset + std::mem::size_of::<TensorHeader>());
    std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
}
