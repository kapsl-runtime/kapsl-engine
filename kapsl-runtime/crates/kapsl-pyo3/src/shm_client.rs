use kapsl_engine_api::{BinaryTensorPacket, TensorDtype};
use kapsl_shm::allocator::{ShmPoolAllocator, TieredShmAllocator};
use kapsl_shm::memory::{ShmManager, TensorHeader};
use kapsl_shm::ring_buffer::LockFreeRingBuffer;
use kapsl_transport::{RequestMetadata, ResponseMetadata};
use pyo3::prelude::*;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

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

/// PyO3 wrapper for shared memory client
#[pyclass]
pub struct KapslShmClient {
    shm: Arc<ShmManager>,
    allocator: TieredShmAllocator,
    request_id_counter: u64,
}

#[pymethods]
#[allow(clippy::useless_conversion)]
impl KapslShmClient {
    #[new]
    fn new(shm_name: String) -> PyResult<Self> {
        let shm = ShmManager::connect(&shm_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(e.to_string()))?;

        let tensor_pool_offset = shm.tensor_pool_offset();
        let max_tensor_size = shm.max_tensor_size();
        let allocator = TieredShmAllocator::new_with_default_classes(
            tensor_pool_offset,
            max_tensor_size,
            Duration::from_secs(30),
        );

        Ok(Self {
            shm: Arc::new(shm),
            allocator,
            request_id_counter: 1,
        })
    }

    fn infer(&mut self, shape: Vec<i64>, dtype: String, data: Vec<u8>) -> PyResult<Vec<u8>> {
        let request_id = self.request_id_counter;
        self.request_id_counter += 1;

        // Write tensor to shared memory
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
        unsafe {
            let dtype = TensorDtype::from_str(&dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            write_tensor_to_shm(self.shm.as_ptr(), tensor_offset, &shape, &dtype, &data);
        }

        // Create request metadata
        let metadata = RequestMetadata {
            request_id,
            model_id: 0,
            priority: 0,
            force_cpu: false,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            _padding: [0; 2],
        };

        let request = ShmRequest {
            metadata,
            tensor_offset: tensor_offset as u64,
            tensor_size: tensor_size as u64,
        };

        // Push to request queue
        unsafe {
            let req_queue_offset = self.shm.request_queue_offset();
            let req_queue = LockFreeRingBuffer::<ShmRequest>::connect(
                self.shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                1024,
            );

            req_queue
                .push(request)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }

        // Wait for response using pipe notification
        #[cfg(unix)]
        {
            let read_fd = self.shm.notify_read_fd();

            if read_fd >= 0 {
                // Use select() to wait for pipe notification

                unsafe {
                    let mut read_fds: libc::fd_set = std::mem::zeroed();
                    libc::FD_ZERO(&mut read_fds);
                    libc::FD_SET(read_fd, &mut read_fds);

                    // 5 second timeout
                    let mut timeout = libc::timeval {
                        tv_sec: 5,
                        tv_usec: 0,
                    };

                    // Release GIL while waiting
                    let ret = Python::with_gil(|py| {
                        py.allow_threads(|| {
                            libc::select(
                                read_fd + 1,
                                &mut read_fds,
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                                &mut timeout,
                            )
                        })
                    });

                    if ret > 0 {
                        // Drain notification pipe
                        let mut buf = [0u8; 128];
                        loop {
                            let n = libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, 128);
                            if n < 0 || n < 128 {
                                break;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(windows)]
        {
            // On Windows the notification value in shared memory is a CRT file descriptor,
            // not a waitable Win32 HANDLE. We therefore rely on the bounded polling loop
            // below to detect responses.
        }

        // Poll for response
        let start = std::time::Instant::now();
        loop {
            // Check response queue
            let response = unsafe {
                let resp_queue = LockFreeRingBuffer::<ShmResponse>::connect(
                    self.shm.as_ptr().add(self.shm.response_queue_offset()) as *mut ShmResponse,
                    1024,
                );
                resp_queue.pop()
            };

            if let Some(resp) = response {
                if resp.metadata.request_id == request_id {
                    // Process response...
                    if resp.metadata.status != 0 {
                        // Handle error...
                        let error_msg = if resp.error_offset > 0 {
                            // Read error string from SHM
                            unsafe {
                                let ptr = self.shm.as_ptr().add(resp.error_offset as usize);
                                let len_ptr = ptr as *const u64;
                                let len = *len_ptr as usize;
                                let data_ptr = ptr.add(8);
                                let slice = std::slice::from_raw_parts(data_ptr, len);
                                String::from_utf8_lossy(slice).to_string()
                            }
                        } else {
                            "Unknown error".to_string()
                        };
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error_msg));
                    }

                    // Success - read tensor
                    let result_offset = resp.result_offset as usize;
                    let header_size = std::mem::size_of::<TensorHeader>();

                    unsafe {
                        let header_ptr =
                            self.shm.as_ptr().add(result_offset) as *const TensorHeader;
                        let header = &*header_ptr;

                        let byte_ptr =
                            self.shm.as_ptr().add(result_offset + header_size) as *const u8;
                        let byte_len = header.data_size as usize;
                        let byte_data = std::slice::from_raw_parts(byte_ptr, byte_len);
                        return Ok(byte_data.to_vec());
                    }
                }
            }

            if start.elapsed().as_secs() > 5 {
                return Err(PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                    "Request timed out",
                ));
            }

            // Yield to Python interpreter to allow signal handling
            Python::with_gil(|py| py.check_signals())?;

            // Small sleep if we missed the notification or it was spurious
            std::thread::sleep(std::time::Duration::from_micros(1));
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

// Helper functions

unsafe fn write_tensor_to_shm(
    base: *mut u8,
    offset: usize,
    shape: &[i64],
    dtype: &TensorDtype,
    data: &[u8],
) -> BinaryTensorPacket {
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
    let shape: Vec<i64> = header.shape[0..header.ndim as usize].to_vec();

    let dtype = match header.dtype {
        0 => TensorDtype::Float32,
        1 => TensorDtype::Float64,
        2 => TensorDtype::Int32,
        3 => TensorDtype::Int64,
        _ => TensorDtype::Float32,
    };

    let data_ptr = base.add(offset + std::mem::size_of::<TensorHeader>());
    let data = std::slice::from_raw_parts(data_ptr, header.data_size as usize).to_vec();

    BinaryTensorPacket { shape, dtype, data }
}
