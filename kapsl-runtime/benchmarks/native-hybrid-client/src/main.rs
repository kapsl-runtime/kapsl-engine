use clap::Parser;
use kapsl_ipc::protocol::{
    HybridRequest, HybridResponse, RequestHeader, OP_HYBRID_INFER, STATUS_OK,
};
use kapsl_shm::allocator::SimpleShmAllocator;
use kapsl_shm::memory::{ShmManager, TensorHeader};
use kapsl_transport::RequestMetadata;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[cfg(unix)]
use tokio::net::UnixStream as IpcStream;

#[cfg(windows)]
use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient as IpcStream};

#[derive(Parser)]
struct Args {
    /// Shared memory name (e.g., /kapsl_shm_12345)
    #[arg(long)]
    shm_name: String,

    /// Socket path
    #[arg(long, default_value = "/tmp/kapsl.sock")]
    socket: String,

    /// Number of requests to send
    #[arg(long, default_value_t = 100)]
    num_requests: usize,

    /// Worker ID (for logging)
    #[arg(long, default_value_t = 0)]
    worker_id: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Connect to SHM
    let shm = Arc::new(ShmManager::connect(&args.shm_name)?);
    let tensor_pool_offset = shm.tensor_pool_offset();
    let max_tensor_size = shm.max_tensor_size();
    let slot_size = 10 * 1024 * 1024; // 10MB
    let num_slots = max_tensor_size / slot_size;
    let allocator = SimpleShmAllocator::new(tensor_pool_offset, slot_size, num_slots);

    // Connect to socket
    #[cfg(unix)]
    let mut stream = IpcStream::connect(&args.socket).await?;

    #[cfg(windows)]
    let mut stream = ClientOptions::new().open(&args.socket)?;

    // Prepare test data (SqueezeNet input: 1x3x224x224 float32)
    let input_shape = vec![1i64, 3, 224, 224];
    let input_elements: usize = input_shape.iter().map(|&d| d as usize).product();
    let input_data: Vec<f32> = vec![0.0; input_elements];
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Warmup
    for _ in 0..10 {
        let _ =
            send_inference_request(&mut stream, &shm, &allocator, &input_shape, &input_bytes, 1)
                .await;
    }

    // Benchmark
    let start = Instant::now();
    let mut latencies = Vec::new();

    for i in 0..args.num_requests {
        let req_start = Instant::now();
        match send_inference_request(
            &mut stream,
            &shm,
            &allocator,
            &input_shape,
            &input_bytes,
            i as u64 + 2,
        )
        .await
        {
            Ok(_) => {
                latencies.push(req_start.elapsed().as_secs_f64() * 1000.0);
            }
            Err(e) => {
                eprintln!("Worker {} request {} failed: {}", args.worker_id, i, e);
            }
        }
    }

    let total_time = start.elapsed().as_secs_f64();
    let avg_latency: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;

    println!(
        "Worker {}: {} requests in {:.2}s, avg latency: {:.2}ms",
        args.worker_id,
        latencies.len(),
        total_time,
        avg_latency
    );

    Ok(())
}

async fn send_inference_request(
    stream: &mut IpcStream,
    shm: &Arc<ShmManager>,
    allocator: &SimpleShmAllocator,
    shape: &[i64],
    data: &[u8],
    request_id: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Allocate SHM slot
    let tensor_size = std::mem::size_of::<TensorHeader>() + data.len();
    let tensor_offset = allocator
        .try_allocate(tensor_size)
        .ok_or_else(|| format!("SHM tensor pool exhausted (required={} bytes)", tensor_size))?;
    let _request_lease = RequestSlotLease::new(allocator, tensor_offset);

    // Write tensor to SHM
    unsafe {
        write_tensor_to_shm(shm.as_ptr(), tensor_offset, shape, "float32", data);
    }

    // Create HybridRequest
    let metadata = RequestMetadata::new(request_id, 0, 0, false);
    let request = HybridRequest {
        metadata,
        shm_offset: tensor_offset as u64,
        shm_size: tensor_size as u64,
    };

    // Serialize payload
    let payload = bincode::serialize(&request)?;

    // Send header
    let header = RequestHeader {
        model_id: 0,
        op_code: OP_HYBRID_INFER,
        payload_size: payload.len() as u32,
    };

    stream.write_all(&header.model_id.to_le_bytes()).await?;
    stream.write_all(&header.op_code.to_le_bytes()).await?;
    stream.write_all(&header.payload_size.to_le_bytes()).await?;
    stream.write_all(&payload).await?;

    // Read response header
    let mut status_buf = [0u8; 4];
    stream.read_exact(&mut status_buf).await?;
    let status = u32::from_le_bytes(status_buf);

    let mut size_buf = [0u8; 4];
    stream.read_exact(&mut size_buf).await?;
    let payload_size = u32::from_le_bytes(size_buf);

    // Read payload
    let mut resp_payload = vec![0u8; payload_size as usize];
    stream.read_exact(&mut resp_payload).await?;

    if status != STATUS_OK {
        return Err(format!("Server error: {}", String::from_utf8_lossy(&resp_payload)).into());
    }

    let response: HybridResponse = bincode::deserialize(&resp_payload)?;

    // Read result from SHM
    let result_offset = response.shm_offset as usize;
    let header_size = std::mem::size_of::<TensorHeader>();

    unsafe {
        let header_ptr = shm.as_ptr().add(result_offset) as *const TensorHeader;
        let header = &*header_ptr;

        let byte_ptr = shm.as_ptr().add(result_offset + header_size) as *const u8;
        let byte_len = header.data_size as usize;
        let byte_data = std::slice::from_raw_parts(byte_ptr, byte_len);
        Ok(byte_data.to_vec())
    }
}

struct RequestSlotLease<'a> {
    allocator: &'a SimpleShmAllocator,
    offset: usize,
}

impl<'a> RequestSlotLease<'a> {
    fn new(allocator: &'a SimpleShmAllocator, offset: usize) -> Self {
        Self { allocator, offset }
    }
}

impl Drop for RequestSlotLease<'_> {
    fn drop(&mut self) {
        let _ = self.allocator.release(self.offset);
    }
}

unsafe fn write_tensor_to_shm(
    base: *mut u8,
    offset: usize,
    shape: &[i64],
    dtype: &str,
    data: &[u8],
) {
    let mut shape_array = [0i64; 8];
    for (i, &s) in shape.iter().enumerate() {
        shape_array[i] = s;
    }

    let dtype_byte = match dtype {
        "float32" => 0,
        "float64" => 1,
        "int32" => 2,
        "int64" => 3,
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
