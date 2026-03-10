use kapsl_shm::memory::{ShmManager, TensorHeader};
use kapsl_shm::ring_buffer::LockFreeRingBuffer;
use kapsl_transport::{RequestMetadata, ResponseMetadata};
use std::time::Instant;

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

fn main() {
    // Get PID from command line or use default
    let args: Vec<String> = std::env::args().collect();
    let shm_name = if args.len() > 1 {
        format!("/kapsl_shm_{}", args[1])
    } else {
        eprintln!("Usage: {} <pid>", args[0]);
        eprintln!("Example: {} $(pgrep -f 'kapsl.*shm')", args[0]);
        std::process::exit(1);
    };

    println!("🚀 Native Rust SHM Benchmark\n");
    println!("Connecting to: {}", shm_name);

    // Connect to shared memory
    let shm = match ShmManager::connect(&shm_name) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to connect to SHM: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ Connected to shared memory");

    let req_queue_offset = shm.request_queue_offset();
    let resp_queue_offset = shm.response_queue_offset();

    println!("Request queue offset: {}", req_queue_offset);
    println!("Response queue offset: {}\n", resp_queue_offset);

    const NUM_REQUESTS: usize = 100;
    let mut latencies = Vec::with_capacity(NUM_REQUESTS);

    // Prepare dummy MNIST input (1x1x28x28 float32)
    let input_data: Vec<f32> = vec![0.0; 28 * 28];
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    println!("Running {} requests...", NUM_REQUESTS);

    for request_id in 1..=NUM_REQUESTS {
        let start = Instant::now();

        // Allocate tensor slot
        let tensor_offset = 128 * 1024 + (request_id % 100) * 10_000_000;

        // Write tensor to shared memory
        unsafe {
            write_tensor_to_shm(
                shm.as_ptr(),
                tensor_offset,
                &[1, 1, 28, 28],
                "float32",
                &input_bytes,
            );
        }

        // Create request metadata
        let metadata = RequestMetadata {
            request_id: request_id as u64,
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
            tensor_size: (std::mem::size_of::<TensorHeader>() + input_bytes.len()) as u64,
        };

        // Push to request queue
        unsafe {
            let req_queue = LockFreeRingBuffer::<ShmRequest>::connect(
                shm.as_ptr().add(req_queue_offset) as *mut ShmRequest,
                1024,
            );

            if let Err(e) = req_queue.push(request) {
                eprintln!("Failed to push request: {:?}", e);
                continue;
            }
        }

        // Wait for response using pipe notification
        let read_fd = shm.notify_read_fd();

        if read_fd >= 0 {
            // Use select() to wait for pipe notification
            unsafe {
                #[cfg(unix)]
                {
                    let mut read_fds: libc::fd_set = std::mem::zeroed();
                    libc::FD_ZERO(&mut read_fds);
                    libc::FD_SET(read_fd, &mut read_fds);

                    let mut timeout = libc::timeval {
                        tv_sec: 5,
                        tv_usec: 0,
                    };

                    let ret = libc::select(
                        read_fd + 1,
                        &mut read_fds,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        &mut timeout,
                    );

                    if ret > 0 {
                        // Drain notification pipe completely
                        let mut buf = [0u8; 128];
                        loop {
                            let n = libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, 128);
                            if n < 0 {
                                // EAGAIN or error
                                break;
                            }
                            if n < 128 {
                                break;
                            }
                        }
                    }
                }

                #[cfg(windows)]
                {}
            }
        }

        // Pop response from queue
        let response = unsafe {
            let resp_queue = LockFreeRingBuffer::<ShmResponse>::connect(
                shm.as_ptr().add(resp_queue_offset) as *mut ShmResponse,
                1024,
            );

            // Loop with reasonable timeout (1 second)
            let loop_start = std::time::Instant::now();
            loop {
                if let Some(resp) = resp_queue.pop() {
                    if resp.metadata.request_id == request_id as u64 {
                        break resp;
                    }
                }

                if loop_start.elapsed().as_millis() > 1000 {
                    eprintln!("Request {} timed out after 1s", request_id);
                    break ShmResponse {
                        metadata: kapsl_transport::ResponseMetadata {
                            request_id: 0,
                            status: 1,
                            _padding: [0; 7],
                            latency_ns: 0,
                        },
                        result_offset: 0,
                        result_size: 0,
                        error_offset: 0,
                    };
                }

                // Yield to let server push
                std::thread::yield_now();
            }
        };

        let elapsed = start.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms

        if response.metadata.status != 0 {
            eprintln!("Request {} failed", request_id);
        }
    }

    // Calculate statistics
    let total: f64 = latencies.iter().sum();
    let avg = total / latencies.len() as f64;

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("\n{}", "=".repeat(60));
    println!("Results ({} requests)", NUM_REQUESTS);
    println!("{}", "-".repeat(60));
    println!("Average:  {:.3}ms", avg);
    println!("p50:      {:.3}ms", p50);
    println!("p95:      {:.3}ms", p95);
    println!("p99:      {:.3}ms", p99);
    println!("Min:      {:.3}ms", latencies[0]);
    println!("Max:      {:.3}ms", latencies[latencies.len() - 1]);
    println!("{}", "=".repeat(60));

    println!("\n🎯 Native Rust SHM latency: {:.3}ms", avg);
    println!("   (Compare to Python SHM: ~1.084ms)");
    println!("   (Compare to Unix socket: ~0.172ms)");
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
