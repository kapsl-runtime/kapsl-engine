use kapsl_engine_api::{BinaryTensorPacket, TensorDtype};
use kapsl_ipc::{RequestHeader, ResponseHeader, OP_INFER, STATUS_OK};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::thread;
#[cfg(windows)]
use tokio::net::windows::named_pipe::NamedPipeClient;

#[cfg(unix)]
fn main() {
    run_benchmark();
}

#[cfg(windows)]
#[tokio::main]
async fn main() {
    run_benchmark_async().await;
}

#[cfg(unix)]
fn run_benchmark() {
    let socket_path = "/tmp/kapsl.sock";

    // Test 4 models with different backends
    let model_configs = vec![
        (0, "MNIST-CPU", vec![1, 1, 28, 28], 28 * 28),
        (1, "SqueezeNet-CPU", vec![1, 3, 224, 224], 3 * 224 * 224),
        (2, "MNIST-CUDA", vec![1, 1, 28, 28], 28 * 28),
        (3, "MNIST-TensorRT", vec![1, 1, 28, 28], 28 * 28),
    ];

    let handles: Vec<_> = model_configs
        .into_iter()
        .map(|(model_id, name, shape, data_len)| {
            let socket_path = socket_path.to_string();
            thread::spawn(move || {
                log::info!("[{}] Connecting to {}...", name, socket_path);
                let mut stream = UnixStream::connect(socket_path).expect("Failed to connect");
                log::info!("[{}] Connected!", name);

                let input = BinaryTensorPacket {
                    shape,
                    dtype: TensorDtype::Float32,
                    data: vec![0u8; data_len * 4], // 4 bytes per float
                };

                let input_bytes = bincode::serialize(&input).unwrap();

                let header = RequestHeader {
                    model_id,
                    op_code: OP_INFER,
                    payload_size: input_bytes.len() as u32,
                };

                let header_bytes = bincode::serialize(&header).unwrap();

                log::info!("[{}] Sending inference request...", name);
                let start = std::time::Instant::now();
                stream.write_all(&header_bytes).unwrap();
                stream.write_all(&input_bytes).unwrap();

                // Read response
                let mut header_buf = [0u8; 8];
                stream.read_exact(&mut header_buf).unwrap();

                let resp_header: ResponseHeader = bincode::deserialize(&header_buf).unwrap();

                if resp_header.status == STATUS_OK {
                    let mut payload = vec![0u8; resp_header.payload_size as usize];
                    stream.read_exact(&mut payload).unwrap();

                    let output: BinaryTensorPacket = bincode::deserialize(&payload).unwrap();
                    let duration = start.elapsed();
                    log::info!(
                        "[{}] ✓ Response received in {:?} | Shape: {:?}",
                        name,
                        duration,
                        output.shape
                    );
                } else {
                    let mut payload = vec![0u8; resp_header.payload_size as usize];
                    stream.read_exact(&mut payload).unwrap();
                    let error_msg = String::from_utf8(payload).unwrap();
                    log::info!("[{}] ✗ Error: {}", name, error_msg);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[cfg(windows)]
async fn run_benchmark_async() {
    let socket = r"\\.\pipe\kapsl";

    let model_configs = vec![
        (0, "MNIST-CPU", vec![1, 1, 28, 28], 28 * 28),
        (1, "SqueezeNet-CPU", vec![1, 3, 224, 224], 3 * 224 * 224),
        (2, "MNIST-CUDA", vec![1, 1, 28, 28], 28 * 28),
        (3, "MNIST-TensorRT", vec![1, 1, 28, 28], 28 * 28),
    ];

    let handles: Vec<_> = model_configs
        .into_iter()
        .map(|(model_id, name, shape, data_len)| {
            let socket = socket.to_string();
            tokio::spawn(async move {
                use tokio::{
                    io::{AsyncReadExt, AsyncWriteExt},
                    net::windows::named_pipe::{ClientOptions, PipeMode},
                };

                log::info!("[{}] Connecting to {}...", name, socket);
                let mut stream = ClientOptions::new()
                    .pipe_mode(PipeMode::Byte)
                    .open(socket)
                    .expect("Failed to connect");
                log::info!("[{}] Connected!", name);

                let input = BinaryTensorPacket {
                    shape,
                    dtype: TensorDtype::Float32,
                    data: vec![0u8; data_len * 4], // 4 bytes per float
                };

                let input_bytes = bincode::serialize(&input).unwrap();

                let header = RequestHeader {
                    model_id,
                    op_code: OP_INFER,
                    payload_size: input_bytes.len() as u32,
                };

                let header_bytes = bincode::serialize(&header).unwrap();

                log::info!("[{}] Sending inference request...", name);
                let start = std::time::Instant::now();
                stream.write_all(&header_bytes).await.unwrap();
                stream.write_all(&input_bytes).await.unwrap();

                // Read response
                let mut header_buf = [0u8; 8];
                stream.read_exact(&mut header_buf).await.unwrap();

                let resp_header: ResponseHeader = bincode::deserialize(&header_buf).unwrap();

                if resp_header.status == STATUS_OK {
                    let mut payload = vec![0u8; resp_header.payload_size as usize];
                    stream.read_exact(&mut payload).await.unwrap();

                    let output: BinaryTensorPacket = bincode::deserialize(&payload).unwrap();
                    let duration = start.elapsed();
                    log::info!(
                        "[{}] ✓ Response received in {:?} | Shape: {:?}",
                        name,
                        duration,
                        output.shape
                    );
                } else {
                    let mut payload = vec![0u8; resp_header.payload_size as usize];
                    stream.read_exact(&mut payload).await.unwrap();
                    let error_msg = String::from_utf8(payload).unwrap();
                    log::info!("[{}] ✗ Error: {}", name, error_msg);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.await.unwrap();
    }
}
