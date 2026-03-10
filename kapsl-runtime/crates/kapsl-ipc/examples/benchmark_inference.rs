// Inference Latency Benchmark
//
// Measures actual MNIST model inference performance

use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::time::Instant;
#[cfg(windows)]
use tokio::net::windows::named_pipe::NamedPipeClient;

const OP_INFER: u32 = 1;
const STATUS_OK: u32 = 0;

fn create_mnist_input() -> BinaryTensorPacket {
    // Create dummy 28x28 MNIST image (all 0.5)
    let size = 28 * 28;
    let data: Vec<f32> = vec![0.5; size];
    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

    BinaryTensorPacket {
        shape: vec![1, 1, 28, 28],
        dtype: TensorDtype::Float32,
        data: bytes,
    }
}

async fn send_inference_request(
    #[cfg(unix)] stream: &mut UnixStream,
    #[cfg(windows)] stream: &mut NamedPipeClient,
    model_id: u32,
) -> Result<(BinaryTensorPacket, f64), String> {
    let input = create_mnist_input();
    let request = InferenceRequest {
        input,
        additional_inputs: Vec::new(),
        session_id: None,
        metadata: None,
        cancellation: None,
    };

    // Serialize request
    let payload =
        bincode::serialize(&request).map_err(|e| format!("Serialization error: {}", e))?;

    // Build header
    let model_id_bytes = model_id.to_le_bytes();
    let op_code_bytes = OP_INFER.to_le_bytes();
    let payload_size_bytes = (payload.len() as u32).to_le_bytes();

    // Start timing
    let start = Instant::now();

    // Send request
    #[cfg(unix)]
    {
        stream
            .write_all(&model_id_bytes)
            .map_err(|e| format!("Write error: {}", e))?;
        stream
            .write_all(&op_code_bytes)
            .map_err(|e| format!("Write error: {}", e))?;
        stream
            .write_all(&payload_size_bytes)
            .map_err(|e| format!("Write error: {}", e))?;
        stream
            .write_all(&payload)
            .map_err(|e| format!("Write error: {}", e))?;

        // Read response header
        let mut status_buf = [0u8; 4];
        stream
            .read_exact(&mut status_buf)
            .map_err(|e| format!("Read error: {}", e))?;
        let status = u32::from_le_bytes(status_buf);

        let mut resp_size_buf = [0u8; 4];
        stream
            .read_exact(&mut resp_size_buf)
            .map_err(|e| format!("Read error: {}", e))?;
        let resp_size = u32::from_le_bytes(resp_size_buf);

        // Read response payload
        let mut resp_payload = vec![0u8; resp_size as usize];
        stream
            .read_exact(&mut resp_payload)
            .map_err(|e| format!("Read error: {}", e))?;

        let latency = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms

        if status == STATUS_OK {
            let output: BinaryTensorPacket = bincode::deserialize(&resp_payload)
                .map_err(|e| format!("Deserialization error: {}", e))?;
            Ok((output, latency))
        } else {
            let error_msg = String::from_utf8_lossy(&resp_payload);
            Err(format!("Inference error: {}", error_msg))
        }
    }

    #[cfg(windows)]
    {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        stream
            .write_all(&model_id_bytes)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
        stream
            .write_all(&op_code_bytes)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
        stream
            .write_all(&payload_size_bytes)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
        stream
            .write_all(&payload)
            .await
            .map_err(|e| format!("Write error: {}", e))?;

        // Read response header
        let mut status_buf = [0u8; 4];
        stream
            .read_exact(&mut status_buf)
            .await
            .map_err(|e| format!("Read error: {}", e))?;
        let status = u32::from_le_bytes(status_buf);

        let mut resp_size_buf = [0u8; 4];
        stream
            .read_exact(&mut resp_size_buf)
            .await
            .map_err(|e| format!("Read error: {}", e))?;
        let resp_size = u32::from_le_bytes(resp_size_buf);

        // Read response payload
        let mut resp_payload = vec![0u8; resp_size as usize];
        stream
            .read_exact(&mut resp_payload)
            .await
            .map_err(|e| format!("Read error: {}", e))?;

        let latency = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms

        if status == STATUS_OK {
            let output: BinaryTensorPacket = bincode::deserialize(&resp_payload)
                .map_err(|e| format!("Deserialization error: {}", e))?;
            Ok((output, latency))
        } else {
            let error_msg = String::from_utf8_lossy(&resp_payload);
            Err(format!("Inference error: {}", error_msg))
        }
    }
}

#[cfg(unix)]
#[tokio::main]
async fn main() {
    run_benchmark().await;
}

#[cfg(windows)]
#[tokio::main]
async fn main() {
    run_benchmark_async().await;
}

#[cfg(unix)]
async fn run_benchmark() {
    println!("🔬 MNIST Inference Latency Benchmark");
    println!("={}", "=".repeat(59));
    println!();

    let socket_path = "/tmp/kapsl.sock";
    let model_id = 0;
    let num_warmup = 5;
    let num_requests = 20;

    println!("Configuration:");
    println!("  Socket: {}", socket_path);
    println!("  Model ID: {}", model_id);
    println!("  Warmup requests: {}", num_warmup);
    println!("  Benchmark requests: {}", num_requests);
    println!();

    // Warmup phase
    println!("🔥 Warming up...");
    for i in 0..num_warmup {
        match UnixStream::connect(socket_path) {
            Ok(mut stream) => match send_inference_request(&mut stream, model_id).await {
                Ok((output, latency)) => {
                    println!(
                        "  Warmup {}/{}: {:.2}ms (output shape: {:?})",
                        i + 1,
                        num_warmup,
                        latency,
                        output.shape
                    );
                }
                Err(e) => {
                    eprintln!("  ❌ Warmup {}/{} failed: {}", i + 1, num_warmup, e);
                }
            },
            Err(e) => {
                eprintln!("❌ Failed to connect: {}", e);
                std::process::exit(1);
            }
        }
    }

    run_benchmark_requests(socket_path, model_id, num_requests).await;
}

#[cfg(windows)]
async fn run_benchmark_async() {
    use tokio::net::windows::named_pipe::{ClientOptions, PipeMode};

    println!("🔬 MNIST Inference Latency Benchmark");
    println!("={}", "=".repeat(59));
    println!();

    let socket_path = r"\\.\pipe\kapsl";
    let model_id = 0;
    let num_warmup = 5;
    let num_requests = 20;

    println!("Configuration:");
    println!("  Socket: {}", socket_path);
    println!("  Model ID: {}", model_id);
    println!("  Warmup requests: {}", num_warmup);
    println!("  Benchmark requests: {}", num_requests);
    println!();

    // Warmup phase
    println!("🔥 Warming up...");
    for i in 0..num_warmup {
        match ClientOptions::new()
            .pipe_mode(PipeMode::Byte)
            .open(socket_path)
        {
            Ok(mut stream) => match send_inference_request(&mut stream, model_id).await {
                Ok((output, latency)) => {
                    println!(
                        "  Warmup {}/{}: {:.2}ms (output shape: {:?})",
                        i + 1,
                        num_warmup,
                        latency,
                        output.shape
                    );
                }
                Err(e) => {
                    eprintln!("  ❌ Warmup {}/{} failed: {}", i + 1, num_warmup, e);
                }
            },
            Err(e) => {
                eprintln!("❌ Failed to connect: {}", e);
                std::process::exit(1);
            }
        }
    }

    run_benchmark_requests_async(socket_path, model_id, num_requests).await;
}

#[cfg(unix)]
async fn run_benchmark_requests(socket_path: &str, model_id: u32, num_requests: i32) {
    println!();
    println!("📊 Running benchmark...");

    let mut latencies = Vec::new();
    let mut successes = 0;

    for i in 0..num_requests {
        match UnixStream::connect(socket_path) {
            Ok(mut stream) => match send_inference_request(&mut stream, model_id).await {
                Ok((output, latency)) => {
                    latencies.push(latency);
                    successes += 1;
                    println!(
                        "  Request {:2}/{}: {:.2}ms ✅ (output: {:?})",
                        i + 1,
                        num_requests,
                        latency,
                        output.shape
                    );
                }
                Err(e) => {
                    eprintln!("  Request {:2}/{}: Failed - {}", i + 1, num_requests, e);
                }
            },
            Err(e) => {
                eprintln!(
                    "  Request {:2}/{}: Connection failed - {}",
                    i + 1,
                    num_requests,
                    e
                );
            }
        }
    }

    print_results(num_requests, successes, latencies);
}

#[cfg(windows)]
async fn run_benchmark_requests_async(socket_path: &str, model_id: u32, num_requests: i32) {
    use tokio::net::windows::named_pipe::{ClientOptions, PipeMode};

    println!();
    println!("📊 Running benchmark...");

    let mut latencies = Vec::new();
    let mut successes = 0;

    for i in 0..num_requests {
        match ClientOptions::new()
            .pipe_mode(PipeMode::Byte)
            .open(socket_path)
        {
            Ok(mut stream) => match send_inference_request(&mut stream, model_id).await {
                Ok((output, latency)) => {
                    latencies.push(latency);
                    successes += 1;
                    println!(
                        "  Request {:2}/{}: {:.2}ms ✅ (output: {:?})",
                        i + 1,
                        num_requests,
                        latency,
                        output.shape
                    );
                }
                Err(e) => {
                    eprintln!("  Request {:2}/{}: Failed - {}", i + 1, num_requests, e);
                }
            },
            Err(e) => {
                eprintln!(
                    "  Request {:2}/{}: Connection failed - {}",
                    i + 1,
                    num_requests,
                    e
                );
            }
        }
    }

    print_results(num_requests, successes, latencies);
}

fn print_results(num_requests: i32, successes: i32, latencies: Vec<f64>) {
    println!();
    println!("={}", "=".repeat(59));
    println!("Results:");
    println!("-{}", "-".repeat(59));

    if !latencies.is_empty() {
        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let min = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted = latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
        let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

        println!("Total requests: {}", num_requests);
        println!("Successful: {}", successes);
        println!("Failed: {}", num_requests - successes);
        println!();
        println!("Latency Statistics:");
        println!("  Average: {:.2}ms", avg);
        println!("  Median:  {:.2}ms", median);
        println!("  Min:     {:.2}ms", min);
        println!("  Max:     {:.2}ms", max);
        println!("  P95:     {:.2}ms", p95);
        println!("  P99:     {:.2}ms", p99);
        println!();

        // Throughput calculation
        let total_time = latencies.iter().sum::<f64>() / 1000.0; // Convert to seconds
        let throughput = num_requests as f64 / total_time;
        println!("Throughput: {:.1} inferences/sec", throughput);
        println!();

        // Performance rating
        if avg < 5.0 {
            println!("✅ Excellent performance (<5ms)");
        } else if avg < 10.0 {
            println!("✅ Very good performance (<10ms)");
        } else if avg < 50.0 {
            println!("✅ Good performance (<50ms)");
        } else if avg < 100.0 {
            println!("⚠️  Acceptable performance (<100ms)");
        } else {
            println!("❌ Poor performance (>100ms)");
        }
    } else {
        println!("❌ No successful requests");
    }

    println!("={}", "=".repeat(59));
}
