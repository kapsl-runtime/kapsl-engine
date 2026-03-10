use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
use kapsl_ipc::{RequestHeader, ResponseHeader, OP_INFER, STATUS_OK};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::net::UnixStream;

#[cfg(unix)]
fn main() {
    run_client();
}

#[cfg(windows)]
#[tokio::main]
async fn main() {
    run_client_async().await;
}

#[cfg(unix)]
fn run_client() {
    let socket_path = "/tmp/kapsl_test.sock";
    // Usually Deepseek would be the first model loaded if we ran kapsl with it.
    // If multiple models are loaded, their IDs start from 0.
    let model_id = 0;

    println!("[Deepseek Client] Connecting to {}...", socket_path);
    let mut stream = UnixStream::connect(socket_path).expect("Failed to connect");
    println!("[Deepseek Client] Connected!");

    // Input: "Hello world" tokens (approx)
    // Shape: [1, 2] -> Batch 1, Sequence Length 2
    let shape = vec![1, 2];

    // Using int64 for input_ids as expected by most ONNX LLMs
    let input_ids: Vec<i64> = vec![15496, 1234];

    // Serialize data to bytes
    let mut data = Vec::new();
    for &v in &input_ids {
        data.extend_from_slice(&v.to_ne_bytes());
    }

    let input = BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Int64,
        data,
    };

    let request = InferenceRequest {
        input,
        additional_inputs: Vec::new(),
        session_id: None,
        metadata: None,
        cancellation: None,
    };

    let input_bytes = bincode::serialize(&request).unwrap();

    let header = RequestHeader {
        model_id,
        op_code: OP_INFER,
        payload_size: input_bytes.len() as u32,
    };

    let header_bytes = bincode::serialize(&header).unwrap();

    println!("[Deepseek Client] Sending inference request...");
    let start = std::time::Instant::now();
    stream.write_all(&header_bytes).unwrap();
    stream.write_all(&input_bytes).unwrap();

    // Read response
    let mut header_buf = [0u8; 8];
    if let Err(e) = stream.read_exact(&mut header_buf) {
        println!("[Deepseek Client] Failed to read response header: {}", e);
        return;
    }

    let resp_header: ResponseHeader = bincode::deserialize(&header_buf).unwrap();

    if resp_header.status == STATUS_OK {
        let mut payload = vec![0u8; resp_header.payload_size as usize];
        stream.read_exact(&mut payload).unwrap();

        let output: BinaryTensorPacket = bincode::deserialize(&payload).unwrap();
        let duration = start.elapsed();
        println!(
            "[Deepseek Client] ✓ Response received in {:?} | Shape: {:?} | DataType: {}",
            duration, output.shape, output.dtype
        );

        // Print some output data/logits summary
        if output.dtype == TensorDtype::Float32 {
            let count = output.data.len() / 4;
            println!("[Deepseek Client] Output element count: {}", count);
            // Print first few floats
            let mut floats = Vec::new();
            for i in 0..std::cmp::min(10, count) {
                let start = i * 4;
                let bytes: [u8; 4] = output.data[start..start + 4].try_into().unwrap();
                floats.push(f32::from_ne_bytes(bytes));
            }
            println!("[Deepseek Client] First 10 outputs: {:?}", floats);
        }
    } else {
        let mut payload = vec![0u8; resp_header.payload_size as usize];
        stream.read_exact(&mut payload).unwrap();
        let error_msg = String::from_utf8(payload).unwrap();
        println!("[Deepseek Client] ✗ Error: {}", error_msg);
    }
}

#[cfg(windows)]
async fn run_client_async() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::windows::named_pipe::{ClientOptions, PipeMode};

    let socket_path = r"\\.\pipe\kapsl";
    let model_id = 0;

    println!("[Deepseek Client] Connecting to {}...", socket_path);
    let mut stream = match ClientOptions::new()
        .pipe_mode(PipeMode::Byte)
        .open(socket_path)
    {
        Ok(stream) => stream,
        Err(e) => {
            println!("[Deepseek Client] Failed to connect: {}", e);
            return;
        }
    };
    println!("[Deepseek Client] Connected!");

    // Input: "Hello world" tokens (approx)
    // Shape: [1, 2] -> Batch 1, Sequence Length 2
    let shape = vec![1, 2];
    let input_ids: Vec<i64> = vec![15496, 1234];

    let mut data = Vec::new();
    for &v in &input_ids {
        data.extend_from_slice(&v.to_le_bytes());
    }

    let input = BinaryTensorPacket {
        shape,
        dtype: TensorDtype::Int64,
        data,
    };

    let request = InferenceRequest {
        input,
        additional_inputs: Vec::new(),
        session_id: None,
        metadata: None,
        cancellation: None,
    };

    let input_bytes = bincode::serialize(&request).unwrap();

    let header = RequestHeader {
        model_id,
        op_code: OP_INFER,
        payload_size: input_bytes.len() as u32,
    };

    println!("[Deepseek Client] Sending inference request...");
    let start = std::time::Instant::now();

    if let Err(e) = stream.write_all(&header.model_id.to_le_bytes()).await {
        println!("[Deepseek Client] Failed to send model_id: {}", e);
        return;
    }
    if let Err(e) = stream.write_all(&header.op_code.to_le_bytes()).await {
        println!("[Deepseek Client] Failed to send op_code: {}", e);
        return;
    }
    if let Err(e) = stream.write_all(&header.payload_size.to_le_bytes()).await {
        println!("[Deepseek Client] Failed to send payload_size: {}", e);
        return;
    }
    if let Err(e) = stream.write_all(&input_bytes).await {
        println!("[Deepseek Client] Failed to send payload: {}", e);
        return;
    }
    if let Err(e) = stream.flush().await {
        println!("[Deepseek Client] Failed to flush request: {}", e);
        return;
    }

    let mut status_buf = [0u8; 4];
    if let Err(e) = stream.read_exact(&mut status_buf).await {
        println!("[Deepseek Client] Failed to read response status: {}", e);
        return;
    }
    let mut size_buf = [0u8; 4];
    if let Err(e) = stream.read_exact(&mut size_buf).await {
        println!(
            "[Deepseek Client] Failed to read response payload size: {}",
            e
        );
        return;
    }

    let resp_header = ResponseHeader {
        status: u32::from_le_bytes(status_buf),
        payload_size: u32::from_le_bytes(size_buf),
    };

    let mut payload = vec![0u8; resp_header.payload_size as usize];
    if let Err(e) = stream.read_exact(&mut payload).await {
        println!("[Deepseek Client] Failed to read response payload: {}", e);
        return;
    }

    if resp_header.status == STATUS_OK {
        let output: BinaryTensorPacket = match bincode::deserialize(&payload) {
            Ok(output) => output,
            Err(e) => {
                println!("[Deepseek Client] Failed to decode output: {}", e);
                return;
            }
        };

        let duration = start.elapsed();
        println!(
            "[Deepseek Client] ✓ Response received in {:?} | Shape: {:?} | DataType: {}",
            duration, output.shape, output.dtype
        );

        if output.dtype == TensorDtype::Float32 {
            let count = output.data.len() / 4;
            println!("[Deepseek Client] Output element count: {}", count);
            let mut floats = Vec::new();
            for i in 0..std::cmp::min(10, count) {
                let start = i * 4;
                let bytes: [u8; 4] = output.data[start..start + 4].try_into().unwrap();
                floats.push(f32::from_le_bytes(bytes));
            }
            println!("[Deepseek Client] First 10 outputs: {:?}", floats);
        }
    } else {
        let error_msg = String::from_utf8_lossy(&payload);
        println!("[Deepseek Client] ✗ Error: {}", error_msg);
    }
}
