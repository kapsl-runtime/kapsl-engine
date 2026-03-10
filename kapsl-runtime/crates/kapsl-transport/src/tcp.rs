use crate::connection_pool::{ConnectionFactory, ConnectionPool, PoolConfig};
use crate::{RequestMetadata, ResponseMetadata, TransportClient, TransportError};
use async_trait::async_trait;
use kapsl_engine_api::BinaryTensorPacket;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Factory for creating TCP connections
pub struct TcpConnectionFactory {
    host: String,
    port: u16,
}

impl TcpConnectionFactory {
    pub fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }
}

#[async_trait]
impl ConnectionFactory for TcpConnectionFactory {
    type Connection = TcpStream;
    type Error = std::io::Error;

    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        let addr = format!("{}:{}", self.host, self.port);
        TcpStream::connect(addr).await
    }

    async fn is_valid(&self, _conn: &Self::Connection) -> bool {
        // Simple check: we can't easily check if a TCP stream is closed without reading/writing.
        // For now, assume it's valid if we have it.
        // A more robust check would be to peek, but TcpStream doesn't support peek easily.
        true
    }
}

/// TCP Client implementation using connection pooling
pub struct TcpClient {
    pool: ConnectionPool<TcpStream, TcpConnectionFactory>,
}

impl TcpClient {
    pub fn new(host: String, port: u16, pool_config: PoolConfig) -> Self {
        let factory = TcpConnectionFactory::new(host, port);
        let pool = ConnectionPool::new(pool_config, factory);
        Self { pool }
    }
}

#[async_trait]
impl TransportClient for TcpClient {
    fn transport_type(&self) -> &'static str {
        "tcp"
    }

    async fn infer(
        &self,
        model_id: u32,
        input: BinaryTensorPacket,
    ) -> Result<BinaryTensorPacket, TransportError> {
        // Get a connection from the pool
        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| TransportError::Connection(e.to_string()))?;

        // Create request metadata
        // For now, generate a random request ID or use a counter if we had one.
        // Since TransportClient doesn't have state for ID, use timestamp or random.
        // Using timestamp nanos as ID for simplicity here.
        let request_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let metadata = RequestMetadata::new(request_id, model_id, 0, false);

        // Serialize metadata and input
        let metadata_bytes = bincode::serialize(&metadata)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;
        let input_bytes =
            bincode::serialize(&input).map_err(|e| TransportError::Serialization(e.to_string()))?;

        // Send request
        // Format: [metadata_len: u32][metadata_bytes][input_len: u32][input_bytes]
        let metadata_len = metadata_bytes.len() as u32;
        let input_len = input_bytes.len() as u32;

        conn.write_u32(metadata_len).await?;
        conn.write_all(&metadata_bytes).await?;
        conn.write_u32(input_len).await?;
        conn.write_all(&input_bytes).await?;
        conn.flush().await?;

        // Read response
        // Format: [metadata_len: u32][metadata_bytes][output_len: u32][output_bytes]
        let resp_metadata_len = conn.read_u32().await?;
        let mut resp_metadata_buf = vec![0u8; resp_metadata_len as usize];
        conn.read_exact(&mut resp_metadata_buf).await?;

        let resp_metadata: ResponseMetadata = bincode::deserialize(&resp_metadata_buf)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;

        if !resp_metadata.is_success() {
            // If error, maybe we should read the error message?
            // The protocol might return error string in output bytes?
            // Assuming output bytes contain error string if status != 0
            let output_len = conn.read_u32().await?;
            let mut output_buf = vec![0u8; output_len as usize];
            conn.read_exact(&mut output_buf).await?;
            // Try to deserialize as string, or just return generic error
            return Err(TransportError::ServerError(format!(
                "Remote error (status {})",
                resp_metadata.status
            )));
        }

        let output_len = conn.read_u32().await?;
        let mut output_buf = vec![0u8; output_len as usize];
        conn.read_exact(&mut output_buf).await?;

        let output: BinaryTensorPacket = bincode::deserialize(&output_buf)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;

        Ok(output)
    }
}
