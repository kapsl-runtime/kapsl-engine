use async_trait::async_trait;
use kapsl_engine_api::BinaryTensorPacket;
use kapsl_transport::connection_pool::{ConnectionFactory, ConnectionPool, PoolConfig};
use kapsl_transport::{RequestMetadata, ResponseMetadata, TransportClient, TransportError};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[cfg(unix)]
use tokio::net::UnixStream;

#[cfg(windows)]
use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient};

// Platform-specific connection type alias
#[cfg(unix)]
type IpcConnection = UnixStream;

#[cfg(windows)]
type IpcConnection = NamedPipeClient;

/// Factory for creating IPC connections (Unix Domain Sockets on Unix, Named Pipes on Windows)
pub struct IpcConnectionFactory {
    socket_path: String,
}

impl IpcConnectionFactory {
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }
}

#[async_trait]
impl ConnectionFactory for IpcConnectionFactory {
    #[cfg(unix)]
    type Connection = UnixStream;
    #[cfg(windows)]
    type Connection = NamedPipeClient;
    type Error = std::io::Error;

    async fn connect(&self) -> Result<Self::Connection, Self::Error> {
        #[cfg(unix)]
        return UnixStream::connect(&self.socket_path).await;

        #[cfg(windows)]
        return ClientOptions::new().open(&self.socket_path);
    }

    async fn is_valid(&self, _conn: &Self::Connection) -> bool {
        // Similar to TCP, hard to check validity without I/O.
        true
    }
}

/// IPC Client implementation using connection pooling
pub struct IpcClient {
    pool: ConnectionPool<IpcConnection, IpcConnectionFactory>,
}

impl IpcClient {
    pub fn new(socket_path: String, pool_config: PoolConfig) -> Self {
        let factory = IpcConnectionFactory::new(socket_path);
        let pool = ConnectionPool::new(pool_config, factory);
        Self { pool }
    }
}

#[async_trait]
impl TransportClient for IpcClient {
    fn transport_type(&self) -> &'static str {
        "ipc"
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
        let metadata_len = metadata_bytes.len() as u32;
        let input_len = input_bytes.len() as u32;

        conn.write_u32(metadata_len).await?;
        conn.write_all(&metadata_bytes).await?;
        conn.write_u32(input_len).await?;
        conn.write_all(&input_bytes).await?;
        conn.flush().await?;

        // Read response
        let resp_metadata_len = conn.read_u32().await?;
        let mut resp_metadata_buf = vec![0u8; resp_metadata_len as usize];
        conn.read_exact(&mut resp_metadata_buf).await?;

        let resp_metadata: ResponseMetadata = bincode::deserialize(&resp_metadata_buf)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;

        if !resp_metadata.is_success() {
            let output_len = conn.read_u32().await?;
            let mut output_buf = vec![0u8; output_len as usize];
            conn.read_exact(&mut output_buf).await?;
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
