#![allow(clippy::useless_conversion)]

use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, TensorDtype};
use kapsl_ipc::{
    RequestHeader, ResponseHeader, OP_INFER, OP_INFER_STREAM, STATUS_OK, STATUS_STREAM_CHUNK,
    STATUS_STREAM_END,
};
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::TcpStream;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::str::FromStr;
use std::sync::Mutex;
pub mod hybrid_client;
pub mod shm_client;

pub use hybrid_client::KapslHybridClient;
pub use shm_client::KapslShmClient;

const DEFAULT_MAX_POOL_SIZE: usize = 8;
#[cfg(unix)]
const DEFAULT_SOCKET_ENDPOINT: &str = "/tmp/kapsl.sock";
#[cfg(windows)]
const DEFAULT_SOCKET_ENDPOINT: &str = r"\\.\pipe\kapsl";
const DEFAULT_TCP_HOST: &str = "127.0.0.1";
const DEFAULT_TCP_PORT: u16 = 9096;

trait ReadWriteConnection: Read + Write + Send {}
impl<T: Read + Write + Send> ReadWriteConnection for T {}
type ClientConnection = Box<dyn ReadWriteConnection>;

enum TransportProtocol {
    Socket,
    Tcp,
    Pipe,
}

enum ConnectionTarget {
    #[cfg(unix)]
    UnixSocket(String),
    #[cfg(windows)]
    NamedPipe(String),
    Tcp(String),
}

enum ClientError {
    Io(std::io::Error),
    InvalidEndpoint(String),
    InvalidDtype(String),
    Serialization(String),
    Server(String),
}

impl From<std::io::Error> for ClientError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ClientError> for PyErr {
    fn from(value: ClientError) -> Self {
        match value {
            ClientError::Io(err) => {
                PyErr::new::<pyo3::exceptions::PyConnectionError, _>(err.to_string())
            }
            ClientError::InvalidEndpoint(msg) | ClientError::InvalidDtype(msg) => {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
            }
            ClientError::Serialization(msg) | ClientError::Server(msg) => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg)
            }
        }
    }
}

impl TransportProtocol {
    fn parse(raw: &str) -> Result<Self, ClientError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "socket" | "unix" | "local" => Ok(Self::Socket),
            "tcp" => Ok(Self::Tcp),
            "pipe" | "named_pipe" | "named-pipe" => Ok(Self::Pipe),
            other => Err(ClientError::InvalidEndpoint(format!(
                "Unsupported protocol '{}'. Use one of: socket, tcp, pipe",
                other
            ))),
        }
    }
}

impl ConnectionTarget {
    fn default_local() -> Self {
        #[cfg(unix)]
        {
            Self::UnixSocket(DEFAULT_SOCKET_ENDPOINT.to_string())
        }
        #[cfg(windows)]
        {
            Self::NamedPipe(DEFAULT_SOCKET_ENDPOINT.to_string())
        }
    }

    fn protocol_name(&self) -> &'static str {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(_) => "socket",
            #[cfg(windows)]
            Self::NamedPipe(_) => "pipe",
            Self::Tcp(_) => "tcp",
        }
    }

    fn endpoint_display(&self) -> String {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(path) => format!("unix://{}", path),
            #[cfg(windows)]
            Self::NamedPipe(path) => format!("pipe://{}", path),
            Self::Tcp(addr) => format!("tcp://{}", addr),
        }
    }

    #[cfg(windows)]
    fn normalize_pipe_path(path: &str) -> String {
        if path.starts_with(r"\\.\pipe\") {
            path.to_string()
        } else {
            format!(r"\\.\pipe\{}", path)
        }
    }

    fn from_endpoint(endpoint: &str) -> Result<Self, ClientError> {
        let endpoint = endpoint.trim();
        if endpoint.is_empty() {
            return Err(ClientError::InvalidEndpoint(
                "Endpoint cannot be empty".to_string(),
            ));
        }

        if let Some(addr) = endpoint.strip_prefix("tcp://") {
            if addr.is_empty() {
                return Err(ClientError::InvalidEndpoint(
                    "tcp:// endpoint must include host:port".to_string(),
                ));
            }
            return Ok(Self::Tcp(addr.to_string()));
        }

        #[cfg(unix)]
        {
            if let Some(path) = endpoint.strip_prefix("unix://") {
                if path.is_empty() {
                    return Err(ClientError::InvalidEndpoint(
                        "unix:// endpoint must include a socket path".to_string(),
                    ));
                }
                return Ok(Self::UnixSocket(path.to_string()));
            }
            return Ok(Self::UnixSocket(endpoint.to_string()));
        }

        #[cfg(windows)]
        {
            if endpoint.starts_with("unix://") {
                return Err(ClientError::InvalidEndpoint(
                    "unix:// endpoints are only supported on Unix".to_string(),
                ));
            }

            if let Some(pipe_path) = endpoint.strip_prefix("pipe://") {
                if pipe_path.is_empty() {
                    return Err(ClientError::InvalidEndpoint(
                        "pipe:// endpoint must include a pipe name".to_string(),
                    ));
                }
                return Ok(Self::NamedPipe(Self::normalize_pipe_path(pipe_path)));
            }

            return Ok(Self::NamedPipe(endpoint.to_string()));
        }
    }

    fn from_options(
        endpoint: Option<&str>,
        protocol: Option<&str>,
        host: Option<&str>,
        port: Option<u16>,
        socket_path: Option<&str>,
        pipe_name: Option<&str>,
    ) -> Result<Self, ClientError> {
        fn normalize_opt<'a>(value: Option<&'a str>) -> Option<&'a str> {
            value.and_then(|raw| {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
        }

        let endpoint = normalize_opt(endpoint);
        let protocol = normalize_opt(protocol);
        let host = normalize_opt(host);
        let socket_path = normalize_opt(socket_path);
        let pipe_name = normalize_opt(pipe_name);

        if let Some(protocol_raw) = protocol {
            let parsed = TransportProtocol::parse(protocol_raw)?;
            return match parsed {
                TransportProtocol::Tcp => {
                    if socket_path.is_some() || pipe_name.is_some() {
                        return Err(ClientError::InvalidEndpoint(
                            "protocol='tcp' cannot be combined with socket_path or pipe_name"
                                .to_string(),
                        ));
                    }

                    if let Some(endpoint_value) = endpoint {
                        if host.is_some() || port.is_some() {
                            return Err(ClientError::InvalidEndpoint(
                                "When endpoint is provided, do not also pass host/port".to_string(),
                            ));
                        }
                        if let Some(addr) = endpoint_value.strip_prefix("tcp://") {
                            if addr.is_empty() {
                                return Err(ClientError::InvalidEndpoint(
                                    "tcp:// endpoint must include host:port".to_string(),
                                ));
                            }
                            return Ok(Self::Tcp(addr.to_string()));
                        }
                        if endpoint_value.contains("://") {
                            return Err(ClientError::InvalidEndpoint(
                                "protocol='tcp' expects endpoint as host:port or tcp://host:port"
                                    .to_string(),
                            ));
                        }
                        return Ok(Self::Tcp(endpoint_value.to_string()));
                    }

                    let host = host.unwrap_or(DEFAULT_TCP_HOST);
                    let port = port.unwrap_or(DEFAULT_TCP_PORT);
                    Ok(Self::Tcp(format!("{}:{}", host, port)))
                }
                TransportProtocol::Socket => {
                    if host.is_some() || port.is_some() || pipe_name.is_some() {
                        return Err(ClientError::InvalidEndpoint(
                            "protocol='socket' cannot be combined with host/port/pipe_name"
                                .to_string(),
                        ));
                    }
                    if let Some(path_or_uri) = socket_path.or(endpoint) {
                        return Self::from_endpoint(path_or_uri);
                    }
                    Ok(Self::default_local())
                }
                TransportProtocol::Pipe => {
                    if host.is_some() || port.is_some() || socket_path.is_some() {
                        return Err(ClientError::InvalidEndpoint(
                            "protocol='pipe' cannot be combined with host/port/socket_path"
                                .to_string(),
                        ));
                    }
                    let value = pipe_name.or(endpoint).unwrap_or(DEFAULT_SOCKET_ENDPOINT);
                    #[cfg(unix)]
                    {
                        let _ = value;
                        Err(ClientError::InvalidEndpoint(
                            "protocol='pipe' is only supported on Windows".to_string(),
                        ))
                    }
                    #[cfg(windows)]
                    {
                        if let Some(raw) = value.strip_prefix("pipe://") {
                            if raw.is_empty() {
                                return Err(ClientError::InvalidEndpoint(
                                    "pipe:// endpoint must include a pipe name".to_string(),
                                ));
                            }
                            return Ok(Self::NamedPipe(Self::normalize_pipe_path(raw)));
                        }
                        if value.starts_with("tcp://") || value.starts_with("unix://") {
                            return Err(ClientError::InvalidEndpoint(
                                "protocol='pipe' expects a named pipe path".to_string(),
                            ));
                        }
                        Ok(Self::NamedPipe(Self::normalize_pipe_path(value)))
                    }
                }
            };
        }

        if let Some(endpoint_value) = endpoint {
            return Self::from_endpoint(endpoint_value);
        }

        if host.is_some() || port.is_some() {
            let host = host.unwrap_or(DEFAULT_TCP_HOST);
            let port = port.unwrap_or(DEFAULT_TCP_PORT);
            return Ok(Self::Tcp(format!("{}:{}", host, port)));
        }

        if let Some(path) = socket_path {
            return Self::from_endpoint(path);
        }

        if let Some(name) = pipe_name {
            #[cfg(unix)]
            {
                let _ = name;
                return Err(ClientError::InvalidEndpoint(
                    "pipe_name is only supported on Windows".to_string(),
                ));
            }
            #[cfg(windows)]
            {
                return Ok(Self::NamedPipe(Self::normalize_pipe_path(name)));
            }
        }

        Ok(Self::default_local())
    }
}

#[pyclass]
struct KapslClient {
    target: ConnectionTarget,
    max_pool_size: usize,
    connection_pool: Mutex<VecDeque<ClientConnection>>,
}

impl KapslClient {
    fn connect_stream(&self) -> Result<ClientConnection, ClientError> {
        match &self.target {
            #[cfg(unix)]
            ConnectionTarget::UnixSocket(path) => Ok(Box::new(UnixStream::connect(path)?)),
            #[cfg(windows)]
            ConnectionTarget::NamedPipe(path) => {
                use std::fs::OpenOptions;
                Ok(Box::new(
                    OpenOptions::new().read(true).write(true).open(path)?,
                ))
            }
            ConnectionTarget::Tcp(addr) => {
                let stream = TcpStream::connect(addr)?;
                let _ = stream.set_nodelay(true);
                Ok(Box::new(stream))
            }
        }
    }

    fn checkout_connection(&self) -> Result<ClientConnection, ClientError> {
        if let Ok(mut pool) = self.connection_pool.lock() {
            if let Some(stream) = pool.pop_front() {
                return Ok(stream);
            }
        }
        self.connect_stream()
    }

    fn return_connection(&self, stream: ClientConnection) {
        if self.max_pool_size == 0 {
            return;
        }
        if let Ok(mut pool) = self.connection_pool.lock() {
            if pool.len() < self.max_pool_size {
                pool.push_back(stream);
            }
        }
    }

    fn infer_impl(
        &self,
        stream: &mut dyn ReadWriteConnection,
        model_id: u32,
        shape: &[i64],
        dtype: &str,
        data: &[u8],
    ) -> Result<Vec<u8>, ClientError> {
        let dtype =
            TensorDtype::from_str(dtype).map_err(|e| ClientError::InvalidDtype(e.to_string()))?;
        let input = BinaryTensorPacket {
            shape: shape.to_vec(),
            dtype,
            data: data.to_vec(),
        };

        // Wrap in InferenceRequest (what the server expects)
        let request = InferenceRequest {
            input,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let input_bytes =
            bincode::serialize(&request).map_err(|e| ClientError::Serialization(e.to_string()))?;

        let header = RequestHeader {
            model_id,
            op_code: OP_INFER,
            payload_size: input_bytes.len() as u32,
        };

        // Write header as raw bytes (not bincode)
        stream.write_all(&header.model_id.to_le_bytes())?;
        stream.write_all(&header.op_code.to_le_bytes())?;
        stream.write_all(&header.payload_size.to_le_bytes())?;
        stream.write_all(&input_bytes)?;
        stream.flush()?; // Ensure data is sent

        // Read response
        let mut header_buf = [0u8; 8]; // 2 * u32 = 8 bytes
        stream.read_exact(&mut header_buf)?;

        let resp_header: ResponseHeader = bincode::deserialize(&header_buf)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;

        let mut payload = vec![0u8; resp_header.payload_size as usize];
        stream.read_exact(&mut payload)?;

        if resp_header.status != STATUS_OK {
            let error_msg = String::from_utf8_lossy(&payload);
            return Err(ClientError::Server(error_msg.to_string()));
        }

        // Deserialize payload to BinaryTensorPacket to get data
        let output: BinaryTensorPacket = bincode::deserialize(&payload)
            .map_err(|e| ClientError::Serialization(e.to_string()))?;

        Ok(output.data)
    }
}

#[pymethods]
#[allow(clippy::useless_conversion)]
impl KapslClient {
    #[new]
    #[pyo3(signature = (
        endpoint = None,
        *,
        protocol = None,
        host = None,
        port = None,
        socket_path = None,
        pipe_name = None,
        max_pool_size = DEFAULT_MAX_POOL_SIZE
    ))]
    fn new(
        endpoint: Option<String>,
        protocol: Option<String>,
        host: Option<String>,
        port: Option<u16>,
        socket_path: Option<String>,
        pipe_name: Option<String>,
        max_pool_size: usize,
    ) -> PyResult<Self> {
        let target = ConnectionTarget::from_options(
            endpoint.as_deref(),
            protocol.as_deref(),
            host.as_deref(),
            port,
            socket_path.as_deref(),
            pipe_name.as_deref(),
        )
        .map_err(PyErr::from)?;
        Ok(Self {
            target,
            max_pool_size,
            connection_pool: Mutex::new(VecDeque::new()),
        })
    }

    fn protocol(&self) -> String {
        self.target.protocol_name().to_string()
    }

    fn endpoint(&self) -> String {
        self.target.endpoint_display()
    }

    fn infer(
        &self,
        model_id: u32,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
    ) -> PyResult<Vec<u8>> {
        let mut stream = self.checkout_connection().map_err(PyErr::from)?;
        match self.infer_impl(&mut stream, model_id, &shape, &dtype, &data) {
            Ok(output) => {
                self.return_connection(stream);
                Ok(output)
            }
            Err(ClientError::Io(_)) => {
                // Connection likely stale; retry once with a fresh socket.
                let mut fresh = self.connect_stream().map_err(PyErr::from)?;
                match self.infer_impl(&mut fresh, model_id, &shape, &dtype, &data) {
                    Ok(output) => {
                        self.return_connection(fresh);
                        Ok(output)
                    }
                    Err(err) => Err(err.into()),
                }
            }
            Err(err) => {
                self.return_connection(stream);
                Err(err.into())
            }
        }
    }

    fn infer_stream(
        &self,
        model_id: u32,
        shape: Vec<i64>,
        dtype: String,
        data: Vec<u8>,
    ) -> PyResult<StreamIterator> {
        let mut stream = self.connect_stream().map_err(PyErr::from)?;

        // Send request
        let dtype = TensorDtype::from_str(&dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let input = BinaryTensorPacket { shape, dtype, data };
        let request = InferenceRequest {
            input,
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };
        let input_bytes = bincode::serialize(&request)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let header = RequestHeader {
            model_id,
            op_code: OP_INFER_STREAM,
            payload_size: input_bytes.len() as u32,
        };

        stream.write_all(&header.model_id.to_le_bytes())?;
        stream.write_all(&header.op_code.to_le_bytes())?;
        stream.write_all(&header.payload_size.to_le_bytes())?;
        stream.write_all(&input_bytes)?;
        stream.flush()?;

        Ok(StreamIterator { stream })
    }
}

#[pyclass]
struct StreamIterator {
    stream: ClientConnection,
}

#[pymethods]
impl StreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Vec<u8>>> {
        let stream = &mut slf.stream;

        // Read header
        let mut header_buf = [0u8; 8];
        match stream.read_exact(&mut header_buf) {
            Ok(_) => {}
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e.to_string(),
                ))
            }
        }

        let resp_header: ResponseHeader = bincode::deserialize(&header_buf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        if resp_header.payload_size > 0 {
            let mut payload = vec![0u8; resp_header.payload_size as usize];
            stream
                .read_exact(&mut payload)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            if resp_header.status == STATUS_STREAM_CHUNK {
                // Deserialize payload to BinaryTensorPacket
                let packet: BinaryTensorPacket = bincode::deserialize(&payload).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                return Ok(Some(packet.data));
            } else if resp_header.status == kapsl_ipc::STATUS_ERR {
                let error_msg = String::from_utf8_lossy(&payload);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    error_msg.to_string(),
                ));
            }
        }

        if resp_header.status == STATUS_STREAM_END {
            return Ok(None);
        }

        // If we get here, it's an unexpected status or empty chunk that isn't END
        Ok(None)
    }
}

#[pymodule]
fn kapsl_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KapslClient>()?;
    m.add_class::<KapslShmClient>()?;
    m.add_class::<KapslHybridClient>()?;
    Ok(())
}
