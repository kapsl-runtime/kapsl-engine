//! Persistent HTTP clients and bounded SSE decoding for managed vLLM.
//!
//! The managed engine still has a synchronous one-shot inference contract, so
//! this bridge owns both a pooled blocking client for that compatibility path
//! and a pooled asynchronous client for readiness probes and streaming.  SSE
//! decoding directly polls the upstream Hyper body: there is deliberately no
//! intermediate task, channel, or thread, so downstream demand is the
//! backpressure mechanism.

use futures::Stream;
use hyper::body::{Bytes, HttpBody};
use hyper::client::connect::{Connected, Connection};
use hyper::client::HttpConnector;
use hyper::header::{HeaderMap, CONTENT_TYPE};
use hyper::http::uri::PathAndQuery;
use hyper::{Body, Client, Method, Request, StatusCode, Uri};
use kapsl_engine_api::CancellationToken;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio::time::Instant;

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);
const DEFAULT_MAX_SSE_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_ERROR_BODY_BYTES: usize = 1024 * 1024;
const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Clone)]
pub(crate) struct ManagedVllmBridgeTelemetry {
    metrics: kapsl_monitor::metrics::KapslMetrics,
    model: String,
    replica: String,
}

impl ManagedVllmBridgeTelemetry {
    pub(crate) fn new(
        metrics: kapsl_monitor::metrics::KapslMetrics,
        model: String,
        replica: impl ToString,
    ) -> Arc<Self> {
        Arc::new(Self {
            metrics,
            model,
            replica: replica.to_string(),
        })
    }

    fn request(&self, mode: &'static str, streaming: bool) {
        self.metrics
            .managed_vllm
            .bridge_requests_total
            .with_label_values(&[
                self.model.as_str(),
                self.replica.as_str(),
                mode,
                if streaming { "true" } else { "false" },
            ])
            .inc();
    }

    fn stage(&self, mode: &'static str, stage: &'static str, elapsed: Duration) {
        self.metrics
            .managed_vllm
            .bridge_stage_seconds
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), mode, stage])
            .observe(elapsed.as_secs_f64());
    }

    fn relayed(&self, mode: &'static str, bytes: usize) {
        let labels = &[self.model.as_str(), self.replica.as_str(), mode];
        self.metrics
            .managed_vllm
            .bridge_relayed_bytes_total
            .with_label_values(labels)
            .inc_by(u64::try_from(bytes).unwrap_or(u64::MAX));
        self.metrics
            .managed_vllm
            .bridge_relayed_chunks_total
            .with_label_values(labels)
            .inc();
    }

    fn stream_started(&self, mode: &'static str) {
        self.metrics
            .managed_vllm
            .bridge_active_streams
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), mode])
            .inc();
    }

    fn stream_finished(&self, mode: &'static str) {
        self.metrics
            .managed_vllm
            .bridge_active_streams
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), mode])
            .dec();
    }

    fn cancellation(&self, mode: &'static str) {
        self.metrics
            .managed_vllm
            .bridge_cancellations_total
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), mode])
            .inc();
    }

    fn error(&self, mode: &'static str, error: &ManagedVllmBridgeError) {
        self.metrics
            .managed_vllm
            .bridge_upstream_errors_total
            .with_label_values(&[
                self.model.as_str(),
                self.replica.as_str(),
                mode,
                bridge_error_kind(error),
            ])
            .inc();
    }

    fn connection_attempt(&self) {
        self.metrics
            .managed_vllm
            .bridge_connection_attempts_total
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), "async"])
            .inc();
    }

    fn connection_opened(&self) {
        self.metrics
            .managed_vllm
            .bridge_open_connections
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), "async"])
            .inc();
    }

    fn connection_closed(&self) {
        self.metrics
            .managed_vllm
            .bridge_open_connections
            .with_label_values(&[self.model.as_str(), self.replica.as_str(), "async"])
            .dec();
    }
}

fn bridge_error_kind(error: &ManagedVllmBridgeError) -> &'static str {
    match error {
        ManagedVllmBridgeError::HeaderTimeout => "header_timeout",
        ManagedVllmBridgeError::IdleBodyTimeout => "idle_timeout",
        ManagedVllmBridgeError::TotalTimeout => "total_timeout",
        ManagedVllmBridgeError::Cancelled => "cancelled",
        ManagedVllmBridgeError::UpstreamStatus { .. } => "upstream_status",
        ManagedVllmBridgeError::SseBufferExceeded { .. }
        | ManagedVllmBridgeError::ResponseBodyExceeded { .. } => "limit",
        ManagedVllmBridgeError::Body(_) => "body",
        ManagedVllmBridgeError::Request(_) => "transport",
        ManagedVllmBridgeError::InvalidEndpoint(_)
        | ManagedVllmBridgeError::InvalidPath(_)
        | ManagedVllmBridgeError::InvalidTimeout(_)
        | ManagedVllmBridgeError::InvalidLimit(_)
        | ManagedVllmBridgeError::BuildRequest(_) => "local",
    }
}

#[derive(Clone)]
struct InstrumentedConnector {
    inner: HttpConnector,
    telemetry: Option<Arc<ManagedVllmBridgeTelemetry>>,
}

impl hyper::service::Service<Uri> for InstrumentedConnector {
    type Response = InstrumentedConnection;
    type Error = <HttpConnector as hyper::service::Service<Uri>>::Error;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        if let Some(telemetry) = &self.telemetry {
            telemetry.connection_attempt();
        }
        let connecting = self.inner.call(uri);
        let telemetry = self.telemetry.clone();
        Box::pin(async move {
            let inner = connecting.await?;
            if let Some(telemetry) = &telemetry {
                telemetry.connection_opened();
            }
            Ok(InstrumentedConnection { inner, telemetry })
        })
    }
}

struct InstrumentedConnection {
    inner: TcpStream,
    telemetry: Option<Arc<ManagedVllmBridgeTelemetry>>,
}

impl AsyncRead for InstrumentedConnection {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buffer)
    }
}

impl AsyncWrite for InstrumentedConnection {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        Pin::new(&mut self.inner).poll_write(cx, buffer)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

impl Connection for InstrumentedConnection {
    fn connected(&self) -> Connected {
        self.inner.connected()
    }
}

impl Drop for InstrumentedConnection {
    fn drop(&mut self) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.connection_closed();
        }
    }
}

/// Per-call deadlines. The asynchronous header deadline also covers acquiring
/// or establishing the pooled connection. `idle_body` is reset after every
/// upstream body chunk, while `total` is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ManagedVllmRequestTimeouts {
    pub(crate) headers: Duration,
    pub(crate) idle_body: Duration,
    pub(crate) total: Duration,
}

impl ManagedVllmRequestTimeouts {
    pub(crate) fn new(headers: Duration, idle_body: Duration, total: Duration) -> Self {
        Self {
            headers,
            idle_body,
            total,
        }
    }

    fn validate(self) -> Result<Self, ManagedVllmBridgeError> {
        if self.headers.is_zero() || self.idle_body.is_zero() || self.total.is_zero() {
            return Err(ManagedVllmBridgeError::InvalidTimeout(
                "managed vLLM HTTP timeouts must be non-zero".to_string(),
            ));
        }
        if Instant::now().checked_add(self.total).is_none() {
            return Err(ManagedVllmBridgeError::InvalidTimeout(
                "managed vLLM total HTTP timeout exceeds the platform deadline range".to_string(),
            ));
        }
        Ok(self)
    }
}

impl Default for ManagedVllmRequestTimeouts {
    fn default() -> Self {
        Self {
            headers: Duration::from_secs(30),
            idle_body: Duration::from_secs(30),
            total: Duration::from_secs(600),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ManagedVllmBridgeConfig {
    pub(crate) connect_timeout: Duration,
    pub(crate) pool_idle_timeout: Duration,
    pub(crate) max_sse_buffer_bytes: usize,
    pub(crate) max_error_body_bytes: usize,
}

impl Default for ManagedVllmBridgeConfig {
    fn default() -> Self {
        Self {
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            pool_idle_timeout: DEFAULT_POOL_IDLE_TIMEOUT,
            max_sse_buffer_bytes: DEFAULT_MAX_SSE_BUFFER_BYTES,
            max_error_body_bytes: DEFAULT_MAX_ERROR_BODY_BYTES,
        }
    }
}

impl ManagedVllmBridgeConfig {
    fn validate(self) -> Result<Self, ManagedVllmBridgeError> {
        if self.connect_timeout.is_zero() || self.pool_idle_timeout.is_zero() {
            return Err(ManagedVllmBridgeError::InvalidTimeout(
                "managed vLLM connection and pool timeouts must be non-zero".to_string(),
            ));
        }
        if self.max_sse_buffer_bytes == 0 || self.max_error_body_bytes == 0 {
            return Err(ManagedVllmBridgeError::InvalidLimit(
                "managed vLLM HTTP buffer limits must be non-zero".to_string(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub(crate) enum ManagedVllmBridgeError {
    InvalidEndpoint(String),
    InvalidPath(String),
    InvalidTimeout(String),
    InvalidLimit(String),
    BuildRequest(String),
    Request(String),
    HeaderTimeout,
    IdleBodyTimeout,
    TotalTimeout,
    Cancelled,
    Body(String),
    SseBufferExceeded { limit: usize },
    ResponseBodyExceeded { limit: usize },
    UpstreamStatus { status: StatusCode, body: Vec<u8> },
}

#[cfg(test)]
impl ManagedVllmBridgeError {
    fn upstream_status(&self) -> Option<StatusCode> {
        match self {
            Self::UpstreamStatus { status, .. } => Some(*status),
            _ => None,
        }
    }

    fn upstream_body(&self) -> Option<&[u8]> {
        match self {
            Self::UpstreamStatus { body, .. } => Some(body),
            _ => None,
        }
    }
}

impl fmt::Display for ManagedVllmBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEndpoint(message)
            | Self::InvalidPath(message)
            | Self::InvalidTimeout(message)
            | Self::InvalidLimit(message)
            | Self::BuildRequest(message)
            | Self::Request(message)
            | Self::Body(message) => formatter.write_str(message),
            Self::HeaderTimeout => formatter.write_str("managed vLLM response headers timed out"),
            Self::IdleBodyTimeout => formatter.write_str("managed vLLM response body became idle"),
            Self::TotalTimeout => formatter.write_str("managed vLLM request timed out"),
            Self::Cancelled => formatter.write_str("managed vLLM request was cancelled"),
            Self::SseBufferExceeded { limit } => write!(
                formatter,
                "managed vLLM SSE event exceeded the {limit}-byte buffer limit"
            ),
            Self::ResponseBodyExceeded { limit } => write!(
                formatter,
                "managed vLLM response body exceeded the {limit}-byte buffer limit"
            ),
            Self::UpstreamStatus { status, .. } => {
                write!(formatter, "managed vLLM returned HTTP {status}")
            }
        }
    }
}

impl Error for ManagedVllmBridgeError {}

pub(crate) type ManagedVllmSseStream =
    Pin<Box<dyn Stream<Item = Result<Vec<u8>, ManagedVllmBridgeError>> + Send + 'static>>;

/// Raw HTTP response-body chunks. The stream polls Hyper directly, so dropping
/// it cancels the upstream body without an intermediate task or channel.
pub(crate) type ManagedVllmByteStream =
    Pin<Box<dyn Stream<Item = Result<Vec<u8>, ManagedVllmBridgeError>> + Send + 'static>>;

pub(crate) struct ManagedVllmBufferedResponse {
    pub(crate) status: StatusCode,
    pub(crate) headers: HeaderMap,
    pub(crate) body: Vec<u8>,
}

pub(crate) struct ManagedVllmRawResponse {
    pub(crate) status: StatusCode,
    pub(crate) headers: HeaderMap,
    pub(crate) body: ManagedVllmByteStream,
}

pub(crate) struct ManagedVllmSseResponse {
    pub(crate) status: StatusCode,
    pub(crate) headers: HeaderMap,
    pub(crate) events: ManagedVllmSseStream,
}

struct BodyCollectionTelemetry {
    telemetry: Option<Arc<ManagedVllmBridgeTelemetry>>,
    mode: &'static str,
    started: Instant,
}

/// Cloneable handle to two persistent connection pools for one managed child.
/// Cloning this type shares both pools; it does not construct new clients.
#[derive(Clone)]
pub(crate) struct ManagedVllmHttpBridge {
    origin: Arc<Uri>,
    blocking_client: ureq::Agent,
    async_client: Client<InstrumentedConnector, Body>,
    config: ManagedVllmBridgeConfig,
    telemetry: Option<Arc<ManagedVllmBridgeTelemetry>>,
}

impl ManagedVllmHttpBridge {
    fn record_request(&self, mode: &'static str, streaming: bool) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.request(mode, streaming);
        }
    }

    fn record_stage(&self, mode: &'static str, stage: &'static str, elapsed: Duration) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.stage(mode, stage, elapsed);
        }
    }

    fn record_error(&self, mode: &'static str, error: &ManagedVllmBridgeError) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.error(mode, error);
            if matches!(error, ManagedVllmBridgeError::Cancelled) {
                telemetry.cancellation(mode);
            }
        }
    }

    pub(crate) fn new(endpoint: impl AsRef<str>) -> Result<Self, ManagedVllmBridgeError> {
        Self::with_config_and_telemetry(endpoint, ManagedVllmBridgeConfig::default(), None)
    }

    pub(crate) fn new_with_telemetry(
        endpoint: impl AsRef<str>,
        telemetry: Arc<ManagedVllmBridgeTelemetry>,
    ) -> Result<Self, ManagedVllmBridgeError> {
        Self::with_config_and_telemetry(
            endpoint,
            ManagedVllmBridgeConfig::default(),
            Some(telemetry),
        )
    }

    fn with_config_and_telemetry(
        endpoint: impl AsRef<str>,
        config: ManagedVllmBridgeConfig,
        telemetry: Option<Arc<ManagedVllmBridgeTelemetry>>,
    ) -> Result<Self, ManagedVllmBridgeError> {
        let config = config.validate()?;
        let origin = endpoint
            .as_ref()
            .parse::<Uri>()
            .map_err(|error| ManagedVllmBridgeError::InvalidEndpoint(error.to_string()))?;
        if origin.scheme_str() != Some("http") || origin.authority().is_none() {
            return Err(ManagedVllmBridgeError::InvalidEndpoint(
                "managed vLLM endpoint must be an absolute http:// origin".to_string(),
            ));
        }
        if origin.path() != "/" && !origin.path().is_empty() {
            return Err(ManagedVllmBridgeError::InvalidEndpoint(
                "managed vLLM endpoint must not contain a path".to_string(),
            ));
        }

        let blocking_client: ureq::Agent = ureq::Agent::config_builder()
            .max_idle_age(config.pool_idle_timeout)
            .max_idle_connections_per_host(16)
            .build()
            .into();

        let mut connector = HttpConnector::new();
        connector.set_connect_timeout(Some(config.connect_timeout));
        connector.set_nodelay(true);
        let async_client = Client::builder()
            .pool_idle_timeout(config.pool_idle_timeout)
            .pool_max_idle_per_host(16)
            .build(InstrumentedConnector {
                inner: connector,
                telemetry: telemetry.clone(),
            });

        Ok(Self {
            origin: Arc::new(origin),
            blocking_client,
            async_client,
            config,
            telemetry,
        })
    }

    /// Send a one-shot JSON request using the shared blocking connection pool.
    /// HTTP error statuses are returned to the caller as ordinary responses so
    /// the caller can preserve the upstream OpenAI error body.
    pub(crate) fn post_json_sync(
        &self,
        path: &str,
        body: &[u8],
        timeouts: ManagedVllmRequestTimeouts,
    ) -> Result<ureq::http::Response<ureq::Body>, ManagedVllmBridgeError> {
        const MODE: &str = "legacy";
        self.record_request(MODE, false);
        let started = Instant::now();
        let timeouts = timeouts.validate()?;
        let uri = self.uri(path)?;
        self.record_stage(MODE, "upstream_dispatch", started.elapsed());
        let response = self
            .blocking_client
            .post(uri.to_string())
            .header("content-type", "application/json")
            .config()
            .timeout_global(Some(timeouts.total))
            .timeout_per_call(Some(timeouts.total))
            .timeout_connect(Some(
                self.config
                    .connect_timeout
                    .min(timeouts.headers)
                    .min(timeouts.total),
            ))
            .timeout_recv_response(Some(timeouts.headers.min(timeouts.total)))
            .timeout_recv_body(Some(timeouts.idle_body.min(timeouts.total)))
            .http_status_as_error(false)
            .build()
            .send(body)
            .map_err(|error| map_ureq_error(error, false));
        match &response {
            Ok(response) => {
                self.record_stage(MODE, "upstream_headers", started.elapsed());
                if !response.status().is_success() {
                    self.record_error(
                        MODE,
                        &ManagedVllmBridgeError::UpstreamStatus {
                            status: StatusCode::from_u16(response.status().as_u16())
                                .expect("ureq returned a valid HTTP status"),
                            body: Vec::new(),
                        },
                    );
                }
            }
            Err(error) => self.record_error(MODE, error),
        }
        response
    }

    /// Probe an endpoint with the shared asynchronous client. Successful
    /// responses are drained so their connection can return to the pool.
    pub(crate) async fn check_health(
        &self,
        path: &str,
        timeouts: ManagedVllmRequestTimeouts,
        cancellation: Option<CancellationToken>,
    ) -> Result<(), ManagedVllmBridgeError> {
        const MODE: &str = "health";
        self.record_request(MODE, false);
        let timeouts = timeouts.validate()?;
        let started = Instant::now();
        let request = Request::builder()
            .method(Method::GET)
            .uri(self.uri(path)?)
            .body(Body::empty())
            .map_err(|error| ManagedVllmBridgeError::BuildRequest(error.to_string()))?;
        self.record_stage(MODE, "upstream_dispatch", started.elapsed());
        let response = self
            .send_async(request, timeouts, started, cancellation.as_ref(), MODE)
            .await?;
        let status = response.status();
        let body = collect_body_bounded(
            response.into_body(),
            self.config.max_error_body_bytes,
            started + timeouts.total,
            timeouts.idle_body,
            cancellation,
            BodyCollectionTelemetry {
                telemetry: self.telemetry.clone(),
                mode: MODE,
                started,
            },
        )
        .await;
        let body = match body {
            Ok(body) => body,
            Err(error) => {
                self.record_error(MODE, &error);
                return Err(error);
            }
        };
        if !status.is_success() {
            let error = ManagedVllmBridgeError::UpstreamStatus { status, body };
            self.record_error(MODE, &error);
            return Err(error);
        }
        Ok(())
    }

    /// Start an OpenAI-compatible SSE request. Only response headers are read
    /// eagerly. Each subsequent poll of `events` reads only enough upstream
    /// bytes to produce the next data event.
    pub(crate) async fn post_json_sse(
        &self,
        path: &str,
        body: Vec<u8>,
        timeouts: ManagedVllmRequestTimeouts,
        cancellation: Option<CancellationToken>,
    ) -> Result<ManagedVllmSseResponse, ManagedVllmBridgeError> {
        const MODE: &str = "async_translated";
        self.record_request(MODE, true);
        let timeouts = timeouts.validate()?;
        let started = Instant::now();
        let request = Request::builder()
            .method(Method::POST)
            .uri(self.uri(path)?)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .map_err(|error| ManagedVllmBridgeError::BuildRequest(error.to_string()))?;
        self.record_stage(MODE, "upstream_dispatch", started.elapsed());
        let response = self
            .send_async(request, timeouts, started, cancellation.as_ref(), MODE)
            .await?;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.into_body();
        let total_deadline = started + timeouts.total;

        if !status.is_success() {
            let body = collect_body_bounded(
                body,
                self.config.max_error_body_bytes,
                total_deadline,
                timeouts.idle_body,
                cancellation,
                BodyCollectionTelemetry {
                    telemetry: self.telemetry.clone(),
                    mode: MODE,
                    started,
                },
            )
            .await;
            let body = match body {
                Ok(body) => body,
                Err(error) => {
                    self.record_error(MODE, &error);
                    return Err(error);
                }
            };
            let error = ManagedVllmBridgeError::UpstreamStatus { status, body };
            self.record_error(MODE, &error);
            return Err(error);
        }

        Ok(ManagedVllmSseResponse {
            status,
            headers,
            events: instrument_response_stream(
                decode_sse_body(
                    body,
                    self.config.max_sse_buffer_bytes,
                    total_deadline,
                    timeouts.idle_body,
                    cancellation,
                ),
                self.telemetry.clone(),
                MODE,
                started,
            ),
        })
    }

    /// Send an OpenAI-compatible request and collect its body without treating
    /// non-success statuses as transport failures. This lets the caller relay
    /// the exact upstream status and JSON error shape. `maximum_body_bytes`
    /// bounds both successful and error responses.
    pub(crate) async fn post_json_buffered(
        &self,
        path: &str,
        body: Vec<u8>,
        timeouts: ManagedVllmRequestTimeouts,
        cancellation: Option<CancellationToken>,
        maximum_body_bytes: usize,
    ) -> Result<ManagedVllmBufferedResponse, ManagedVllmBridgeError> {
        const MODE: &str = "wire";
        self.record_request(MODE, false);
        if maximum_body_bytes == 0 {
            return Err(ManagedVllmBridgeError::InvalidLimit(
                "managed vLLM response body limit must be non-zero".to_string(),
            ));
        }
        let timeouts = timeouts.validate()?;
        let started = Instant::now();
        let request = Request::builder()
            .method(Method::POST)
            .uri(self.uri(path)?)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .map_err(|error| ManagedVllmBridgeError::BuildRequest(error.to_string()))?;
        self.record_stage(MODE, "upstream_dispatch", started.elapsed());
        let response = self
            .send_async(request, timeouts, started, cancellation.as_ref(), MODE)
            .await?;
        let status = response.status();
        let headers = response.headers().clone();
        let body = collect_body_bounded(
            response.into_body(),
            maximum_body_bytes,
            started + timeouts.total,
            timeouts.idle_body,
            cancellation,
            BodyCollectionTelemetry {
                telemetry: self.telemetry.clone(),
                mode: MODE,
                started,
            },
        )
        .await;
        let body = match body {
            Ok(body) => body,
            Err(error) => {
                self.record_error(MODE, &error);
                return Err(error);
            }
        };
        if let Some(telemetry) = &self.telemetry {
            telemetry.relayed(MODE, body.len());
        }
        if !status.is_success() {
            self.record_error(
                MODE,
                &ManagedVllmBridgeError::UpstreamStatus {
                    status,
                    body: Vec::new(),
                },
            );
        }
        Ok(ManagedVllmBufferedResponse {
            status,
            headers,
            body,
        })
    }

    /// Start a raw OpenAI-compatible response relay. Headers are returned
    /// eagerly for every HTTP status and body chunks remain demand-driven.
    pub(crate) async fn post_json_raw(
        &self,
        path: &str,
        body: Vec<u8>,
        timeouts: ManagedVllmRequestTimeouts,
        cancellation: Option<CancellationToken>,
    ) -> Result<ManagedVllmRawResponse, ManagedVllmBridgeError> {
        const MODE: &str = "wire";
        self.record_request(MODE, true);
        let timeouts = timeouts.validate()?;
        let started = Instant::now();
        let request = Request::builder()
            .method(Method::POST)
            .uri(self.uri(path)?)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .map_err(|error| ManagedVllmBridgeError::BuildRequest(error.to_string()))?;
        self.record_stage(MODE, "upstream_dispatch", started.elapsed());
        let response = self
            .send_async(request, timeouts, started, cancellation.as_ref(), MODE)
            .await?;
        let status = response.status();
        let headers = response.headers().clone();
        if !status.is_success() {
            self.record_error(
                MODE,
                &ManagedVllmBridgeError::UpstreamStatus {
                    status,
                    body: Vec::new(),
                },
            );
        }
        Ok(ManagedVllmRawResponse {
            status,
            headers,
            body: instrument_response_stream(
                decode_raw_body(
                    response.into_body(),
                    started + timeouts.total,
                    timeouts.idle_body,
                    cancellation,
                ),
                self.telemetry.clone(),
                MODE,
                started,
            ),
        })
    }

    fn uri(&self, path: &str) -> Result<Uri, ManagedVllmBridgeError> {
        let path_and_query = path
            .parse::<PathAndQuery>()
            .map_err(|error| ManagedVllmBridgeError::InvalidPath(error.to_string()))?;
        if !path_and_query.path().starts_with('/') {
            return Err(ManagedVllmBridgeError::InvalidPath(
                "managed vLLM request path must start with '/'".to_string(),
            ));
        }
        Uri::builder()
            .scheme(
                self.origin
                    .scheme()
                    .expect("validated managed vLLM origin has a scheme")
                    .clone(),
            )
            .authority(
                self.origin
                    .authority()
                    .expect("validated managed vLLM origin has an authority")
                    .clone(),
            )
            .path_and_query(path_and_query)
            .build()
            .map_err(|error| ManagedVllmBridgeError::InvalidPath(error.to_string()))
    }

    async fn send_async(
        &self,
        request: Request<Body>,
        timeouts: ManagedVllmRequestTimeouts,
        started: Instant,
        cancellation: Option<&CancellationToken>,
        mode: &'static str,
    ) -> Result<hyper::Response<Body>, ManagedVllmBridgeError> {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            let error = ManagedVllmBridgeError::Cancelled;
            self.record_error(mode, &error);
            return Err(error);
        }
        let total_deadline = started + timeouts.total;
        let header_deadline = (started + timeouts.headers).min(total_deadline);
        let request = self.async_client.request(request);
        tokio::pin!(request);
        let result = loop {
            tokio::select! {
                result = &mut request => {
                    break result.map_err(|error| ManagedVllmBridgeError::Request(error.to_string()));
                }
                _ = tokio::time::sleep_until(header_deadline) => {
                    break if Instant::now() >= total_deadline {
                        Err(ManagedVllmBridgeError::TotalTimeout)
                    } else {
                        Err(ManagedVllmBridgeError::HeaderTimeout)
                    };
                }
                _ = tokio::time::sleep(CANCELLATION_POLL_INTERVAL), if cancellation.is_some() => {
                    if cancellation.is_some_and(CancellationToken::is_cancelled) {
                        break Err(ManagedVllmBridgeError::Cancelled);
                    }
                }
            }
        };
        match &result {
            Ok(_) => self.record_stage(mode, "upstream_headers", started.elapsed()),
            Err(error) => self.record_error(mode, error),
        }
        result
    }
}

/// Preserve the timeout category reported by the blocking compatibility
/// client. Without this conversion, a request-scoped ureq timeout is flattened
/// into a generic backend error by the managed engine.
pub(crate) fn map_ureq_body_error(error: ureq::Error) -> ManagedVllmBridgeError {
    map_ureq_error(error, true)
}

fn map_ureq_error(error: ureq::Error, reading_body: bool) -> ManagedVllmBridgeError {
    match error {
        ureq::Error::Timeout(timeout) => match timeout {
            ureq::Timeout::Global | ureq::Timeout::PerCall => ManagedVllmBridgeError::TotalTimeout,
            ureq::Timeout::RecvBody => ManagedVllmBridgeError::IdleBodyTimeout,
            // Resolution, connection, request upload, and response-header
            // deadlines all happen before a response body is available.
            _ => ManagedVllmBridgeError::HeaderTimeout,
        },
        ureq::Error::BodyExceedsLimit(limit) => ManagedVllmBridgeError::ResponseBodyExceeded {
            limit: usize::try_from(limit).unwrap_or(usize::MAX),
        },
        error if reading_body => ManagedVllmBridgeError::Body(error.to_string()),
        error => ManagedVllmBridgeError::Request(error.to_string()),
    }
}

struct SseDecodeState {
    body: Body,
    buffer: Vec<u8>,
    pending_chunk: Option<Bytes>,
    limit: usize,
    total_deadline: Instant,
    idle_body: Duration,
    cancellation: Option<CancellationToken>,
    eof: bool,
}

fn decode_sse_body(
    body: Body,
    limit: usize,
    total_deadline: Instant,
    idle_body: Duration,
    cancellation: Option<CancellationToken>,
) -> ManagedVllmSseStream {
    let state = SseDecodeState {
        body,
        buffer: Vec::new(),
        pending_chunk: None,
        limit,
        total_deadline,
        idle_body,
        cancellation,
        eof: false,
    };
    Box::pin(futures::stream::try_unfold(state, |mut state| async move {
        loop {
            // A single Hyper chunk may already contain many complete events.
            // Re-check request control before touching that buffered data so a
            // cancelled or expired stream cannot continue yielding solely
            // because it does not need another upstream body poll yet.
            check_stream_control(state.total_deadline, state.cancellation.as_ref())?;

            if let Some(raw_event) = take_complete_event(&mut state.buffer) {
                if let Some(data) = event_data(&raw_event) {
                    if data == b"[DONE]" {
                        // The translated compatibility stream completes as
                        // soon as vLLM emits its terminal event. Keep draining
                        // the already-bounded Hyper body in a lightweight
                        // Tokio task so an HTTP/1 connection is returned to
                        // the shared pool even when EOF is a later poll. The
                        // task owns no handoff buffer and remains subject to
                        // the body deadlines. Once the terminal event has
                        // been delivered, normal downstream stream teardown
                        // cancels the request token too; that cancellation
                        // must not abort the connection-reuse drain.
                        let body = std::mem::replace(&mut state.body, Body::empty());
                        spawn_terminal_body_drain(body, state.total_deadline, state.idle_body);
                        state.buffer.clear();
                        state.pending_chunk = None;
                        state.eof = true;
                    }
                    return Ok(Some((data, state)));
                }
                continue;
            }

            if state.eof {
                if state.buffer.is_empty() {
                    return Ok(None);
                }
                let final_event = std::mem::take(&mut state.buffer);
                return Ok(event_data(&final_event).map(|data| (data, state)));
            }

            // A transport chunk can contain any number of SSE events. Feed it
            // into the bounded pending-event buffer incrementally so a large
            // chunk of individually small events is not mistaken for one
            // oversized event. `Bytes::split_to` retains the unconsumed tail
            // without copying it into the decoder buffer.
            if let Some(mut chunk) = state.pending_chunk.take() {
                let remaining = state.limit.saturating_sub(state.buffer.len());
                if remaining == 0 {
                    return Err(ManagedVllmBridgeError::SseBufferExceeded { limit: state.limit });
                }
                let take = remaining.min(chunk.len());
                let prefix = chunk.split_to(take);
                state.buffer.extend_from_slice(&prefix);
                if !chunk.is_empty() {
                    state.pending_chunk = Some(chunk);
                }
                continue;
            }

            match next_body_chunk(
                &mut state.body,
                state.total_deadline,
                state.idle_body,
                state.cancellation.as_ref(),
            )
            .await?
            {
                Some(chunk) => state.pending_chunk = Some(chunk),
                None => state.eof = true,
            }
        }
    }))
}

fn decode_raw_body(
    body: Body,
    total_deadline: Instant,
    idle_body: Duration,
    cancellation: Option<CancellationToken>,
) -> ManagedVllmByteStream {
    Box::pin(futures::stream::try_unfold(
        (body, cancellation),
        move |(mut body, cancellation)| async move {
            next_body_chunk(&mut body, total_deadline, idle_body, cancellation.as_ref())
                .await
                .map(|chunk| chunk.map(|chunk| (chunk.to_vec(), (body, cancellation))))
        },
    ))
}

struct InstrumentedResponseStream {
    inner: ManagedVllmByteStream,
    telemetry: Option<Arc<ManagedVllmBridgeTelemetry>>,
    mode: &'static str,
    started: Instant,
    first_byte_seen: bool,
    terminal: bool,
}

impl Stream for InstrumentedResponseStream {
    type Item = Result<Vec<u8>, ManagedVllmBridgeError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(bytes))) => {
                let received = Instant::now();
                if !self.first_byte_seen {
                    self.first_byte_seen = true;
                    if let Some(telemetry) = &self.telemetry {
                        telemetry.stage(self.mode, "first_upstream_byte", self.started.elapsed());
                        telemetry.stage(
                            self.mode,
                            "first_upstream_to_downstream",
                            received.elapsed(),
                        );
                    }
                }
                if let Some(telemetry) = &self.telemetry {
                    telemetry.relayed(self.mode, bytes.len());
                }
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.terminal = true;
                if let Some(telemetry) = &self.telemetry {
                    telemetry.error(self.mode, &error);
                    if matches!(error, ManagedVllmBridgeError::Cancelled) {
                        telemetry.cancellation(self.mode);
                    }
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                self.terminal = true;
                Poll::Ready(None)
            }
        }
    }
}

impl Drop for InstrumentedResponseStream {
    fn drop(&mut self) {
        if let Some(telemetry) = &self.telemetry {
            telemetry.stream_finished(self.mode);
            if !self.terminal {
                telemetry.cancellation(self.mode);
            }
        }
    }
}

fn instrument_response_stream(
    inner: ManagedVllmByteStream,
    telemetry: Option<Arc<ManagedVllmBridgeTelemetry>>,
    mode: &'static str,
    started: Instant,
) -> ManagedVllmByteStream {
    if let Some(telemetry) = &telemetry {
        telemetry.stream_started(mode);
    }
    Box::pin(InstrumentedResponseStream {
        inner,
        telemetry,
        mode,
        started,
        first_byte_seen: false,
        terminal: false,
    })
}

fn spawn_terminal_body_drain(mut body: Body, total_deadline: Instant, idle_body: Duration) {
    tokio::spawn(async move {
        loop {
            match next_body_chunk(&mut body, total_deadline, idle_body, None).await {
                Ok(Some(_)) => {}
                Ok(None) => return,
                Err(error) => {
                    log::debug!(
                        "managed vLLM terminal response-body drain stopped before EOF: {error}"
                    );
                    return;
                }
            }
        }
    });
}

fn check_stream_control(
    total_deadline: Instant,
    cancellation: Option<&CancellationToken>,
) -> Result<(), ManagedVllmBridgeError> {
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        return Err(ManagedVllmBridgeError::Cancelled);
    }
    if Instant::now() >= total_deadline {
        return Err(ManagedVllmBridgeError::TotalTimeout);
    }
    Ok(())
}

/// Remove one complete SSE event. Both LF and CRLF line endings are accepted,
/// including a mixed `\n\r\n` event boundary.
fn take_complete_event(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let (event_end, consumed) = (0..buffer.len()).find_map(|index| {
        if buffer[index] != b'\n' {
            return None;
        }
        if buffer.get(index + 1) == Some(&b'\n') {
            return Some((index, index + 2));
        }
        if buffer.get(index + 1) == Some(&b'\r') && buffer.get(index + 2) == Some(&b'\n') {
            return Some((index, index + 3));
        }
        None
    })?;
    let mut event = buffer.drain(..consumed).collect::<Vec<_>>();
    event.truncate(event_end);
    Some(event)
}

/// Extract and join the data fields from an SSE event. Per the SSE format, one
/// optional ASCII space after `data:` is removed and multiple fields are joined
/// with a newline.
fn event_data(event: &[u8]) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    let mut saw_data = false;
    for raw_line in event.split(|byte| *byte == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        if line.first() == Some(&b':') {
            continue;
        }
        let (field, value) = match line.iter().position(|byte| *byte == b':') {
            Some(colon) => (&line[..colon], &line[colon + 1..]),
            None => (line, &[][..]),
        };
        if field != b"data" {
            continue;
        }
        if saw_data {
            data.push(b'\n');
        }
        let value = value.strip_prefix(b" ").unwrap_or(value);
        data.extend_from_slice(value);
        saw_data = true;
    }
    saw_data.then_some(data)
}

async fn next_body_chunk(
    body: &mut Body,
    total_deadline: Instant,
    idle_body: Duration,
    cancellation: Option<&CancellationToken>,
) -> Result<Option<Bytes>, ManagedVllmBridgeError> {
    check_stream_control(total_deadline, cancellation)?;
    let idle_deadline = (Instant::now() + idle_body).min(total_deadline);
    let next = body.data();
    tokio::pin!(next);
    loop {
        tokio::select! {
            result = &mut next => {
                return result
                    .transpose()
                    .map_err(|error| ManagedVllmBridgeError::Body(error.to_string()));
            }
            _ = tokio::time::sleep_until(idle_deadline) => {
                return if Instant::now() >= total_deadline {
                    Err(ManagedVllmBridgeError::TotalTimeout)
                } else {
                    Err(ManagedVllmBridgeError::IdleBodyTimeout)
                };
            }
            _ = tokio::time::sleep(CANCELLATION_POLL_INTERVAL), if cancellation.is_some() => {
                if cancellation.is_some_and(CancellationToken::is_cancelled) {
                    return Err(ManagedVllmBridgeError::Cancelled);
                }
            }
        }
    }
}

async fn collect_body_bounded(
    mut body: Body,
    limit: usize,
    total_deadline: Instant,
    idle_body: Duration,
    cancellation: Option<CancellationToken>,
    metrics: BodyCollectionTelemetry,
) -> Result<Vec<u8>, ManagedVllmBridgeError> {
    let mut output = Vec::new();
    let mut first_byte_seen = false;
    while let Some(chunk) =
        next_body_chunk(&mut body, total_deadline, idle_body, cancellation.as_ref()).await?
    {
        if !first_byte_seen {
            first_byte_seen = true;
            if let Some(telemetry) = &metrics.telemetry {
                telemetry.stage(
                    metrics.mode,
                    "first_upstream_byte",
                    metrics.started.elapsed(),
                );
            }
        }
        let length = output
            .len()
            .checked_add(chunk.len())
            .ok_or(ManagedVllmBridgeError::ResponseBodyExceeded { limit })?;
        if length > limit {
            return Err(ManagedVllmBridgeError::ResponseBodyExceeded { limit });
        }
        output.extend_from_slice(&chunk);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use hyper::service::{make_service_fn, service_fn};
    use hyper::{Response, Server};
    use std::convert::Infallible;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::oneshot;

    struct TestServer {
        endpoint: String,
        shutdown: Option<oneshot::Sender<()>>,
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
        }
    }

    async fn start_server<F, Fut>(connections: Arc<AtomicUsize>, handler: F) -> TestServer
    where
        F: Fn(Request<Body>) -> Fut + Clone + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Response<Body>, Infallible>> + Send + 'static,
    {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let make_service = make_service_fn(move |_| {
            connections.fetch_add(1, Ordering::SeqCst);
            let handler = handler.clone();
            async move { Ok::<_, Infallible>(service_fn(handler)) }
        });
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = Server::from_tcp(listener)
            .unwrap()
            .serve(make_service)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            });
        tokio::spawn(async move {
            let _ = server.await;
        });
        TestServer {
            endpoint: format!("http://{address}"),
            shutdown: Some(shutdown_tx),
        }
    }

    fn test_timeouts() -> ManagedVllmRequestTimeouts {
        ManagedVllmRequestTimeouts::new(
            Duration::from_secs(2),
            Duration::from_secs(2),
            Duration::from_secs(5),
        )
    }

    #[test]
    fn timeout_validation_rejects_an_unrepresentable_deadline() {
        let timeouts = ManagedVllmRequestTimeouts::new(
            Duration::from_secs(1),
            Duration::from_secs(1),
            Duration::MAX,
        );
        assert!(matches!(
            timeouts.validate(),
            Err(ManagedVllmBridgeError::InvalidTimeout(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn persistent_hyper_client_reuses_connections() {
        let connections = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(AtomicUsize::new(0));
        let requests_for_server = requests.clone();
        let server = start_server(connections.clone(), move |_request| {
            requests_for_server.fetch_add(1, Ordering::SeqCst);
            async {
                Ok(Response::builder()
                    .header(hyper::header::CONTENT_LENGTH, "2")
                    .body(Body::from("{}"))
                    .unwrap())
            }
        })
        .await;
        let bridge = ManagedVllmHttpBridge::new(&server.endpoint).unwrap();

        let blocking = bridge.clone();
        tokio::task::spawn_blocking(move || {
            for _ in 0..2 {
                let mut response = blocking
                    .post_json_sync("/v1/chat/completions", b"{}", test_timeouts())
                    .unwrap();
                response.body_mut().read_to_string().unwrap();
                drop(response);
            }
        })
        .await
        .unwrap();
        let blocking_connections = connections.load(Ordering::SeqCst);
        // ureq can retire an otherwise reusable HTTP/1 socket while this
        // parallel test harness is scheduling unrelated servers. The bridge
        // structurally shares one Agent; avoid making opportunistic pool
        // retirement an exact socket-count contract here. Hyper's asynchronous
        // pool reuse below is deterministic and remains asserted end to end.

        bridge
            .check_health("/health", test_timeouts(), None)
            .await
            .unwrap();
        bridge
            .check_health("/health", test_timeouts(), None)
            .await
            .unwrap();

        assert_eq!(requests.load(Ordering::SeqCst), 4);
        assert_eq!(
            connections.load(Ordering::SeqCst) - blocking_connections,
            1,
            "Hyper should reuse its connection"
        );
    }

    #[tokio::test]
    async fn telemetry_tracks_connection_stream_bytes_and_terminal_release() {
        let connections = Arc::new(AtomicUsize::new(0));
        let server = start_server(connections.clone(), |_request| async move {
            Ok(Response::new(Body::wrap_stream(futures::stream::iter(
                vec![
                    Ok::<_, Infallible>(Bytes::from_static(b"data: one\n\n")),
                    Ok::<_, Infallible>(Bytes::from_static(b"data: [DONE]\n\n")),
                ],
            ))))
        })
        .await;
        let registry = Arc::new(prometheus::Registry::new());
        let metrics = kapsl_monitor::metrics::KapslMetrics::new(&registry);
        let bridge = ManagedVllmHttpBridge::new_with_telemetry(
            &server.endpoint,
            ManagedVllmBridgeTelemetry::new(metrics.clone(), "model-a".to_string(), 3),
        )
        .unwrap();
        let mut response = bridge
            .post_json_raw(
                "/v1/chat/completions",
                b"{}".to_vec(),
                ManagedVllmRequestTimeouts::default(),
                None,
            )
            .await
            .unwrap();
        while response.body.next().await.is_some() {}
        drop(response);

        assert_eq!(connections.load(Ordering::Relaxed), 1);
        assert_eq!(
            metrics
                .managed_vllm
                .bridge_connection_attempts_total
                .with_label_values(&["model-a", "3", "async"])
                .get(),
            1
        );
        assert_eq!(
            metrics
                .managed_vllm
                .bridge_active_streams
                .with_label_values(&["model-a", "3", "wire"])
                .get(),
            0
        );
        assert_eq!(
            metrics
                .managed_vllm
                .bridge_relayed_chunks_total
                .with_label_values(&["model-a", "3", "wire"])
                .get(),
            2
        );
        assert_eq!(
            metrics
                .managed_vllm
                .bridge_relayed_bytes_total
                .with_label_values(&["model-a", "3", "wire"])
                .get(),
            25
        );
    }

    #[tokio::test]
    async fn buffered_post_preserves_non_success_status_headers_and_body() {
        let server = start_server(Arc::new(AtomicUsize::new(0)), |_request| async {
            Ok(Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header(CONTENT_TYPE, "application/json")
                .header("retry-after", "3")
                .body(Body::from(r#"{"error":{"message":"busy"}}"#))
                .unwrap())
        })
        .await;
        let bridge = ManagedVllmHttpBridge::new(&server.endpoint).unwrap();

        let response = bridge
            .post_json_buffered(
                "/v1/chat/completions",
                b"{}".to_vec(),
                test_timeouts(),
                None,
                1024,
            )
            .await
            .unwrap();

        assert_eq!(response.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers.get("retry-after").unwrap(), "3");
        assert_eq!(response.body, br#"{"error":{"message":"busy"}}"#);
    }

    #[tokio::test]
    async fn raw_post_preserves_chunks_and_cancellation_interrupts_idle_body() {
        let body_polls = Arc::new(AtomicUsize::new(0));
        let observed_polls = body_polls.clone();
        let server = start_server(Arc::new(AtomicUsize::new(0)), move |_request| {
            let observed_polls = observed_polls.clone();
            async move {
                let chunks = futures::stream::unfold(0usize, move |index| {
                    let observed_polls = observed_polls.clone();
                    async move {
                        if index == 0 {
                            observed_polls.fetch_add(1, Ordering::SeqCst);
                            return Some((
                                Ok::<_, Infallible>(Bytes::from_static(b"data: first\n\n")),
                                1,
                            ));
                        }
                        futures::future::pending().await
                    }
                });
                Ok(Response::builder()
                    .header(CONTENT_TYPE, "text/event-stream")
                    .body(Body::wrap_stream(chunks))
                    .unwrap())
            }
        })
        .await;
        let bridge = ManagedVllmHttpBridge::new(&server.endpoint).unwrap();
        let cancellation = CancellationToken::new();
        let mut response = bridge
            .post_json_raw(
                "/v1/chat/completions",
                b"{}".to_vec(),
                test_timeouts(),
                Some(cancellation.clone()),
            )
            .await
            .unwrap();

        assert_eq!(
            response.body.next().await.unwrap().unwrap(),
            b"data: first\n\n"
        );
        assert_eq!(body_polls.load(Ordering::SeqCst), 1);

        let task = tokio::spawn(async move { response.body.next().await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancellation.cancel();
        let result = tokio::time::timeout(Duration::from_millis(250), task)
            .await
            .expect("cancellation should wake raw body polling")
            .unwrap()
            .unwrap();
        assert!(matches!(result, Err(ManagedVllmBridgeError::Cancelled)));
    }

    #[tokio::test]
    async fn sse_decoder_handles_single_byte_network_chunks() {
        let connections = Arc::new(AtomicUsize::new(0));
        let methods = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let methods_for_server = methods.clone();
        let payload = b": heartbeat\r\n\r\ndata: {\"text\":\"hello\"}\r\n\r\ndata: [DONE]\n\n";
        let server = start_server(connections, move |request| {
            methods_for_server.lock().push(request.method().clone());
            let chunks = payload
                .iter()
                .copied()
                .map(|byte| Ok::<_, Infallible>(Bytes::copy_from_slice(&[byte])));
            async move {
                Ok(Response::builder()
                    .header(CONTENT_TYPE, "text/event-stream")
                    .body(Body::wrap_stream(futures::stream::iter(chunks)))
                    .unwrap())
            }
        })
        .await;
        let bridge = ManagedVllmHttpBridge::new(&server.endpoint).unwrap();
        let response = bridge
            .post_json_sse(
                "/v1/chat/completions",
                b"{}".to_vec(),
                test_timeouts(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(
            response.headers.get(CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let events = response.events.collect::<Vec<_>>().await;
        assert_eq!(
            events.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
            vec![b"{\"text\":\"hello\"}".to_vec(), b"[DONE]".to_vec()]
        );
        assert_eq!(&*methods.lock(), &[Method::POST]);
    }

    #[tokio::test]
    async fn terminal_event_drains_the_remaining_body_for_pool_reuse() {
        let polled = Arc::new(AtomicUsize::new(0));
        let observed = polled.clone();
        let chunks = futures::stream::unfold(0usize, move |index| {
            let observed = observed.clone();
            async move {
                let chunk = match index {
                    0 => Bytes::from_static(b"data: [DONE]\n\n"),
                    1 => Bytes::from_static(b": trailer one\n\n"),
                    2 => Bytes::from_static(b": trailer two\n\n"),
                    _ => return None,
                };
                if index > 0 {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                observed.fetch_add(1, Ordering::SeqCst);
                Some((Ok::<_, Infallible>(chunk), index + 1))
            }
        });
        let cancellation = CancellationToken::new();
        let mut decoded = decode_sse_body(
            Body::wrap_stream(chunks),
            1024,
            Instant::now() + Duration::from_secs(2),
            Duration::from_secs(1),
            Some(cancellation.clone()),
        );

        assert_eq!(decoded.next().await.unwrap().unwrap(), b"[DONE]");
        cancellation.cancel();
        drop(decoded);
        tokio::time::timeout(Duration::from_secs(1), async {
            while polled.load(Ordering::SeqCst) != 3 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal drain should consume the response body through EOF");
    }

    #[tokio::test]
    async fn decoder_is_pull_based_and_applies_backpressure() {
        let polls = Arc::new(AtomicUsize::new(0));
        let polls_for_stream = polls.clone();
        let chunks = futures::stream::unfold(0usize, move |index| {
            let polls = polls_for_stream.clone();
            async move {
                if index >= 3 {
                    return None;
                }
                polls.fetch_add(1, Ordering::SeqCst);
                Some((
                    Ok::<_, Infallible>(Bytes::from(format!("data: {index}\n\n"))),
                    index + 1,
                ))
            }
        });
        let body = Body::wrap_stream(chunks);
        let mut events = decode_sse_body(
            body,
            1024,
            Instant::now() + Duration::from_secs(2),
            Duration::from_secs(1),
            None,
        );

        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(events.next().await.unwrap().unwrap(), b"0");
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        tokio::task::yield_now().await;
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn oversized_unterminated_event_fails_closed() {
        let body = Body::from(format!("data: {}", "x".repeat(64)));
        let mut events = decode_sse_body(
            body,
            32,
            Instant::now() + Duration::from_secs(2),
            Duration::from_secs(1),
            None,
        );
        assert!(matches!(
            events.next().await.unwrap(),
            Err(ManagedVllmBridgeError::SseBufferExceeded { limit: 32 })
        ));
    }

    #[tokio::test]
    async fn large_network_chunk_with_small_events_respects_event_limit() {
        let body = Body::from("data: 0\n\ndata: 1\n\ndata: 2\n\n");
        let events = decode_sse_body(
            body,
            12,
            Instant::now() + Duration::from_secs(2),
            Duration::from_secs(1),
            None,
        )
        .collect::<Vec<_>>()
        .await;

        assert_eq!(
            events.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
            vec![b"0".to_vec(), b"1".to_vec(), b"2".to_vec()]
        );
    }

    #[tokio::test]
    async fn cancellation_interrupts_an_idle_body() {
        let body = Body::wrap_stream(futures::stream::pending::<Result<Bytes, Infallible>>());
        let cancellation = CancellationToken::new();
        let mut events = decode_sse_body(
            body,
            1024,
            Instant::now() + Duration::from_secs(5),
            Duration::from_secs(5),
            Some(cancellation.clone()),
        );
        let task = tokio::spawn(async move { events.next().await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancellation.cancel();
        let result = tokio::time::timeout(Duration::from_millis(250), task)
            .await
            .expect("cancellation should wake body polling")
            .unwrap()
            .unwrap();
        assert!(matches!(result, Err(ManagedVllmBridgeError::Cancelled)));
    }

    #[tokio::test]
    async fn cancellation_preempts_events_buffered_in_one_network_chunk() {
        let body = Body::from("data: first\n\ndata: second\n\ndata: third\n\n");
        let cancellation = CancellationToken::new();
        let mut events = decode_sse_body(
            body,
            1024,
            Instant::now() + Duration::from_secs(2),
            Duration::from_secs(1),
            Some(cancellation.clone()),
        );

        assert_eq!(events.next().await.unwrap().unwrap(), b"first");
        cancellation.cancel();
        assert!(matches!(
            events.next().await.unwrap(),
            Err(ManagedVllmBridgeError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn total_deadline_preempts_events_buffered_in_one_network_chunk() {
        let body = Body::from("data: first\n\ndata: second\n\ndata: third\n\n");
        let total_deadline = Instant::now() + Duration::from_millis(50);
        let mut events = decode_sse_body(body, 1024, total_deadline, Duration::from_secs(1), None);

        assert_eq!(events.next().await.unwrap().unwrap(), b"first");
        tokio::time::sleep_until(total_deadline + Duration::from_millis(1)).await;
        assert!(matches!(
            events.next().await.unwrap(),
            Err(ManagedVllmBridgeError::TotalTimeout)
        ));
    }

    #[tokio::test]
    async fn non_success_status_retains_the_upstream_body() {
        let connections = Arc::new(AtomicUsize::new(0));
        let server = start_server(connections, |_request| async {
            Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"error":{"message":"bad request"}}"#))
                .unwrap())
        })
        .await;
        let bridge = ManagedVllmHttpBridge::new(&server.endpoint).unwrap();
        let error = match bridge
            .post_json_sse(
                "/v1/chat/completions",
                b"{}".to_vec(),
                test_timeouts(),
                None,
            )
            .await
        {
            Ok(_) => panic!("upstream error should not become a successful stream"),
            Err(error) => error,
        };
        assert_eq!(error.upstream_status(), Some(StatusCode::BAD_REQUEST));
        assert_eq!(
            error.upstream_body().unwrap(),
            br#"{"error":{"message":"bad request"}}"#
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_header_timeout_retains_its_timeout_category() {
        let connections = Arc::new(AtomicUsize::new(0));
        let server = start_server(connections, |_request| async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(Response::new(Body::from("{}")))
        })
        .await;
        let bridge = ManagedVllmHttpBridge::new(&server.endpoint).unwrap();

        let error = tokio::task::spawn_blocking(move || {
            bridge
                .post_json_sync(
                    "/v1/chat/completions",
                    b"{}",
                    ManagedVllmRequestTimeouts::new(
                        Duration::from_millis(20),
                        Duration::from_secs(1),
                        Duration::from_secs(1),
                    ),
                )
                .unwrap_err()
        })
        .await
        .unwrap();

        assert!(matches!(error, ManagedVllmBridgeError::HeaderTimeout));
    }

    #[test]
    fn endpoint_and_path_are_restricted_to_the_managed_origin() {
        assert!(ManagedVllmHttpBridge::new("https://example.com").is_err());
        assert!(ManagedVllmHttpBridge::new("http://127.0.0.1:8000/base").is_err());
        let bridge = ManagedVllmHttpBridge::new("http://127.0.0.1:8000").unwrap();
        assert!(bridge.uri("http://attacker.invalid/").is_err());
        assert_eq!(
            bridge.uri("/health?probe=1").unwrap().to_string(),
            "http://127.0.0.1:8000/health?probe=1"
        );
    }
}
