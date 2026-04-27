use base64::Engine as _;
use image::{imageops::FilterType, DynamicImage};
use kapsl_engine_api::{
    BinaryTensorPacket, InferenceRequest, NamedTensor, RequestMetadata as EngineRequestMetadata,
    TensorDtype,
};
use serde::Deserialize;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(unix)]
use std::{
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
};

const OPTIONAL_ADAPTERS_ENV: &str = "KAPSL_INFER_ADAPTERS";
const LEGACY_OPTIONAL_ADAPTERS_ENV: &str = "KAPSL_INFER_ADAPTERS";
const DISABLE_INLINE_MEDIA_PREPROCESS_ENV: &str = "KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS";
const LEGACY_DISABLE_INLINE_MEDIA_PREPROCESS_ENV: &str = "KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS";

fn env_var_alias(primary: &str, legacy: &str) -> Option<String> {
    std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(legacy).ok())
}

#[derive(Debug, Clone)]
pub(crate) enum InferRequestError {
    BadRequest(String),
    Internal(String),
}

impl InferRequestError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub(crate) fn is_internal(&self) -> bool {
        matches!(self, Self::Internal(_))
    }
}

impl fmt::Display for InferRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferRequestError::BadRequest(message) => write!(f, "{}", message),
            InferRequestError::Internal(message) => write!(f, "{}", message),
        }
    }
}

type InferResult<T> = Result<T, InferRequestError>;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Deserialize)]
struct MediaPayload {
    #[serde(default)]
    kind: Option<MediaKind>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(alias = "base64")]
    data_base64: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MediaTensorLayout {
    Nchw,
    Nhwc,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MediaChannelMode {
    Rgb,
    Bgr,
    Grayscale,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PixelNormalization {
    Auto,
    None,
    ZeroToOne,
    MinusOneToOne,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct MediaTensorOptions {
    target_width: Option<u32>,
    target_height: Option<u32>,
    layout: MediaTensorLayout,
    channels: MediaChannelMode,
    dtype: TensorDtype,
    normalize: PixelNormalization,
    frame_count: Option<usize>,
    frame_stride: usize,
    start_time_ms: Option<u64>,
    end_time_ms: Option<u64>,
}

impl Default for MediaTensorOptions {
    fn default() -> Self {
        Self {
            target_width: None,
            target_height: None,
            layout: MediaTensorLayout::Nchw,
            channels: MediaChannelMode::Rgb,
            dtype: TensorDtype::Float32,
            normalize: PixelNormalization::Auto,
            frame_count: Some(1),
            frame_stride: 1,
            start_time_ms: None,
            end_time_ms: None,
        }
    }
}

impl MediaTensorOptions {
    fn resolved_normalization(&self) -> PixelNormalization {
        match self.normalize {
            PixelNormalization::Auto => match self.dtype {
                TensorDtype::Uint8 => PixelNormalization::None,
                _ => PixelNormalization::ZeroToOne,
            },
            other => other,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct NamedMediaPayload {
    name: String,
    media: MediaPayload,
    #[serde(default)]
    tensor_options: Option<MediaTensorOptions>,
}

#[derive(Debug, Clone, Deserialize)]
struct HttpMediaInferenceRequest {
    media: MediaPayload,
    #[serde(default)]
    additional_media_inputs: Vec<NamedMediaPayload>,
    #[serde(default)]
    additional_inputs: Vec<NamedTensor>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    metadata: Option<EngineRequestMetadata>,
    #[serde(default, alias = "preprocess", alias = "options")]
    tensor_options: MediaTensorOptions,
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(crate) trait ModelRequestAdapter: Send + Sync {
    fn name(&self) -> &'static str;

    fn supported_frameworks(&self) -> &'static [&'static str] {
        &["*"]
    }

    fn supports_payload(&self, payload: &serde_json::Value) -> bool;

    fn adapt(&self, payload: serde_json::Value) -> InferResult<InferenceRequest>;

    fn supports_framework(&self, framework: &str) -> bool {
        self.supported_frameworks()
            .iter()
            .any(|candidate| *candidate == "*" || candidate.eq_ignore_ascii_case(framework))
    }
}

struct TensorRequestAdapter;

impl ModelRequestAdapter for TensorRequestAdapter {
    fn name(&self) -> &'static str {
        "tensor_json"
    }

    fn supports_payload(&self, payload: &serde_json::Value) -> bool {
        payload.get("input").is_some()
    }

    fn adapt(&self, payload: serde_json::Value) -> InferResult<InferenceRequest> {
        // Extract top-level generation params before consuming the payload.
        // Callers often send {"input": ..., "max_tokens": 256, "min_tokens": 256}
        // which serde would silently drop since they're not InferenceRequest fields.
        let top_max_tokens = payload
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let top_min_tokens = payload
            .get("min_tokens")
            .or_else(|| payload.get("min_new_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let top_temperature = payload
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);

        let mut request =
            serde_json::from_value::<InferenceRequest>(payload).map_err(|e| {
                InferRequestError::bad_request(format!("Invalid tensor infer payload: {}", e))
            })?;

        if top_max_tokens.is_some() || top_min_tokens.is_some() || top_temperature.is_some() {
            let meta = request
                .metadata
                .get_or_insert_with(EngineRequestMetadata::default);
            if meta.max_new_tokens.is_none() {
                meta.max_new_tokens = top_max_tokens;
            }
            if meta.min_new_tokens.is_none() {
                meta.min_new_tokens = top_min_tokens;
            }
            if meta.temperature.is_none() {
                meta.temperature = top_temperature;
            }
        }

        Ok(request)
    }
}

struct MediaRequestAdapter;

impl ModelRequestAdapter for MediaRequestAdapter {
    fn name(&self) -> &'static str {
        "media_base64"
    }

    fn supports_payload(&self, payload: &serde_json::Value) -> bool {
        payload.get("media").is_some()
    }

    fn adapt(&self, payload: serde_json::Value) -> InferResult<InferenceRequest> {
        let media_request =
            serde_json::from_value::<HttpMediaInferenceRequest>(payload).map_err(|e| {
                InferRequestError::bad_request(format!("Invalid media infer payload: {}", e))
            })?;
        media_infer_request_to_inference_request(media_request)
    }
}

#[cfg(feature = "infer-adapter-echo")]
#[derive(Debug, Clone, Deserialize)]
struct EchoTensorInferenceRequest {
    echo_input: BinaryTensorPacket,
    #[serde(default)]
    additional_inputs: Vec<NamedTensor>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    metadata: Option<EngineRequestMetadata>,
}

#[cfg(feature = "infer-adapter-echo")]
struct EchoTensorRequestAdapter;

#[cfg(feature = "infer-adapter-echo")]
impl ModelRequestAdapter for EchoTensorRequestAdapter {
    fn name(&self) -> &'static str {
        "echo_tensor"
    }

    fn supports_payload(&self, payload: &serde_json::Value) -> bool {
        payload.get("echo_input").is_some()
    }

    fn adapt(&self, payload: serde_json::Value) -> InferResult<InferenceRequest> {
        let request =
            serde_json::from_value::<EchoTensorInferenceRequest>(payload).map_err(|e| {
                InferRequestError::bad_request(format!("Invalid echo tensor payload: {}", e))
            })?;
        Ok(InferenceRequest {
            input: request.echo_input,
            additional_inputs: request.additional_inputs,
            session_id: request.session_id,
            metadata: request.metadata,
            cancellation: None,
        })
    }
}

#[allow(dead_code)]
enum OptionalAdapterRegistrationOutcome {
    Registered,
    AlreadyRegistered,
    FeatureDisabled,
    Unknown,
}

pub(crate) struct RequestAdapterRegistry {
    adapters: Vec<Box<dyn ModelRequestAdapter>>,
}

impl RequestAdapterRegistry {
    pub(crate) fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    pub(crate) fn register(&mut self, adapter: Box<dyn ModelRequestAdapter>) {
        self.adapters.push(adapter);
    }

    fn register_if_missing(&mut self, adapter: Box<dyn ModelRequestAdapter>) -> bool {
        if self.has_adapter_name(adapter.name()) {
            return false;
        }
        self.register(adapter);
        true
    }

    fn has_adapter_name(&self, name: &str) -> bool {
        self.adapters.iter().any(|adapter| adapter.name() == name)
    }

    fn new_default() -> Self {
        let mut registry = Self::new();
        registry.register_if_missing(Box::new(TensorRequestAdapter));
        registry.register_if_missing(Box::new(MediaRequestAdapter));
        register_optional_adapters_from_env(&mut registry);
        registry
    }

    fn adapt(&self, framework: &str, payload: serde_json::Value) -> InferResult<InferenceRequest> {
        let framework = framework.trim().to_ascii_lowercase();
        let framework = if framework.is_empty() {
            "unknown".to_string()
        } else {
            framework
        };

        let mut payload_matched_any = false;
        let mut framework_filtered = Vec::new();
        let mut selected_adapter_index = None;
        for (index, adapter) in self.adapters.iter().enumerate() {
            if !adapter.supports_payload(&payload) {
                continue;
            }
            payload_matched_any = true;
            if !adapter.supports_framework(&framework) {
                framework_filtered.push(adapter.name());
                continue;
            }
            selected_adapter_index = Some(index);
            break;
        }

        if let Some(index) = selected_adapter_index {
            return self.adapters[index].adapt(payload);
        }

        if payload_matched_any {
            if framework_filtered.is_empty() {
                return Err(InferRequestError::bad_request(
                    "Infer payload matched adapter shape but could not be adapted",
                ));
            }
            return Err(InferRequestError::bad_request(format!(
                "No infer adapter supports framework `{}` for this payload. Matching adapters: {}",
                framework,
                framework_filtered.join(", ")
            )));
        }

        let adapters = self
            .adapters
            .iter()
            .map(|adapter| format!("`{}`", adapter.name()))
            .collect::<Vec<_>>()
            .join(", ");
        Err(InferRequestError::bad_request(format!(
            "Invalid infer payload for framework `{}`. Known adapters: {}",
            framework, adapters
        )))
    }
}

fn register_optional_adapters_from_env(registry: &mut RequestAdapterRegistry) {
    let Some(spec) = env_var_alias(OPTIONAL_ADAPTERS_ENV, LEGACY_OPTIONAL_ADAPTERS_ENV) else {
        return;
    };

    apply_optional_adapter_spec(registry, &spec);
}

fn apply_optional_adapter_spec(registry: &mut RequestAdapterRegistry, spec: &str) {
    for token in spec
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        match register_optional_adapter_token(registry, token) {
            OptionalAdapterRegistrationOutcome::Registered => {
                log::info!(
                    "Enabled optional infer adapter `{}` via {}",
                    token,
                    OPTIONAL_ADAPTERS_ENV
                );
            }
            OptionalAdapterRegistrationOutcome::AlreadyRegistered => {
                log::debug!(
                    "Optional infer adapter `{}` already registered (from {})",
                    token,
                    OPTIONAL_ADAPTERS_ENV
                );
            }
            OptionalAdapterRegistrationOutcome::FeatureDisabled => {
                log::warn!(
                    "Adapter `{}` requested via {}, but this binary was built without required feature support",
                    token,
                    OPTIONAL_ADAPTERS_ENV
                );
            }
            OptionalAdapterRegistrationOutcome::Unknown => {
                log::warn!(
                    "Unknown adapter token `{}` in {}. Known optional adapters: `echo_tensor`",
                    token,
                    OPTIONAL_ADAPTERS_ENV
                );
            }
        }
    }
}

fn register_optional_adapter_token(
    registry: &mut RequestAdapterRegistry,
    token: &str,
) -> OptionalAdapterRegistrationOutcome {
    match token.to_ascii_lowercase().as_str() {
        "echo_tensor" => register_echo_tensor_adapter(registry),
        _ => OptionalAdapterRegistrationOutcome::Unknown,
    }
}

#[cfg(feature = "infer-adapter-echo")]
fn register_echo_tensor_adapter(
    registry: &mut RequestAdapterRegistry,
) -> OptionalAdapterRegistrationOutcome {
    if registry.register_if_missing(Box::new(EchoTensorRequestAdapter)) {
        OptionalAdapterRegistrationOutcome::Registered
    } else {
        OptionalAdapterRegistrationOutcome::AlreadyRegistered
    }
}

#[cfg(not(feature = "infer-adapter-echo"))]
fn register_echo_tensor_adapter(
    _registry: &mut RequestAdapterRegistry,
) -> OptionalAdapterRegistrationOutcome {
    OptionalAdapterRegistrationOutcome::FeatureDisabled
}

pub(crate) fn default_request_adapter_registry() -> RequestAdapterRegistry {
    RequestAdapterRegistry::new_default()
}

pub(crate) fn parse_inference_request_with_registry(
    body: serde_json::Value,
    model_framework: &str,
    registry: &RequestAdapterRegistry,
) -> InferResult<InferenceRequest> {
    registry.adapt(model_framework, body)
}

fn media_infer_request_to_inference_request(
    request: HttpMediaInferenceRequest,
) -> InferResult<InferenceRequest> {
    if inline_media_preprocess_disabled() {
        return Err(InferRequestError::bad_request(
            "Inline media preprocessing is disabled for hot-path inference. Send preprocessed tensors (`input` and `additional_inputs`) or unset KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS.",
        ));
    }

    let primary = media_payload_to_tensor_packet(&request.media, &request.tensor_options)?;
    let mut inference = InferenceRequest {
        input: primary,
        additional_inputs: request.additional_inputs,
        session_id: request.session_id,
        metadata: request.metadata,
        cancellation: None,
    };

    for additional_media in request.additional_media_inputs {
        let options = additional_media
            .tensor_options
            .unwrap_or_else(|| request.tensor_options.clone());
        let tensor = media_payload_to_tensor_packet(&additional_media.media, &options)?;
        inference.additional_inputs.push(NamedTensor {
            name: additional_media.name,
            tensor,
        });
    }

    Ok(inference)
}

fn inline_media_preprocess_disabled() -> bool {
    env_var_alias(
        DISABLE_INLINE_MEDIA_PREPROCESS_ENV,
        LEGACY_DISABLE_INLINE_MEDIA_PREPROCESS_ENV,
    )
    .map(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
    .unwrap_or(false)
}

fn media_payload_to_tensor_packet(
    payload: &MediaPayload,
    options: &MediaTensorOptions,
) -> InferResult<BinaryTensorPacket> {
    if options.frame_stride == 0 {
        return Err(InferRequestError::bad_request(
            "`frame_stride` must be >= 1",
        ));
    }

    if let Some(frame_count) = options.frame_count {
        if frame_count == 0 {
            return Err(InferRequestError::bad_request(
                "`frame_count` must be >= 1 when provided",
            ));
        }
    }

    if let (Some(start), Some(end)) = (options.start_time_ms, options.end_time_ms) {
        if end <= start {
            return Err(InferRequestError::bad_request(
                "`end_time_ms` must be greater than `start_time_ms`",
            ));
        }
    }

    let media_bytes = decode_base64_payload(&payload.data_base64)?;
    let media_kind = detect_media_kind(payload);

    match media_kind {
        MediaKind::Image => image_bytes_to_tensor_packet(&media_bytes, options),
        MediaKind::Video => video_bytes_to_tensor_packet(&media_bytes, options),
    }
}

fn decode_base64_payload(payload: &str) -> InferResult<Vec<u8>> {
    let trimmed = payload.trim();
    let encoded = if let Some((prefix, data)) = trimmed.split_once(',') {
        if prefix.contains("base64") {
            data
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(encoded))
        .map_err(|e| {
            InferRequestError::bad_request(format!("Failed to decode base64 payload: {}", e))
        })
}

fn detect_media_kind(payload: &MediaPayload) -> MediaKind {
    if let Some(kind) = payload.kind {
        return kind;
    }

    if let Some(mime) = payload.mime_type.as_deref() {
        if mime.starts_with("video/") {
            return MediaKind::Video;
        }
        if mime.starts_with("image/") {
            return MediaKind::Image;
        }
    }

    if payload.data_base64.trim_start().starts_with("data:video/") {
        return MediaKind::Video;
    }

    MediaKind::Image
}

fn image_bytes_to_tensor_packet(
    image_bytes: &[u8],
    options: &MediaTensorOptions,
) -> InferResult<BinaryTensorPacket> {
    let image = image::load_from_memory(image_bytes).map_err(|e| {
        InferRequestError::bad_request(format!("Failed to decode image payload: {}", e))
    })?;
    let (frame, width, height, channels) = preprocess_image_frame(image, options)?;
    frames_to_tensor_packet(vec![frame], width, height, channels, options)
}

fn video_bytes_to_tensor_packet(
    video_bytes: &[u8],
    options: &MediaTensorOptions,
) -> InferResult<BinaryTensorPacket> {
    let mut temp_dir_path = std::env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    temp_dir_path.push(format!("kapsl-video-{}-{}", std::process::id(), timestamp));
    fs::create_dir_all(&temp_dir_path).map_err(|e| {
        InferRequestError::internal(format!("Failed to create temporary video directory: {}", e))
    })?;
    set_private_dir_permissions(&temp_dir_path)?;
    let temp_dir = TempDirGuard::new(temp_dir_path);

    let input_path = temp_dir.path().join("input-video.bin");
    write_private_temp_file(&input_path, video_bytes)?;

    let frame_pattern = temp_dir.path().join("frame_%06d.png");
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y");
    if let Some(start_time_ms) = options.start_time_ms {
        ffmpeg
            .arg("-ss")
            .arg(format!("{:.3}", start_time_ms as f64 / 1000.0));
    }
    ffmpeg.arg("-i").arg(&input_path);
    if let Some(end_time_ms) = options.end_time_ms {
        ffmpeg
            .arg("-to")
            .arg(format!("{:.3}", end_time_ms as f64 / 1000.0));
    }
    if options.frame_stride > 1 {
        ffmpeg
            .arg("-vf")
            .arg(format!("select=not(mod(n\\,{}))", options.frame_stride))
            .arg("-vsync")
            .arg("vfr");
    }

    let frame_count = options.frame_count.unwrap_or(1);
    ffmpeg.arg("-frames:v").arg(frame_count.to_string());
    ffmpeg.arg(&frame_pattern);

    let ffmpeg_output = ffmpeg.output().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            InferRequestError::internal(
                "ffmpeg is required for video infer payloads but was not found in PATH",
            )
        } else {
            InferRequestError::internal(format!("Failed to execute ffmpeg: {}", e))
        }
    })?;
    if !ffmpeg_output.status.success() {
        let stderr = String::from_utf8_lossy(&ffmpeg_output.stderr);
        if stderr.trim().is_empty() {
            return Err(InferRequestError::internal(
                "ffmpeg failed to extract frames without stderr output",
            ));
        }
        return Err(InferRequestError::bad_request(format!(
            "ffmpeg failed to extract frames: {}",
            stderr.trim()
        )));
    }

    let mut frame_paths = Vec::new();
    for entry in fs::read_dir(temp_dir.path()).map_err(|e| {
        InferRequestError::internal(format!("Failed to list extracted video frames: {}", e))
    })? {
        let entry = entry.map_err(|e| {
            InferRequestError::internal(format!("Failed to read extracted frame entry: {}", e))
        })?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("frame_"))
        {
            frame_paths.push(path);
        }
    }
    frame_paths.sort();

    if frame_paths.is_empty() {
        return Err(InferRequestError::bad_request(
            "No frames were extracted from video payload",
        ));
    }

    let mut frames = Vec::with_capacity(frame_paths.len());
    let mut width = 0u32;
    let mut height = 0u32;
    let mut channels = 0usize;

    for frame_path in frame_paths {
        let frame_image = image::open(&frame_path).map_err(|e| {
            InferRequestError::bad_request(format!(
                "Failed to decode extracted frame {:?}: {}",
                frame_path, e
            ))
        })?;
        let (frame_data, frame_width, frame_height, frame_channels) =
            preprocess_image_frame(frame_image, options)?;
        if frames.is_empty() {
            width = frame_width;
            height = frame_height;
            channels = frame_channels;
        }
        frames.push(frame_data);
    }

    frames_to_tensor_packet(frames, width, height, channels, options)
}

#[cfg(unix)]
fn set_private_dir_permissions(path: &Path) -> InferResult<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|e| {
        InferRequestError::internal(format!(
            "Failed to set private permissions on temporary video directory: {}",
            e
        ))
    })
}

#[cfg(not(unix))]
fn set_private_dir_permissions(_path: &Path) -> InferResult<()> {
    Ok(())
}

#[cfg(unix)]
fn write_private_temp_file(path: &Path, contents: &[u8]) -> InferResult<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| {
            InferRequestError::internal(format!("Failed to write temporary video input: {}", e))
        })?;
    file.write_all(contents).map_err(|e| {
        InferRequestError::internal(format!("Failed to write temporary video input: {}", e))
    })
}

#[cfg(not(unix))]
fn write_private_temp_file(path: &Path, contents: &[u8]) -> InferResult<()> {
    fs::write(path, contents).map_err(|e| {
        InferRequestError::internal(format!("Failed to write temporary video input: {}", e))
    })
}

fn preprocess_image_frame(
    mut image: DynamicImage,
    options: &MediaTensorOptions,
) -> InferResult<(Vec<u8>, u32, u32, usize)> {
    let target_width = options.target_width.unwrap_or_else(|| image.width());
    let target_height = options.target_height.unwrap_or_else(|| image.height());
    if target_width == 0 || target_height == 0 {
        return Err(InferRequestError::bad_request(
            "`target_width` and `target_height` must be > 0",
        ));
    }

    if image.width() != target_width || image.height() != target_height {
        image = image.resize_exact(target_width, target_height, FilterType::Triangle);
    }

    match options.channels {
        MediaChannelMode::Grayscale => {
            let gray = image.to_luma8();
            Ok((gray.into_raw(), target_width, target_height, 1))
        }
        MediaChannelMode::Rgb => {
            let rgb = image.to_rgb8();
            Ok((rgb.into_raw(), target_width, target_height, 3))
        }
        MediaChannelMode::Bgr => {
            let mut rgb = image.to_rgb8().into_raw();
            for chunk in rgb.chunks_exact_mut(3) {
                chunk.swap(0, 2);
            }
            Ok((rgb, target_width, target_height, 3))
        }
    }
}

fn normalize_pixel(value: u8, normalization: PixelNormalization) -> f32 {
    match normalization {
        PixelNormalization::None => value as f32,
        PixelNormalization::ZeroToOne => value as f32 / 255.0,
        PixelNormalization::MinusOneToOne => value as f32 / 127.5 - 1.0,
        PixelNormalization::Auto => value as f32 / 255.0,
    }
}

fn frames_to_tensor_packet(
    frames: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    channels: usize,
    options: &MediaTensorOptions,
) -> InferResult<BinaryTensorPacket> {
    if frames.is_empty() {
        return Err(InferRequestError::bad_request(
            "At least one frame is required for media infer payload",
        ));
    }

    let frame_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| InferRequestError::internal("Frame size overflow while building tensor"))?;
    let expected_frame_len = frame_pixels.checked_mul(channels).ok_or_else(|| {
        InferRequestError::internal("Frame buffer size overflow while building tensor")
    })?;

    for frame in &frames {
        if frame.len() != expected_frame_len {
            return Err(InferRequestError::bad_request(format!(
                "Inconsistent frame size: expected {} bytes, got {} bytes",
                expected_frame_len,
                frame.len()
            )));
        }
    }

    let mut ordered_u8 = Vec::with_capacity(
        expected_frame_len
            .checked_mul(frames.len())
            .ok_or_else(|| InferRequestError::internal("Tensor buffer size overflow"))?,
    );

    match options.layout {
        MediaTensorLayout::Nhwc => {
            for frame in &frames {
                ordered_u8.extend_from_slice(frame);
            }
        }
        MediaTensorLayout::Nchw => {
            for frame in &frames {
                for channel in 0..channels {
                    for pixel in 0..frame_pixels {
                        ordered_u8.push(frame[pixel * channels + channel]);
                    }
                }
            }
        }
    }

    let normalization = options.resolved_normalization();
    let data = match options.dtype {
        TensorDtype::Uint8 => {
            if !matches!(normalization, PixelNormalization::None) {
                return Err(InferRequestError::bad_request(
                    "Normalization is only supported for floating-point media dtypes",
                ));
            }
            ordered_u8
        }
        TensorDtype::Float32 => {
            let mut out = Vec::with_capacity(ordered_u8.len() * 4);
            for value in ordered_u8 {
                out.extend_from_slice(&normalize_pixel(value, normalization).to_ne_bytes());
            }
            out
        }
        TensorDtype::Float64 => {
            let mut out = Vec::with_capacity(ordered_u8.len() * 8);
            for value in ordered_u8 {
                out.extend_from_slice(
                    &(normalize_pixel(value, normalization) as f64).to_ne_bytes(),
                );
            }
            out
        }
        other => {
            return Err(InferRequestError::bad_request(format!(
                "Unsupported media output dtype: {}. Supported: uint8, float32, float64",
                other
            )))
        }
    };

    let frame_count = frames.len() as i64;
    let shape = match options.layout {
        MediaTensorLayout::Nchw => vec![frame_count, channels as i64, height as i64, width as i64],
        MediaTensorLayout::Nhwc => vec![frame_count, height as i64, width as i64, channels as i64],
    };

    Ok(BinaryTensorPacket {
        shape,
        dtype: options.dtype,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, RgbImage};
    use serde_json::json;
    use std::io::Cursor;

    fn png_base64_1x1_rgb(r: u8, g: u8, b: u8) -> String {
        let image = RgbImage::from_raw(1, 1, vec![r, g, b]).expect("valid 1x1 rgb image");
        let mut cursor = Cursor::new(Vec::<u8>::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut cursor, ImageFormat::Png)
            .expect("encode png");
        base64::engine::general_purpose::STANDARD.encode(cursor.into_inner())
    }

    #[test]
    fn test_parse_tensor_payload_still_supported() {
        let payload = json!({
            "input": {
                "shape": [1, 1, 1, 1],
                "dtype": "float32",
                "data": [0, 0, 128, 63]
            }
        });
        let registry = default_request_adapter_registry();
        let request = parse_inference_request_with_registry(payload, "onnx", &registry)
            .expect("tensor payload parses");
        assert_eq!(request.input.shape, vec![1, 1, 1, 1]);
        assert_eq!(request.input.dtype, TensorDtype::Float32);
        assert_eq!(request.input.data.len(), 4);
    }

    #[test]
    fn test_parse_media_image_payload_to_float32_nchw() {
        let payload = json!({
            "media": {
                "kind": "image",
                "base64": png_base64_1x1_rgb(255, 0, 0)
            },
            "tensor_options": {
                "dtype": "float32",
                "layout": "nchw",
                "channels": "rgb",
                "normalize": "zero_to_one"
            }
        });

        let registry = default_request_adapter_registry();
        let request = parse_inference_request_with_registry(payload, "onnx", &registry)
            .expect("media payload parses");
        assert_eq!(request.input.shape, vec![1, 3, 1, 1]);
        assert_eq!(request.input.dtype, TensorDtype::Float32);

        let values: Vec<f32> = request
            .input
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("4 bytes per f32")))
            .collect();
        assert_eq!(values, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_parse_media_image_with_additional_media_inputs() {
        let payload = json!({
            "media": {
                "kind": "image",
                "base64": png_base64_1x1_rgb(0, 255, 0)
            },
            "additional_media_inputs": [
                {
                    "name": "aux_image",
                    "media": {
                        "kind": "image",
                        "base64": png_base64_1x1_rgb(0, 0, 255)
                    }
                }
            ],
            "tensor_options": {
                "dtype": "uint8",
                "layout": "nhwc",
                "channels": "rgb",
                "normalize": "none"
            }
        });

        let registry = default_request_adapter_registry();
        let request = parse_inference_request_with_registry(payload, "onnx", &registry)
            .expect("media payload parses");
        assert_eq!(request.input.shape, vec![1, 1, 1, 3]);
        assert_eq!(request.additional_inputs.len(), 1);
        assert_eq!(request.additional_inputs[0].name, "aux_image");
        assert_eq!(request.additional_inputs[0].tensor.shape, vec![1, 1, 1, 3]);
        assert_eq!(
            request.additional_inputs[0].tensor.dtype,
            TensorDtype::Uint8
        );
    }

    struct DummyFrameworkAdapter;

    impl ModelRequestAdapter for DummyFrameworkAdapter {
        fn name(&self) -> &'static str {
            "dummy_framework"
        }

        fn supported_frameworks(&self) -> &'static [&'static str] {
            &["dummy-framework"]
        }

        fn supports_payload(&self, payload: &serde_json::Value) -> bool {
            payload.get("dummy").is_some()
        }

        fn adapt(&self, _payload: serde_json::Value) -> InferResult<InferenceRequest> {
            Ok(InferenceRequest::new(BinaryTensorPacket {
                shape: vec![1],
                dtype: TensorDtype::Uint8,
                data: vec![7],
            }))
        }
    }

    #[test]
    fn test_adapter_registry_routes_framework_specific_adapter() {
        let mut registry = RequestAdapterRegistry::new();
        registry.register(Box::new(DummyFrameworkAdapter));
        let request = parse_inference_request_with_registry(
            json!({ "dummy": true }),
            "dummy-framework",
            &registry,
        )
        .expect("framework-specific adapter should parse payload");
        assert_eq!(request.input.shape, vec![1]);
        assert_eq!(request.input.dtype, TensorDtype::Uint8);
        assert_eq!(request.input.data, vec![7]);
    }

    #[test]
    fn test_adapter_registry_rejects_framework_mismatch() {
        let mut registry = RequestAdapterRegistry::new();
        registry.register(Box::new(DummyFrameworkAdapter));
        let err = parse_inference_request_with_registry(
            json!({ "dummy": true }),
            "other-framework",
            &registry,
        )
        .expect_err("framework mismatch must fail");
        assert!(err
            .to_string()
            .contains("No infer adapter supports framework"));
    }

    #[test]
    fn test_optional_adapter_spec_ignores_unknown_token() {
        let mut registry = RequestAdapterRegistry::new();
        registry.register(Box::new(TensorRequestAdapter));
        apply_optional_adapter_spec(&mut registry, "unknown_adapter");
        assert!(registry.has_adapter_name("tensor_json"));
        assert!(!registry.has_adapter_name("unknown_adapter"));
    }

    #[cfg(feature = "infer-adapter-echo")]
    #[test]
    fn test_optional_adapter_spec_registers_echo_adapter_with_feature() {
        let mut registry = RequestAdapterRegistry::new();
        registry.register(Box::new(TensorRequestAdapter));
        apply_optional_adapter_spec(&mut registry, "echo_tensor");
        assert!(registry.has_adapter_name("echo_tensor"));

        let request = parse_inference_request_with_registry(
            json!({
                "echo_input": {
                    "shape": [1],
                    "dtype": "uint8",
                    "data": [9]
                }
            }),
            "onnx",
            &registry,
        )
        .expect("echo adapter should parse echo payload when feature is enabled");
        assert_eq!(request.input.shape, vec![1]);
        assert_eq!(request.input.dtype, TensorDtype::Uint8);
        assert_eq!(request.input.data, vec![9]);
    }

    #[cfg(not(feature = "infer-adapter-echo"))]
    #[test]
    fn test_optional_adapter_spec_does_not_register_echo_adapter_without_feature() {
        let mut registry = RequestAdapterRegistry::new();
        registry.register(Box::new(TensorRequestAdapter));
        apply_optional_adapter_spec(&mut registry, "echo_tensor");
        assert!(!registry.has_adapter_name("echo_tensor"));
    }
}
