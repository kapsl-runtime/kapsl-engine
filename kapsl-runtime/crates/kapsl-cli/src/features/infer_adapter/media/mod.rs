mod image;
mod temp;
mod tensor;
mod types;
mod video;

use self::image::image_bytes_to_tensor_packet;
use self::types::{HttpMediaInferenceRequest, MediaKind, MediaPayload, MediaTensorOptions};
use self::video::video_bytes_to_tensor_packet;
use super::error::{InferRequestError, InferResult};
use super::ModelRequestAdapter;
use base64::Engine as _;
use kapsl_engine_api::{BinaryTensorPacket, InferenceRequest, NamedTensor};

const DISABLE_INLINE_MEDIA_PREPROCESS_ENV: &str = "KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS";

pub(super) struct MediaRequestAdapter;

impl ModelRequestAdapter for MediaRequestAdapter {
    fn name(&self) -> &'static str {
        "media_base64"
    }

    fn supports_payload(&self, payload: &serde_json::Value) -> bool {
        payload.get("media").is_some()
    }

    fn adapt(&self, payload: serde_json::Value) -> InferResult<InferenceRequest> {
        let media_request =
            serde_json::from_value::<HttpMediaInferenceRequest>(payload).map_err(|error| {
                InferRequestError::bad_request(format!("Invalid media infer payload: {}", error))
            })?;
        media_infer_request_to_inference_request(media_request)
    }
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
    std::env::var(DISABLE_INLINE_MEDIA_PREPROCESS_ENV)
        .ok()
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
    match detect_media_kind(payload) {
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
        .map_err(|error| {
            InferRequestError::bad_request(format!("Failed to decode base64 payload: {}", error))
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
