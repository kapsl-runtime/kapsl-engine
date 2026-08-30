use kapsl_engine_api::{NamedTensor, RequestMetadata as EngineRequestMetadata, TensorDtype};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct MediaPayload {
    #[serde(default)]
    pub(super) kind: Option<MediaKind>,
    #[serde(default)]
    pub(super) mime_type: Option<String>,
    #[serde(alias = "base64")]
    pub(super) data_base64: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MediaTensorLayout {
    Nchw,
    Nhwc,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MediaChannelMode {
    Rgb,
    Bgr,
    Grayscale,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PixelNormalization {
    Auto,
    None,
    ZeroToOne,
    MinusOneToOne,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(super) struct MediaTensorOptions {
    pub(super) target_width: Option<u32>,
    pub(super) target_height: Option<u32>,
    pub(super) layout: MediaTensorLayout,
    pub(super) channels: MediaChannelMode,
    pub(super) dtype: TensorDtype,
    pub(super) normalize: PixelNormalization,
    pub(super) frame_count: Option<usize>,
    pub(super) frame_stride: usize,
    pub(super) start_time_ms: Option<u64>,
    pub(super) end_time_ms: Option<u64>,
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
    pub(super) fn resolved_normalization(&self) -> PixelNormalization {
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
pub(super) struct NamedMediaPayload {
    pub(super) name: String,
    pub(super) media: MediaPayload,
    #[serde(default)]
    pub(super) tensor_options: Option<MediaTensorOptions>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct HttpMediaInferenceRequest {
    pub(super) media: MediaPayload,
    #[serde(default)]
    pub(super) additional_media_inputs: Vec<NamedMediaPayload>,
    #[serde(default)]
    pub(super) additional_inputs: Vec<NamedTensor>,
    #[serde(default)]
    pub(super) session_id: Option<String>,
    #[serde(default)]
    pub(super) metadata: Option<EngineRequestMetadata>,
    #[serde(default, alias = "preprocess", alias = "options")]
    pub(super) tensor_options: MediaTensorOptions,
}
