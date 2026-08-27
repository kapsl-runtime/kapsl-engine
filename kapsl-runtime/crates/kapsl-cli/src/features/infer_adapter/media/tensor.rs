use super::super::error::{InferRequestError, InferResult};
use super::types::{MediaTensorLayout, MediaTensorOptions, PixelNormalization};
use kapsl_engine_api::{BinaryTensorPacket, TensorDtype};

fn normalize_pixel(value: u8, normalization: PixelNormalization) -> f32 {
    match normalization {
        PixelNormalization::None => value as f32,
        PixelNormalization::ZeroToOne => value as f32 / 255.0,
        PixelNormalization::MinusOneToOne => value as f32 / 127.5 - 1.0,
        PixelNormalization::Auto => value as f32 / 255.0,
    }
}

pub(super) fn frames_to_tensor_packet(
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
            let mut output = Vec::with_capacity(ordered_u8.len() * 4);
            for value in ordered_u8 {
                output.extend_from_slice(&normalize_pixel(value, normalization).to_ne_bytes());
            }
            output
        }
        TensorDtype::Float64 => {
            let mut output = Vec::with_capacity(ordered_u8.len() * 8);
            for value in ordered_u8 {
                output.extend_from_slice(
                    &(normalize_pixel(value, normalization) as f64).to_ne_bytes(),
                );
            }
            output
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
