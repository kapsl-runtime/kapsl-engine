use super::super::error::{InferRequestError, InferResult};
use super::tensor::frames_to_tensor_packet;
use super::types::{MediaChannelMode, MediaTensorOptions};
use ::image::{imageops::FilterType, DynamicImage};
use kapsl_engine_api::BinaryTensorPacket;

pub(super) fn image_bytes_to_tensor_packet(
    image_bytes: &[u8],
    options: &MediaTensorOptions,
) -> InferResult<BinaryTensorPacket> {
    let image = ::image::load_from_memory(image_bytes).map_err(|error| {
        InferRequestError::bad_request(format!("Failed to decode image payload: {}", error))
    })?;
    let (frame, width, height, channels) = preprocess_image_frame(image, options)?;
    frames_to_tensor_packet(vec![frame], width, height, channels, options)
}

pub(super) fn preprocess_image_frame(
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
