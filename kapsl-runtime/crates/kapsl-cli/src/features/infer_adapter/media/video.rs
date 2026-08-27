use super::super::error::{InferRequestError, InferResult};
use super::image::preprocess_image_frame;
use super::temp::{set_private_dir_permissions, write_private_temp_file, TempDirGuard};
use super::tensor::frames_to_tensor_packet;
use super::types::MediaTensorOptions;
use kapsl_engine_api::BinaryTensorPacket;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn video_bytes_to_tensor_packet(
    video_bytes: &[u8],
    options: &MediaTensorOptions,
) -> InferResult<BinaryTensorPacket> {
    let mut temp_dir_path = std::env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    temp_dir_path.push(format!("kapsl-video-{}-{}", std::process::id(), timestamp));
    fs::create_dir_all(&temp_dir_path).map_err(|error| {
        InferRequestError::internal(format!(
            "Failed to create temporary video directory: {}",
            error
        ))
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

    let ffmpeg_output = ffmpeg.output().map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            InferRequestError::internal(
                "ffmpeg is required for video infer payloads but was not found in PATH",
            )
        } else {
            InferRequestError::internal(format!("Failed to execute ffmpeg: {}", error))
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
    for entry in fs::read_dir(temp_dir.path()).map_err(|error| {
        InferRequestError::internal(format!("Failed to list extracted video frames: {}", error))
    })? {
        let entry = entry.map_err(|error| {
            InferRequestError::internal(format!("Failed to read extracted frame entry: {}", error))
        })?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
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
        let frame_image = ::image::open(&frame_path).map_err(|error| {
            InferRequestError::bad_request(format!(
                "Failed to decode extracted frame {:?}: {}",
                frame_path, error
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
