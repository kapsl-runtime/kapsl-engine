# Media Inference Readiness Checklist

This checklist tracks production readiness for image/video inference via `POST /api/models/:id/infer`.

## Current Baseline

- Status: Partial
- Image path: Beta
- Video path: Experimental
- Multimodal LLM/VLM chat path: Not complete

## API and Adapter Coverage

- [x] Tensor payload adapter is available (`input` JSON tensor payloads).
- [x] Media payload adapter is available (`media` base64 payloads).
- [x] `additional_media_inputs` is supported and converted to `additional_inputs`.
- [x] Media kind auto-detection exists (explicit kind, mime, data URI prefix).
- [ ] Public API docs include media request schema examples.

## Image Preprocessing

- [x] Base64 decode with data URI support.
- [x] Resize options (`target_width`, `target_height`).
- [x] Layout options (`nchw`, `nhwc`).
- [x] Channel options (`rgb`, `bgr`, `grayscale`).
- [x] Dtype options (`uint8`, `float32`, `float64`).
- [x] Pixel normalization options (`none`, `zero_to_one`, `minus_one_to_one`, `auto`).

## Video Preprocessing

- [x] Video bytes are accepted in media payload.
- [x] Frame extraction implemented through `ffmpeg`.
- [x] Sampling controls (`frame_count`, `frame_stride`, `start_time_ms`, `end_time_ms`).
- [x] Secure temp file handling with cleanup.
- [ ] Dedicated integration tests for video payload path.
- [ ] Runtime startup validation for `ffmpeg` availability when video path is intended.

## Backend Compatibility

- [x] ONNX backend supports numeric tensor inputs and outputs.
- [x] ONNX backend supports additional named tensor inputs.
- [ ] Framework-specific adapters for non-ONNX VLM formats.
- [ ] Strong validation that media-derived tensor shape matches model expectations before execution.

## LLM and RAG Interactions

- [x] LLM backend supports UTF-8 string input.
- [ ] LLM backend accepts media tensors directly.
- [ ] Combined image-plus-text multimodal prompting for LLM backend.

## Reliability and Observability

- [x] Adapter-level validation and clear bad-request errors.
- [x] Infer route logs parse and execution failures.
- [ ] Response timing and preprocessing breakdown metrics for media path.
- [ ] Structured error codes for media-specific failures (`decode_failed`, `ffmpeg_missing`, `shape_mismatch`).

## Testing and CI

- [x] Unit tests exist for media image adapter transformations.
- [x] New E2E script added for image and optional video infer checks (`testing-inference/test_media_infer.py`).
- [ ] CI job runs media E2E test against a known vision model package.
- [ ] Golden-output regression test for deterministic media preprocessing.

## Operational Controls

- [x] Inline media preprocessing can be disabled via `KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS` (legacy `KAPSL_DISABLE_INLINE_MEDIA_PREPROCESS` also supported).
- [x] Throughput profile no longer disables media preprocessing by default.
- [ ] Add explicit CLI flag for disabling inline preprocessing (instead of env-only control).

## Recommended Next Milestones

1. Add a CI media E2E lane with a packaged ONNX vision model.
2. Add video integration tests with synthetic clip generation and runtime assertions.
3. Add framework-specific adapters for VLM-style multimodal models.
4. Add structured media error codes and preprocessing latency metrics.
