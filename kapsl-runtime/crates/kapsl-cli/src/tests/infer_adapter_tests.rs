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
