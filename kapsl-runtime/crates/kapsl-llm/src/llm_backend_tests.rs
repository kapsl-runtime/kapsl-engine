#[cfg(test)]
mod tests {
    use super::super::{extract_bos_token, extract_tag, load_model_runtime_config, LLMBackend};
    use crate::sequence::{FinishReason, SequenceGroupOutput};
    use futures::StreamExt;
    use kapsl_engine_api::{BinaryTensorPacket, Engine, InferenceRequest, TensorDtype};
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use tokio::sync::mpsc;

    fn make_temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kapsl_llm_llm_backend_{}_{}",
            label,
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn extract_bos_token_from_tokenizer_json() {
        let tokenizer = json!({
            "post_processor": {
                "single": [
                    { "SpecialToken": { "id": "<s>" } }
                ]
            }
        });
        assert_eq!(extract_bos_token(&tokenizer), Some("<s>".to_string()));

        let tokenizer = json!({
            "post_processor": {
                "special_tokens": {
                    "<bos>": {}
                }
            }
        });
        assert_eq!(extract_bos_token(&tokenizer), Some("<bos>".to_string()));
    }

    #[test]
    fn extract_tag_finds_nearest_angle_brackets() {
        let template = "<|user|> User: {{prompt}} <|assistant|>";
        assert_eq!(extract_tag(template, "User"), Some("<|user|>".to_string()));
    }

    #[test]
    fn load_model_runtime_config_reads_generation_and_template_defaults() {
        let root = make_temp_dir("cfg");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");

        let generation = json!({
            "temperature": 0.5,
            "max_new_tokens": 128,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.2,
            "eos_token_id": 2,
            "bos_token_id": 1
        });
        fs::write(root.join("generation_config.json"), generation.to_string())
            .expect("generation config");

        let cfg = json!({ "model_type": "qwen2" });
        fs::write(root.join("config.json"), cfg.to_string()).expect("config.json");

        let tokenizer = json!({
            "post_processor": {
                "special_tokens": {
                    "<bos>": {}
                }
            }
        });
        fs::write(root.join("tokenizer.json"), tokenizer.to_string()).expect("tokenizer.json");

        let template = "<|user|> User: {{prompt}} <|assistant|>";
        fs::write(root.join("chat_template.jinja"), template).expect("chat template");

        let runtime = load_model_runtime_config(&model_path);
        assert!(runtime.use_chat_template);
        assert_eq!(runtime.prompt_prefix, "<bos><|user|>");
        assert_eq!(runtime.prompt_suffix, "<|assistant|>");

        let sampling = runtime.sampling;
        assert_eq!(sampling.max_tokens, 128);
        assert!((sampling.temperature - 0.5).abs() < f32::EPSILON);
        assert!((sampling.top_p - 0.8).abs() < f32::EPSILON);
        assert_eq!(sampling.top_k, 20);
        assert!((sampling.repetition_penalty - 1.2).abs() < f32::EPSILON);
        assert_eq!(sampling.stop_token_ids, vec![2, 1]);
    }

    #[test]
    fn load_model_runtime_config_accepts_array_eos_token_id() {
        let root = make_temp_dir("cfg_eos_array");
        let model_path = root.join("model.onnx");
        fs::write(&model_path, "").expect("model file");

        let generation = json!({
            "eos_token_id": [1, 106],
            "bos_token_id": 2
        });
        fs::write(root.join("generation_config.json"), generation.to_string())
            .expect("generation config");

        let runtime = load_model_runtime_config(&model_path);
        assert_eq!(runtime.sampling.stop_token_ids, vec![1, 106, 2]);
    }

    #[tokio::test]
    async fn infer_stream_handles_cumulative_and_incremental_outputs() {
        let backend = LLMBackend::new();
        let (tx, mut rx) = mpsc::channel(1);
        *backend.request_tx.lock().unwrap() = Some(tx);

        let request = InferenceRequest {
            input: BinaryTensorPacket {
                shape: vec![1, 2],
                dtype: TensorDtype::Utf8,
                data: b"Hi".to_vec(),
            },
            additional_inputs: Vec::new(),
            session_id: None,
            metadata: None,
            cancellation: None,
        };

        let stream: std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<BinaryTensorPacket, kapsl_engine_api::EngineError>,
                    > + Send,
            >,
        > = backend.infer_stream(&request);

        let handle = tokio::spawn(async move {
            let mut stream = stream;
            let mut chunks = Vec::new();
            while let Some(packet_res) = stream.next().await {
                let packet: BinaryTensorPacket = match packet_res {
                    Ok(packet) => packet,
                    Err(err) => panic!("stream err: {}", err),
                };
                chunks.push(String::from_utf8(packet.data).expect("utf8"));
            }
            chunks
        });

        let seq_group = rx.recv().await.expect("seq_group");

        seq_group
            .response_tx
            .send(SequenceGroupOutput {
                request_id: seq_group.request_id.clone(),
                text: "Hel".to_string(),
                finish_reason: None,
            })
            .await
            .expect("send first chunk");

        seq_group
            .response_tx
            .send(SequenceGroupOutput {
                request_id: seq_group.request_id.clone(),
                text: "lo".to_string(),
                finish_reason: Some(FinishReason::Stop),
            })
            .await
            .expect("send second chunk");

        let chunks = handle.await.expect("join stream task");
        assert_eq!(chunks, vec!["Hel".to_string(), "lo".to_string()]);
    }
}
