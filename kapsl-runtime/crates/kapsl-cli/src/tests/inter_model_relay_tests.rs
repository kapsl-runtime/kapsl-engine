use super::*;

#[test]
fn test_parse_inter_model_routes_supports_multiple_separators() {
    let routes = parse_inter_model_routes(
        "vision=reasoner;audio=reasoner,transcriber\nmonitor->alerter;vision=reasoner",
    );

    assert_eq!(
        routes.get("vision"),
        Some(&vec!["reasoner".to_string()]),
        "duplicate targets should be deduplicated"
    );
    assert_eq!(
        routes.get("audio"),
        Some(&vec!["reasoner".to_string(), "transcriber".to_string()])
    );
    assert_eq!(routes.get("monitor"), Some(&vec!["alerter".to_string()]));
}

#[test]
fn test_relay_prompt_from_output_only_accepts_utf8_payloads() {
    let utf8_packet = BinaryTensorPacket {
        shape: vec![1, 5],
        dtype: TensorDtype::Utf8,
        data: b"hello".to_vec(),
    };
    let prompt = relay_prompt_from_output("vision", &utf8_packet).expect("utf8 prompt");
    assert!(prompt.contains("Report from vision"));
    assert!(prompt.contains("hello"));

    let non_utf8_packet = BinaryTensorPacket {
        shape: vec![1, 1],
        dtype: TensorDtype::Float32,
        data: vec![0, 0, 0, 0],
    };
    assert!(relay_prompt_from_output("vision", &non_utf8_packet).is_none());
}

#[test]
fn test_relay_state_rate_limits_per_source_model_id() {
    let state = InterModelRelayState {
        routes: parse_inter_model_routes("a=b"),
        min_interval: Duration::from_secs(60),
        last_relay_at: Arc::new(Mutex::new(HashMap::new())),
    };

    assert!(state.should_emit(7));
    assert!(!state.should_emit(7), "second emit should be rate limited");
    assert!(state.should_emit(8), "different source id should emit");
}
