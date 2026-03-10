use crate::kernel::AttentionConfig;

#[test]
fn attention_config_scale_default_matches_inverse_sqrt() {
    let config = AttentionConfig {
        scale: None,
        causal: true,
    };

    let value = config.scale_for(4);
    assert!((value - 0.5).abs() < 1e-6);
}

#[test]
fn attention_config_scale_prefers_override() {
    let config = AttentionConfig {
        scale: Some(0.25),
        causal: false,
    };

    let value = config.scale_for(128);
    assert!((value - 0.25).abs() < 1e-6);
}
