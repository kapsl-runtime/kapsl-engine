#[cfg(test)]
mod tests {
    use super::super::{
        build_kv_array_f16, build_kv_array_f32_from_f16, empty_kv_shape, infer_kv_layout,
        parse_safe_load_setting, KvLayout, SafeLoadSetting,
    };
    use half::f16;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn parse_safe_load_setting_handles_bool_and_strings() {
        assert_eq!(
            parse_safe_load_setting(&json!(true)),
            Some(SafeLoadSetting::ForceOn)
        );
        assert_eq!(
            parse_safe_load_setting(&json!(false)),
            Some(SafeLoadSetting::ForceOff)
        );
        assert_eq!(
            parse_safe_load_setting(&json!("auto")),
            Some(SafeLoadSetting::Auto)
        );
        assert_eq!(
            parse_safe_load_setting(&json!("on")),
            Some(SafeLoadSetting::ForceOn)
        );
        assert_eq!(
            parse_safe_load_setting(&json!("off")),
            Some(SafeLoadSetting::ForceOff)
        );
        assert_eq!(parse_safe_load_setting(&json!("maybe")), None);
    }

    #[test]
    fn infer_kv_layout_prefers_head_dim_axis() {
        let mut shapes = HashMap::new();
        shapes.insert("past_key_values.0.key".to_string(), vec![1, 4, 8, 16]);
        assert!(matches!(
            infer_kv_layout(&shapes, 4, 8),
            KvLayout::HeadDimFirst
        ));

        shapes.insert("past_key_values.0.key".to_string(), vec![1, 4, 16, 8]);
        assert!(matches!(infer_kv_layout(&shapes, 4, 8), KvLayout::SeqFirst));

        let shapes = HashMap::new();
        assert!(matches!(infer_kv_layout(&shapes, 4, 8), KvLayout::SeqFirst));
    }

    #[test]
    fn build_kv_array_seq_first_layout() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let arr = build_kv_array_f16(&data, 1, 2, 2, KvLayout::SeqFirst, "key").expect("kv array");
        assert_eq!(arr.shape(), &[1, 1, 2, 2]);
        let got: Vec<f32> = arr.iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn build_kv_array_head_dim_first_layout() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let arr = build_kv_array_f32_from_f16(&data, 1, 2, 2, KvLayout::HeadDimFirst, "key")
            .expect("kv array");
        assert_eq!(arr.shape(), &[1, 1, 2, 2]);
        let got: Vec<f32> = arr.iter().cloned().collect();
        assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn build_kv_array_rejects_invalid_stride() {
        let data = vec![f16::from_f32(1.0)];
        assert!(build_kv_array_f16(&data, 0, 1, 1, KvLayout::SeqFirst, "key").is_err());
    }

    #[test]
    fn empty_kv_shape_prefers_layout_for_rank4() {
        let shape = empty_kv_shape(Some(&vec![1, 2, 3, 4]), KvLayout::SeqFirst, 8, 16);
        assert_eq!(shape, vec![1, 8, 0, 16]);

        let shape = empty_kv_shape(Some(&vec![2, -1, 5]), KvLayout::SeqFirst, 4, 8);
        assert_eq!(shape, vec![1, 1, 5]);

        let shape = empty_kv_shape(None, KvLayout::HeadDimFirst, 3, 7);
        assert_eq!(shape, vec![1, 3, 7, 0]);
    }
}
