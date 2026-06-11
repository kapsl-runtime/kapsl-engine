#[cfg(test)]
mod tests {
    use super::super::{find_model_asset, find_model_root};
    use std::fs;
    use std::path::PathBuf;

    fn make_temp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("kapsl_llm_{}_{}", label, uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn find_model_root_walks_to_metadata() {
        let root = make_temp_dir("root");
        fs::write(root.join("metadata.json"), "{}").expect("metadata");

        let model_dir = root.join("nested").join("model");
        fs::create_dir_all(&model_dir).expect("model dir");
        let model_path = model_dir.join("model.onnx");
        fs::write(&model_path, "").expect("model file");

        let found = find_model_root(&model_path);
        assert_eq!(found, root);
    }

    #[test]
    fn find_model_asset_prefers_model_dir_then_root_fallbacks() {
        let root = make_temp_dir("asset");
        fs::write(root.join("metadata.json"), "{}").expect("metadata");

        let model_dir = root.join("model");
        fs::create_dir_all(&model_dir).expect("model dir");
        let model_path = model_dir.join("model.onnx");
        fs::write(&model_path, "").expect("model file");

        let direct = model_dir.join("tokenizer.json");
        fs::write(&direct, "{}").expect("direct file");

        let found_direct = find_model_asset(&model_path, "tokenizer.json").unwrap();
        assert_eq!(found_direct, direct);

        fs::remove_file(&direct).expect("remove direct");
        let export_dir = root.join("onnx-export");
        fs::create_dir_all(&export_dir).expect("onnx-export dir");
        let export_file = export_dir.join("tokenizer.json");
        fs::write(&export_file, "{}").expect("export file");

        let found_export = find_model_asset(&model_path, "tokenizer.json").unwrap();
        assert_eq!(found_export, export_file);
    }
}
