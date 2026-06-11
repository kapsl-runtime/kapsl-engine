use std::path::{Path, PathBuf};

pub fn find_model_root(model_path: &Path) -> PathBuf {
    let mut current = model_path.parent();
    while let Some(dir) = current {
        if dir.join("metadata.json").exists() {
            return dir.to_path_buf();
        }
        current = dir.parent();
    }

    model_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn find_model_asset(model_path: &Path, filename: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = model_path.parent() {
        candidates.push(dir.to_path_buf());
    }
    let model_root = find_model_root(model_path);
    if model_path
        .parent()
        .map(|dir| dir != model_root.as_path())
        .unwrap_or(true)
    {
        candidates.push(model_root);
    }

    for base in candidates {
        let direct = base.join(filename);
        if direct.exists() {
            return Some(direct);
        }
        let alt = base.join("onnx-export").join(filename);
        if alt.exists() {
            return Some(alt);
        }
    }

    None
}

#[path = "model_paths_tests.rs"]
mod model_paths_tests;
