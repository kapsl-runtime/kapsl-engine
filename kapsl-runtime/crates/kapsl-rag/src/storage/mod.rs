use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum DocStoreError {
    #[error("io error: {0}")]
    Io(String),
}

impl From<io::Error> for DocStoreError {
    fn from(err: io::Error) -> Self {
        DocStoreError::Io(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct DocKey {
    pub tenant_id: String,
    pub workspace_id: String,
    pub source_id: String,
    pub doc_id: String,
}

pub trait DocStore: Send + Sync {
    fn put(&self, key: &DocKey, bytes: &[u8]) -> Result<PathBuf, DocStoreError>;
    fn get(&self, key: &DocKey) -> Result<Vec<u8>, DocStoreError>;
    fn delete(&self, key: &DocKey) -> Result<(), DocStoreError>;
}

#[derive(Debug, Clone)]
pub struct FsDocStore {
    pub root: PathBuf,
}

impl FsDocStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path_for(&self, key: &DocKey) -> PathBuf {
        self.root
            .join(&key.tenant_id)
            .join(&key.workspace_id)
            .join(&key.source_id)
            .join(&key.doc_id)
    }
}

impl DocStore for FsDocStore {
    fn put(&self, key: &DocKey, bytes: &[u8]) -> Result<PathBuf, DocStoreError> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, bytes)?;
        Ok(path)
    }

    fn get(&self, key: &DocKey) -> Result<Vec<u8>, DocStoreError> {
        let path = self.path_for(key);
        let data = fs::read(path)?;
        Ok(data)
    }

    fn delete(&self, key: &DocKey) -> Result<(), DocStoreError> {
        let path = self.path_for(key);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}
