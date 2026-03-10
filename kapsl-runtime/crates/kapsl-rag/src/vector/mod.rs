use async_trait::async_trait;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

#[derive(thiserror::Error, Debug)]
pub enum VectorStoreError {
    #[error("db error: {0}")]
    Db(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl From<rusqlite::Error> for VectorStoreError {
    fn from(err: rusqlite::Error) -> Self {
        VectorStoreError::Db(err.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessControl {
    pub allow_users: Vec<String>,
    pub allow_groups: Vec<String>,
    pub deny_users: Vec<String>,
    pub deny_groups: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    pub id: String,
    pub tenant_id: String,
    pub workspace_id: String,
    pub source_id: String,
    pub doc_id: String,
    pub chunk_index: i64,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub acl: AccessControl,
}

#[derive(Debug, Clone)]
pub struct VectorQuery {
    pub query_embedding: Vec<f32>,
    pub top_k: usize,
    pub tenant_id: String,
    pub workspace_id: String,
    pub source_ids: Option<Vec<String>>,
    pub allowed_users: Vec<String>,
    pub allowed_groups: Vec<String>,
    pub min_score: f32,
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub chunk: EmbeddedChunk,
    pub score: f32,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, chunks: Vec<EmbeddedChunk>) -> Result<(), VectorStoreError>;
    async fn delete_by_doc(
        &self,
        tenant_id: &str,
        workspace_id: &str,
        source_id: &str,
        doc_id: &str,
    ) -> Result<(), VectorStoreError>;
    async fn query(
        &self,
        request: VectorQuery,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError>;
}

pub struct SqliteVectorStore {
    conn: Mutex<Connection>,
}

impl SqliteVectorStore {
    pub fn open(path: &Path) -> Result<Self, VectorStoreError> {
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<(), VectorStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata_json TEXT,
                acl_allow_users TEXT,
                acl_allow_groups TEXT,
                acl_deny_users TEXT,
                acl_deny_groups TEXT,
                updated_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_scope
                ON rag_chunks (tenant_id, workspace_id, source_id, doc_id);
            ",
        )?;
        Ok(())
    }
}

#[async_trait]
impl VectorStore for SqliteVectorStore {
    async fn upsert(&self, chunks: Vec<EmbeddedChunk>) -> Result<(), VectorStoreError> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;
        let tx = conn.transaction()?;
        for chunk in chunks {
            let metadata_json = serde_json::to_string(&chunk.metadata)
                .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
            let acl_allow_users = serde_json::to_string(&chunk.acl.allow_users)
                .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
            let acl_allow_groups = serde_json::to_string(&chunk.acl.allow_groups)
                .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
            let acl_deny_users = serde_json::to_string(&chunk.acl.deny_users)
                .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
            let acl_deny_groups = serde_json::to_string(&chunk.acl.deny_groups)
                .map_err(|e| VectorStoreError::Serialization(e.to_string()))?;
            let embedding_blob = serialize_embedding(&chunk.embedding);
            tx.execute(
                "INSERT OR REPLACE INTO rag_chunks (
                    id, tenant_id, workspace_id, source_id, doc_id, chunk_index,
                    text, embedding, metadata_json, acl_allow_users, acl_allow_groups,
                    acl_deny_users, acl_deny_groups, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, strftime('%s','now'))",
                params![
                    chunk.id,
                    chunk.tenant_id,
                    chunk.workspace_id,
                    chunk.source_id,
                    chunk.doc_id,
                    chunk.chunk_index,
                    chunk.text,
                    embedding_blob,
                    metadata_json,
                    acl_allow_users,
                    acl_allow_groups,
                    acl_deny_users,
                    acl_deny_groups,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    async fn delete_by_doc(
        &self,
        tenant_id: &str,
        workspace_id: &str,
        source_id: &str,
        doc_id: &str,
    ) -> Result<(), VectorStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;
        conn.execute(
            "DELETE FROM rag_chunks WHERE tenant_id = ?1 AND workspace_id = ?2 AND source_id = ?3 AND doc_id = ?4",
            params![tenant_id, workspace_id, source_id, doc_id],
        )?;
        Ok(())
    }

    async fn query(
        &self,
        request: VectorQuery,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        if request.query_embedding.is_empty() {
            return Err(VectorStoreError::InvalidInput(
                "query embedding is empty".to_string(),
            ));
        }
        let conn = self
            .conn
            .lock()
            .map_err(|_| VectorStoreError::Db("vector store mutex poisoned".to_string()))?;

        let mut sql = String::from(
            "SELECT id, tenant_id, workspace_id, source_id, doc_id, chunk_index, text, embedding,
                    metadata_json, acl_allow_users, acl_allow_groups, acl_deny_users, acl_deny_groups
             FROM rag_chunks
             WHERE tenant_id = ? AND workspace_id = ?",
        );
        if let Some(source_ids) = &request.source_ids {
            if !source_ids.is_empty() {
                let placeholders = vec!["?"; source_ids.len()].join(", ");
                sql.push_str(" AND source_id IN (");
                sql.push_str(&placeholders);
                sql.push(')');
            }
        }

        let mut stmt = conn.prepare(&sql)?;
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = Vec::new();
        params_vec.push(&request.tenant_id as &dyn rusqlite::ToSql);
        params_vec.push(&request.workspace_id as &dyn rusqlite::ToSql);
        if let Some(source_ids) = &request.source_ids {
            for source_id in source_ids {
                params_vec.push(source_id as &dyn rusqlite::ToSql);
            }
        }

        let mut rows = stmt.query(params_vec.as_slice())?;
        let mut results = Vec::new();
        let allowed_users: HashSet<String> = request.allowed_users.iter().cloned().collect();
        let allowed_groups: HashSet<String> = request.allowed_groups.iter().cloned().collect();

        while let Some(row) = rows.next()? {
            let embedding_blob: Vec<u8> = row.get(7)?;
            let embedding = deserialize_embedding(&embedding_blob);
            if embedding.len() != request.query_embedding.len() {
                continue;
            }
            let acl_allow_users: String = row.get(9)?;
            let acl_allow_groups: String = row.get(10)?;
            let acl_deny_users: String = row.get(11)?;
            let acl_deny_groups: String = row.get(12)?;

            let allow_users: Vec<String> = parse_json_list(&acl_allow_users);
            let allow_groups: Vec<String> = parse_json_list(&acl_allow_groups);
            let deny_users: Vec<String> = parse_json_list(&acl_deny_users);
            let deny_groups: Vec<String> = parse_json_list(&acl_deny_groups);

            if !is_allowed(&allowed_users, &allowed_groups, &allow_users, &allow_groups) {
                continue;
            }
            if is_denied(&allowed_users, &allowed_groups, &deny_users, &deny_groups) {
                continue;
            }

            // TODO: replace brute-force scoring with an ANN index (HNSW) for scale.
            let score = cosine_similarity(&request.query_embedding, &embedding);
            if score < request.min_score {
                continue;
            }

            let metadata_json: String = row.get(8)?;
            let metadata: HashMap<String, String> =
                serde_json::from_str(&metadata_json).unwrap_or_else(|_| HashMap::new());

            let chunk = EmbeddedChunk {
                id: row.get(0)?,
                tenant_id: row.get(1)?,
                workspace_id: row.get(2)?,
                source_id: row.get(3)?,
                doc_id: row.get(4)?,
                chunk_index: row.get(5)?,
                text: row.get(6)?,
                embedding,
                metadata,
                acl: AccessControl {
                    allow_users,
                    allow_groups,
                    deny_users,
                    deny_groups,
                },
            };

            results.push(VectorSearchResult { chunk, score });
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(request.top_k);
        Ok(results)
    }
}

fn serialize_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn deserialize_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn parse_json_list(value: &str) -> Vec<String> {
    match serde_json::from_str::<Value>(value) {
        Ok(Value::Array(items)) => items
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

fn is_allowed(
    allowed_users: &HashSet<String>,
    allowed_groups: &HashSet<String>,
    acl_users: &[String],
    acl_groups: &[String],
) -> bool {
    if acl_users.is_empty() && acl_groups.is_empty() {
        return true;
    }
    acl_users.iter().any(|u| allowed_users.contains(u))
        || acl_groups.iter().any(|g| allowed_groups.contains(g))
}

fn is_denied(
    allowed_users: &HashSet<String>,
    allowed_groups: &HashSet<String>,
    deny_users: &[String],
    deny_groups: &[String],
) -> bool {
    deny_users.iter().any(|u| allowed_users.contains(u))
        || deny_groups.iter().any(|g| allowed_groups.contains(g))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
