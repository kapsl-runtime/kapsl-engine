use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

pub type RagFilters = HashMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQuery {
    pub query: String,
    pub top_k: usize,
    pub filters: Option<RagFilters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagChunk {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Error, Debug)]
pub enum RagError {
    #[error("RAG backend error: {0}")]
    Backend(String),
    #[error("RAG invalid input: {0}")]
    InvalidInput(String),
}

#[async_trait]
pub trait VectorDbClient: Send + Sync {
    async fn query(&self, request: &RagQuery) -> Result<Vec<RagChunk>, RagError>;
}

#[derive(Debug, Clone)]
pub enum CitationStyle {
    BracketedNumber,
    Inline,
}

#[derive(Debug, Clone)]
pub struct RagPromptConfig {
    pub max_context_tokens: usize,
    pub max_chunks: usize,
    pub max_per_source: usize,
    pub min_score: f32,
    pub dedupe: bool,
    pub truncate: bool,
    pub citation_style: CitationStyle,
    pub fallback_message: String,
}

impl Default for RagPromptConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 1024,
            max_chunks: 8,
            max_per_source: 2,
            min_score: 0.0,
            dedupe: true,
            truncate: true,
            citation_style: CitationStyle::BracketedNumber,
            fallback_message: "No relevant documents found.".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Citation {
    pub index: usize,
    pub chunk_id: String,
    pub source: Option<String>,
    pub title: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RagPrompt {
    pub context: String,
    pub citations: Vec<Citation>,
    pub used_chunks: Vec<RagChunk>,
    pub total_context_tokens: usize,
    pub truncated: bool,
    pub fallback_message: Option<String>,
}

pub trait TokenCounter {
    fn count_tokens(&self, text: &str) -> usize;
}

pub struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

pub fn build_rag_prompt(
    chunks: &[RagChunk],
    config: &RagPromptConfig,
    counter: &dyn TokenCounter,
) -> RagPrompt {
    let selected = select_chunks(chunks, config);
    if selected.is_empty() {
        return RagPrompt {
            context: String::new(),
            citations: Vec::new(),
            used_chunks: Vec::new(),
            total_context_tokens: 0,
            truncated: false,
            fallback_message: Some(config.fallback_message.clone()),
        };
    }

    let mut context_blocks = Vec::new();
    let mut citations = Vec::new();
    let mut used_chunks = Vec::new();
    let mut used_tokens = 0usize;
    let mut truncated = false;

    for chunk in selected.iter() {
        if context_blocks.len() >= config.max_chunks {
            break;
        }

        let mut text = chunk.text.trim().to_string();
        if text.is_empty() {
            continue;
        }

        let remaining = config.max_context_tokens.saturating_sub(used_tokens);
        if remaining == 0 {
            break;
        }

        let chunk_tokens = counter.count_tokens(&text);
        if chunk_tokens > remaining {
            if !config.truncate {
                continue;
            }
            text = truncate_to_tokens(&text, remaining);
            if text.is_empty() {
                break;
            }
            truncated = true;
        }

        let citation_index = citations.len() + 1;
        let formatted =
            format_chunk_with_citation(&text, citation_index, chunk, &config.citation_style);
        context_blocks.push(formatted);
        used_tokens += counter.count_tokens(&text);
        used_chunks.push(chunk.clone());
        citations.push(build_citation(citation_index, chunk));
    }

    RagPrompt {
        context: context_blocks.join("\n\n"),
        citations,
        used_chunks,
        total_context_tokens: used_tokens,
        truncated,
        fallback_message: None,
    }
}

fn select_chunks(chunks: &[RagChunk], config: &RagPromptConfig) -> Vec<RagChunk> {
    let mut scored: Vec<&RagChunk> = chunks
        .iter()
        .filter(|chunk| chunk.score >= config.min_score)
        .collect();
    scored.sort_by(|a, b| {
        let score_cmp = b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal);
        if score_cmp == Ordering::Equal {
            a.id.cmp(&b.id)
        } else {
            score_cmp
        }
    });

    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    let mut per_source: HashMap<String, usize> = HashMap::new();

    for chunk in scored {
        if selected.len() >= config.max_chunks {
            break;
        }

        if config.dedupe {
            let key = normalize_text(&chunk.text);
            if !seen.insert(key) {
                continue;
            }
        }

        if config.max_per_source > 0 {
            let source = extract_source(&chunk.metadata).unwrap_or_else(|| "unknown".to_string());
            let count = per_source.entry(source).or_insert(0);
            if *count >= config.max_per_source {
                continue;
            }
            *count += 1;
        }

        selected.push(chunk.clone());
    }

    selected
}

fn format_chunk_with_citation(
    text: &str,
    index: usize,
    chunk: &RagChunk,
    style: &CitationStyle,
) -> String {
    let source_line = format_source_line(chunk);
    match style {
        CitationStyle::BracketedNumber => {
            if let Some(source_line) = source_line {
                format!("[{}] {}\n{}", index, text, source_line)
            } else {
                format!("[{}] {}", index, text)
            }
        }
        CitationStyle::Inline => {
            if let Some(source_line) = source_line {
                format!("{} ({})\n{}", text, index, source_line)
            } else {
                format!("{} ({})", text, index)
            }
        }
    }
}

fn format_source_line(chunk: &RagChunk) -> Option<String> {
    let title = chunk.metadata.get("title").cloned();
    let url = chunk.metadata.get("url").cloned();
    let source = extract_source(&chunk.metadata);

    let mut parts = Vec::new();
    if let Some(title) = title {
        parts.push(title);
    }
    if let Some(source) = source {
        if !parts.contains(&source) {
            parts.push(source);
        }
    }
    if let Some(url) = url {
        parts.push(url);
    }

    if parts.is_empty() {
        None
    } else {
        Some(format!("Source: {}", parts.join(" | ")))
    }
}

fn build_citation(index: usize, chunk: &RagChunk) -> Citation {
    Citation {
        index,
        chunk_id: chunk.id.clone(),
        source: extract_source(&chunk.metadata),
        title: chunk.metadata.get("title").cloned(),
        url: chunk.metadata.get("url").cloned(),
    }
}

fn extract_source(metadata: &HashMap<String, String>) -> Option<String> {
    for key in ["source", "doc_id", "document_id", "file", "url"] {
        if let Some(value) = metadata.get(key) {
            return Some(value.clone());
        }
    }
    None
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .map(|t| t.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    text.split_whitespace()
        .take(max_tokens)
        .collect::<Vec<_>>()
        .join(" ")
}

#[path = "rag_tests.rs"]
mod rag_tests;
