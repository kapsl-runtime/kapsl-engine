const EMBEDDING_DIMENSION: usize = 256;

fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn embed_text_with_dimension(text: &str, dimension: usize) -> Vec<f32> {
    if dimension == 0 {
        return Vec::new();
    }

    let mut embedding = vec![0.0f32; dimension];
    let mut token_count = 0usize;
    for token in text
        .split_whitespace()
        .map(|token| token.trim_matches(|character: char| !character.is_alphanumeric()))
        .filter(|token| !token.is_empty())
    {
        let normalized = token.to_ascii_lowercase();
        let hash = fnv1a_64(normalized.as_bytes());
        let index = (hash % dimension as u64) as usize;
        let sign = if (hash & 1) == 0 { 1.0 } else { -1.0 };
        embedding[index] += sign;
        token_count += 1;
    }

    if token_count == 0 {
        return embedding;
    }

    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

pub(super) fn embed_text(text: &str) -> Vec<f32> {
    embed_text_with_dimension(text, EMBEDDING_DIMENSION)
}

#[cfg(test)]
mod tests {
    use super::{embed_text, embed_text_with_dimension};

    #[test]
    fn zero_dimension_returns_an_empty_embedding() {
        assert!(embed_text_with_dimension("hello", 0).is_empty());
    }

    #[test]
    fn embedding_is_normalized_and_case_insensitive() {
        let embedding = embed_text("Hello RAG");
        let equivalent = embed_text("hello rag");
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();

        assert_eq!(embedding, equivalent);
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
