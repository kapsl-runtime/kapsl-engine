//! Credential parsing, generation, and comparison helpers.

use super::*;

pub(crate) fn format_authorization_header(token: Option<&str>) -> Option<String> {
    let raw = token?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Some((scheme, _)) = raw.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            return Some(raw.to_string());
        }
    }
    Some(format!("Bearer {}", raw))
}

pub(crate) fn parse_authorization_token(header_value: Option<&str>) -> Option<&str> {
    let raw_header = header_value?;
    let trimmed = raw_header.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((scheme, token)) = trimmed.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            let parsed = token.trim();
            if parsed.is_empty() {
                return None;
            }
            return Some(parsed);
        }
    }
    Some(trimmed)
}

pub(crate) fn generate_random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 8];
    OsRng.fill_bytes(&mut bytes);
    let mut suffix = String::with_capacity(16);
    for byte in bytes {
        suffix.push_str(&format!("{:02x}", byte));
    }
    format!("{}_{}", prefix, suffix)
}

pub(crate) fn generate_api_key() -> String {
    let mut bytes = [0u8; 24];
    OsRng.fill_bytes(&mut bytes);
    let secret = BASE64_URL_SAFE_NO_PAD.encode(bytes);
    format!("kpsl_{}", secret)
}

pub(crate) fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{:02x}", byte));
    }
    output
}

pub(crate) fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0u8;
    for (lhs, rhs) in left.as_bytes().iter().zip(right.as_bytes()) {
        diff |= lhs ^ rhs;
    }
    diff == 0
}

pub(crate) fn authorization_matches_token(
    header_value: Option<&str>,
    expected_token: &str,
) -> bool {
    let Some(raw_header) = header_value else {
        return false;
    };
    let trimmed = raw_header.trim();
    if trimmed.is_empty() {
        return false;
    }
    if let Some((scheme, token)) = trimmed.split_once(' ') {
        if scheme.eq_ignore_ascii_case("bearer") {
            return constant_time_eq(token.trim(), expected_token);
        }
    }
    constant_time_eq(trimmed, expected_token)
}
