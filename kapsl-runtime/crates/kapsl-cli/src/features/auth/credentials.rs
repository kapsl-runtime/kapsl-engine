//! Credential parsing, generation, and comparison helpers.

use super::*;

pub(crate) fn format_authorization_header(token: Option<&str>) -> Option<String> {
    parse_authorization_token(token).map(|token| format!("Bearer {token}"))
}

pub(crate) fn parse_authorization_token(header_value: Option<&str>) -> Option<&str> {
    let trimmed = header_value?.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(separator) = trimmed.find(char::is_whitespace) {
        let (scheme, token) = trimmed.split_at(separator);
        if scheme.eq_ignore_ascii_case("bearer") {
            let parsed = token.trim();
            return (!parsed.is_empty()).then_some(parsed);
        }
    } else if trimmed.eq_ignore_ascii_case("bearer") {
        return None;
    }

    Some(trimmed)
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        encoded.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}

pub(crate) fn generate_random_id(prefix: &str) -> String {
    let mut bytes = [0u8; 8];
    OsRng.fill_bytes(&mut bytes);
    format!("{}_{}", prefix, encode_lower_hex(&bytes))
}

pub(crate) fn generate_api_key() -> String {
    let mut bytes = [0u8; 24];
    OsRng.fill_bytes(&mut bytes);
    let secret = BASE64_URL_SAFE_NO_PAD.encode(bytes);
    format!("kpsl_{}", secret)
}

pub(crate) fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    encode_lower_hex(&digest)
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
