use axum::http::HeaderMap;

use crate::ir::PriorityLane;

/// Constant-time byte comparison to prevent timing side-channels on API keys.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Extract API key from either OpenAI (`Authorization: Bearer ...`) or Anthropic (`x-api-key: ...`).
pub fn extract_api_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(val) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }
    if let Some(val) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        let trimmed = val.trim();
        if let Some(token) = trimmed
            .strip_prefix("Bearer ")
            .or_else(|| trimmed.strip_prefix("bearer "))
        {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token);
            }
        }
    }
    None
}

/// Validate request API key against configured keys. Empty configured key list means open localhost mode.
pub fn authorize(headers: &HeaderMap, configured_keys: &[String]) -> bool {
    if configured_keys.is_empty() {
        return true;
    }
    let Some(provided) = extract_api_key(headers) else {
        return false;
    };
    configured_keys
        .iter()
        .any(|expected| ct_eq(provided.as_bytes(), expected.as_bytes()))
}

/// Read priority lane hint from `X-Cmod-Priority` header (defaults to `Interactive`).
pub fn extract_priority(headers: &HeaderMap) -> PriorityLane {
    match headers
        .get("x-cmod-priority")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("batch") | Some("background") | Some("low") => PriorityLane::Batch,
        _ => PriorityLane::Interactive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn accepts_both_bearer_and_x_api_key() {
        let keys = vec!["secret-123".to_string()];

        let mut h_oai = HeaderMap::new();
        h_oai.insert("authorization", HeaderValue::from_static("Bearer secret-123"));
        assert!(authorize(&h_oai, &keys));

        let mut h_ant = HeaderMap::new();
        h_ant.insert("x-api-key", HeaderValue::from_static("secret-123"));
        assert!(authorize(&h_ant, &keys));

        let mut h_bad = HeaderMap::new();
        h_bad.insert("x-api-key", HeaderValue::from_static("wrong-key"));
        assert!(!authorize(&h_bad, &keys));
    }
}
