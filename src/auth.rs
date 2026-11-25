use axum::http::HeaderMap;
use sha2::{Digest, Sha256};

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

fn sanitize_tenant_segment(raw: &str) -> String {
    raw.trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(32)
        .collect()
}

/// Derive a deterministic, isolated tenant identifier for L1/L2/L3 KV-Cache partitioning.
/// - When `--api-key` is configured: hashes the authenticated API key (`k_<sha256[..12]>`) so
///   different API keys can never share or probe each other's KV-cache. Optional `X-Cmod-Tenant`
///   header further sub-partitions within that key (`k_<hash>_<subtenant>`).
/// - When `--api-key` is empty (open localhost): uses sanitized `X-Cmod-Tenant` if present,
///   otherwise `"default"`.
pub fn resolve_tenant_id(headers: &HeaderMap, configured_keys: &[String]) -> String {
    let sub_tenant = headers
        .get("x-cmod-tenant")
        .and_then(|v| v.to_str().ok())
        .map(sanitize_tenant_segment)
        .filter(|s| !s.is_empty());

    if !configured_keys.is_empty() {
        if let Some(key) = extract_api_key(headers) {
            let mut hasher = Sha256::new();
            hasher.update(key.as_bytes());
            let hex = format!("{:x}", hasher.finalize());
            let key_fp = &hex[..12];
            return match sub_tenant {
                Some(sub) => format!("k_{key_fp}_{sub}"),
                None => format!("k_{key_fp}"),
            };
        }
    }

    sub_tenant.unwrap_or_else(|| "default".to_string())
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

    #[test]
    fn isolates_kv_cache_tenant_id_by_api_key_and_sub_tenant() {
        let keys = vec!["key-alpha".to_string(), "key-beta".to_string()];

        let mut h_a = HeaderMap::new();
        h_a.insert("authorization", HeaderValue::from_static("Bearer key-alpha"));
        let mut h_b = HeaderMap::new();
        h_b.insert("authorization", HeaderValue::from_static("Bearer key-beta"));

        let t_a = resolve_tenant_id(&h_a, &keys);
        let t_b = resolve_tenant_id(&h_b, &keys);
        assert!(t_a.starts_with("k_"));
        assert_ne!(t_a, t_b);

        // Even with the same X-Cmod-Tenant header, different API keys remain isolated
        h_a.insert("x-cmod-tenant", HeaderValue::from_static("finance"));
        h_b.insert("x-cmod-tenant", HeaderValue::from_static("finance"));
        assert_ne!(resolve_tenant_id(&h_a, &keys), resolve_tenant_id(&h_b, &keys));

        // Open localhost mode defaults to "default"
        let h_local = HeaderMap::new();
        assert_eq!(resolve_tenant_id(&h_local, &[]), "default");
    }
}
