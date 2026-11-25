pub mod tokens;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PriorityLane {
    Interactive,
    Batch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    pub text: String,
    pub cacheable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        mime_type: String,
        base64: String,
    },
    ToolUse {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalMessage {
    pub role: Role,
    pub parts: Vec<ContentPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub non_thinking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalRequest {
    pub request_id: String,
    pub model: String,
    pub system: Vec<SystemBlock>,
    pub messages: Vec<CanonicalMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
    pub response_format: ResponseFormat,
    pub sampling: SamplingParams,
    pub stream: bool,
    pub priority: PriorityLane,
    pub prefix_hash: String,
    pub tenant_id: String,
    pub assigned_slot: Option<u32>,
}

impl CanonicalRequest {
    /// Compute deterministic hash of system blocks + tool schemas for KV/prefix cache affinity.
    pub fn compute_prefix_hash(system: &[SystemBlock], tools: &[ToolDefinition]) -> String {
        let mut hasher = Sha256::new();
        for block in system {
            hasher.update(block.text.as_bytes());
            hasher.update(b"\x00");
        }
        for tool in tools {
            hasher.update(tool.name.as_bytes());
            hasher.update(b":");
            if let Ok(s) = serde_json::to_string(&tool.input_schema) {
                hasher.update(s.as_bytes());
            }
            hasher.update(b"\x00");
        }
        let digest = hasher.finalize();
        format!("{:x}", digest)[..16].to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence,
}

impl StopReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::EndTurn | Self::StopSequence => "stop",
            Self::MaxTokens => "length",
            Self::ToolUse => "tool_calls",
        }
    }

    pub fn as_anthropic_str(self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::MaxTokens => "max_tokens",
            Self::ToolUse => "tool_use",
            Self::StopSequence => "stop_sequence",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalResponse {
    pub id: String,
    pub model: String,
    pub text: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<CanonicalToolCall>,
    pub stop_reason: StopReason,
    pub usage: UsageStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalStreamEvent {
    Start {
        id: String,
        model: String,
    },
    ThinkingDelta(String),
    TextDelta(String),
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallDelta {
        index: usize,
        arguments_delta: String,
    },
    Usage(UsageStats),
    Done {
        stop_reason: StopReason,
    },
}

/// Parse a `data:<mime>;base64,<payload>` URL, rejecting remote http/https URLs (SSRF guard).
pub fn parse_data_image_url(url: &str) -> Result<(String, String), String> {
    let trimmed = url.trim();
    if !trimmed.starts_with("data:") {
        return Err(
            "URL gambar eksternal ditolak (proteksi SSRF & privasi LAN). Kirim gambar dalam format data:<mime>;base64,...".to_string(),
        );
    }
    let rest = &trimmed[5..];
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err("Format data URI gambar tidak valid".to_string());
    };
    let Some(mime) = meta.strip_suffix(";base64") else {
        return Err("Data URI gambar harus ber-encoding base64".to_string());
    };
    if !mime.starts_with("image/") {
        return Err(format!("MIME type tidak didukung untuk gambar: {mime}"));
    }
    if payload.trim().is_empty() {
        return Err("Payload base64 gambar kosong".to_string());
    }
    Ok((mime.to_string(), payload.trim().to_string()))
}
