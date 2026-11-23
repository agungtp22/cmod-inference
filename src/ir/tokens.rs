use crate::ir::{CanonicalRequest, ContentPart};

/// Deterministic fast token estimator (~4 chars/token for Latin/Indonesian + image overhead).
pub fn estimate_text_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    if chars == 0 {
        0
    } else {
        (chars + 3) / 4
    }
}

pub fn estimate_request_input_tokens(req: &CanonicalRequest) -> u64 {
    let mut total: u64 = 16; // framing overhead

    for s in &req.system {
        total += estimate_text_tokens(&s.text) + 4;
    }

    for tool in &req.tools {
        total += estimate_text_tokens(&tool.name) + 8;
        if let Some(desc) = &tool.description {
            total += estimate_text_tokens(desc);
        }
        if let Ok(schema) = serde_json::to_string(&tool.input_schema) {
            total += estimate_text_tokens(&schema);
        }
    }

    for msg in &req.messages {
        total += 4;
        for part in &msg.parts {
            match part {
                ContentPart::Text { text } => total += estimate_text_tokens(text),
                // Conservative VLM tile token budget per image
                ContentPart::Image { .. } => total += 1024,
                ContentPart::ToolUse { name, arguments, .. } => {
                    total += estimate_text_tokens(name) + 8;
                    if let Ok(s) = serde_json::to_string(arguments) {
                        total += estimate_text_tokens(&s);
                    }
                }
                ContentPart::ToolResult { content, .. } => {
                    total += estimate_text_tokens(content) + 6;
                }
            }
        }
    }

    total
}

/// Guard against silent context truncation on long documents.
pub fn check_context_window(req: &CanonicalRequest, max_model_len: u32) -> Result<u64, String> {
    let input_tokens = estimate_request_input_tokens(req);
    let requested_output = u64::from(req.sampling.max_tokens.unwrap_or(1024));
    let limit = u64::from(max_model_len);

    if input_tokens >= limit {
        return Err(format!(
            "context_length_exceeded: estimasi input {input_tokens} token melampaui batas server max_model_len={limit}. \
             Aktifkan retrieval/chunking atau naikkan max_model_len."
        ));
    }
    if input_tokens + requested_output > limit {
        return Err(format!(
            "context_length_exceeded: estimasi input ({input_tokens}) + max_tokens ({requested_output}) = {} melebihi max_model_len={limit}.",
            input_tokens + requested_output
        ));
    }
    Ok(input_tokens)
}
