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

/// Auto-prune oversized `ToolResult` blocks from older turns (preserving the system prompt
/// and the last 6 messages verbatim) ONLY when the request would otherwise exceed `max_model_len`.
/// Returns the number of pruned `ToolResult` blocks (0 if the request already fits).
pub fn auto_prune_old_tool_results(req: &mut CanonicalRequest, max_model_len: u32) -> usize {
    const KEEP_RECENT_MESSAGES: usize = 6;
    const MIN_PRUNE_BYTES: usize = 300;
    const PREVIEW_CHARS: usize = 160;

    let requested_output = u64::from(req.sampling.max_tokens.unwrap_or(1024));
    let limit = u64::from(max_model_len);
    if estimate_request_input_tokens(req) + requested_output <= limit {
        return 0;
    }

    let cutoff = req.messages.len().saturating_sub(KEEP_RECENT_MESSAGES);
    if cutoff == 0 {
        return 0;
    }

    let mut pruned = 0usize;
    for idx in 0..cutoff {
        for part in &mut req.messages[idx].parts {
            if let ContentPart::ToolResult { content, .. } = part {
                if content.len() > MIN_PRUNE_BYTES {
                    let orig_len = content.chars().count();
                    let preview: String = content.chars().take(PREVIEW_CHARS).collect();
                    *content = format!(
                        "{preview}\n...[pruned old tool_result: {orig_len} chars to fit context window]"
                    );
                    pruned += 1;
                }
            }
        }
        if estimate_request_input_tokens(req) + requested_output <= limit {
            break;
        }
    }
    pruned
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{CanonicalMessage, PriorityLane, ResponseFormat, Role, SamplingParams};

    #[test]
    fn auto_prunes_old_tool_results_while_keeping_recent_messages_intact() {
        let big_output = "line of terminal output\n".repeat(400); // ~9600 chars (~2400 tok)
        let mut messages = Vec::new();
        // 2 old tool results
        for i in 0..2 {
            messages.push(CanonicalMessage {
                role: Role::Tool,
                parts: vec![ContentPart::ToolResult {
                    tool_call_id: format!("call_{i}"),
                    content: big_output.clone(),
                    is_error: false,
                }],
            });
        }
        // 6 recent messages (including a recent tool result that MUST stay intact)
        for i in 0..5 {
            messages.push(CanonicalMessage {
                role: Role::User,
                parts: vec![ContentPart::Text {
                    text: format!("recent turn {i}"),
                }],
            });
        }
        messages.push(CanonicalMessage {
            role: Role::Tool,
            parts: vec![ContentPart::ToolResult {
                tool_call_id: "call_recent".to_string(),
                content: "recent output ".repeat(40),
                is_error: false,
            }],
        });

        let mut req = CanonicalRequest {
            request_id: "req_1".to_string(),
            model: "test".to_string(),
            system: Vec::new(),
            messages,
            tools: Vec::new(),
            tool_choice: None,
            response_format: ResponseFormat::Text,
            sampling: SamplingParams {
                temperature: None,
                top_p: None,
                max_tokens: Some(512),
                stop: Vec::new(),
                non_thinking: false,
            },
            stream: false,
            priority: PriorityLane::Interactive,
            prefix_hash: String::new(),
            assigned_slot: None,
        };

        // Limit 2048: fails before prune, succeeds after pruning old tool_results
        assert!(check_context_window(&req, 2048).is_err());
        let pruned = auto_prune_old_tool_results(&mut req, 2048);
        assert_eq!(pruned, 2);
        assert!(check_context_window(&req, 2048).is_ok());

        // Recent tool result (index 7, inside last 6 messages) must remain untouched
        if let ContentPart::ToolResult { content, .. } = &req.messages[7].parts[0] {
            assert!(!content.contains("pruned old tool_result"));
        } else {
            panic!("expected recent tool result");
        }
    }
}
