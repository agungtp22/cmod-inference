use std::collections::HashMap;
use std::convert::Infallible;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::auth::{authorize, extract_priority, resolve_tenant_id};
use crate::ir::tokens::{auto_prune_old_tool_results, check_context_window, estimate_request_input_tokens};
use crate::ir::{
    CanonicalMessage, CanonicalRequest, CanonicalStreamEvent, ContentPart, ResponseFormat, Role,
    SamplingParams, SystemBlock, ToolChoice, ToolDefinition, UsageStats,
};
use crate::presets::find_preset;
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub system: Option<Value>,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub thinking: Option<Value>,
    #[serde(default)]
    pub verbose: bool,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicCountTokensRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub system: Option<Value>,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
}

fn anthropic_error(status: StatusCode, err_type: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "type": "error",
            "error": {
                "type": err_type,
                "message": message.into(),
            }
        })),
    )
        .into_response()
}

fn parse_anthropic_system(val: Option<Value>) -> Vec<SystemBlock> {
    match val {
        Some(Value::String(s)) if !s.trim().is_empty() => vec![SystemBlock {
            text: s,
            cacheable: false,
        }],
        Some(Value::Array(blocks)) => blocks
            .into_iter()
            .filter_map(|b| {
                let text = b.get("text").and_then(Value::as_str)?.to_string();
                let cacheable = b.get("cache_control").is_some();
                Some(SystemBlock { text, cacheable })
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_anthropic_messages(raw_msgs: Vec<Value>) -> Result<Vec<CanonicalMessage>, String> {
    let mut out = Vec::new();
    for raw in raw_msgs {
        let role_str = raw.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = raw.get("content");

        match content {
            Some(Value::String(s)) => {
                let role = if role_str == "assistant" {
                    Role::Assistant
                } else {
                    Role::User
                };
                out.push(CanonicalMessage {
                    role,
                    parts: vec![ContentPart::Text { text: s.clone() }],
                });
            }
            Some(Value::Array(blocks)) => {
                let mut regular_parts = Vec::new();
                let mut tool_results = Vec::new();

                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text = block
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            regular_parts.push(ContentPart::Text { text });
                        }
                        Some("image") => {
                            let source = block
                                .get("source")
                                .ok_or_else(|| "Anthropic image block wajib memiliki source".to_string())?;
                            let src_type = source
                                .get("type")
                                .and_then(Value::as_str)
                                .unwrap_or_default();
                            if src_type != "base64" {
                                return Err(
                                    "Hanya source.type='base64' yang diizinkan untuk gambar (proteksi SSRF & privasi LAN)"
                                        .to_string(),
                                );
                            }
                            let mime_type = source
                                .get("media_type")
                                .and_then(Value::as_str)
                                .unwrap_or("image/png")
                                .to_string();
                            let base64 = source
                                .get("data")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .trim()
                                .to_string();
                            if base64.is_empty() {
                                return Err("Payload base64 gambar Anthropic kosong".to_string());
                            }
                            regular_parts.push(ContentPart::Image { mime_type, base64 });
                        }
                        Some("tool_use") => {
                            let id = block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("toolu_1")
                                .to_string();
                            let name = block
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            let arguments = block
                                .get("input")
                                .cloned()
                                .unwrap_or_else(|| json!({}));
                            regular_parts.push(ContentPart::ToolUse {
                                id,
                                name,
                                arguments,
                            });
                        }
                        Some("tool_result") => {
                            let tool_call_id = block
                                .get("tool_use_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            let is_error = block
                                .get("is_error")
                                .and_then(Value::as_bool)
                                .unwrap_or(false);
                            let result_text = match block.get("content") {
                                Some(Value::String(s)) => s.clone(),
                                Some(Value::Array(arr)) => arr
                                    .iter()
                                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                                _ => String::new(),
                            };
                            tool_results.push(ContentPart::ToolResult {
                                tool_call_id,
                                content: result_text,
                                is_error,
                            });
                        }
                        _ => {}
                    }
                }

                if !tool_results.is_empty() {
                    out.push(CanonicalMessage {
                        role: Role::Tool,
                        parts: tool_results,
                    });
                }
                if !regular_parts.is_empty() {
                    let role = if role_str == "assistant" {
                        Role::Assistant
                    } else {
                        Role::User
                    };
                    out.push(CanonicalMessage {
                        role,
                        parts: regular_parts,
                    });
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn parse_anthropic_tools(raw_tools: Option<Vec<Value>>) -> Result<Vec<ToolDefinition>, String> {
    let mut tools = Vec::new();
    if let Some(raw_tools) = raw_tools {
        for t in raw_tools {
            let name = t
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "Anthropic tool wajib memiliki name".to_string())?
                .to_string();
            let description = t
                .get("description")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let input_schema = t
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            tools.push(ToolDefinition {
                name,
                description,
                input_schema,
            });
        }
    }
    Ok(tools)
}

fn parse_anthropic_to_canonical(
    req: AnthropicMessagesRequest,
    default_model: &str,
    priority: crate::ir::PriorityLane,
) -> Result<CanonicalRequest, String> {
    let system = parse_anthropic_system(req.system);
    let messages = parse_anthropic_messages(req.messages)?;
    let tools = parse_anthropic_tools(req.tools)?;

    let tool_choice = req.tool_choice.and_then(|tc| {
        match tc.get("type").and_then(Value::as_str) {
            Some("auto") => Some(ToolChoice::Auto),
            Some("any") => Some(ToolChoice::Required),
            Some("none") => Some(ToolChoice::None),
            Some("tool") => tc
                .get("name")
                .and_then(Value::as_str)
                .map(|n| ToolChoice::Named {
                    name: n.to_string(),
                }),
            _ => None,
        }
    });

    let requested_model = if req.model.trim().is_empty() {
        default_model.to_string()
    } else {
        req.model.trim().to_string()
    };

    let preset = find_preset(&requested_model);
    let temperature = req
        .temperature
        .or_else(|| preset.as_ref().map(|p| p.default_temperature));
    let thinking_mode = req
        .thinking
        .as_ref()
        .and_then(|t| t.get("type").and_then(Value::as_str));
    let non_thinking = match thinking_mode {
        Some("disabled") => true,
        Some("enabled") => false,
        _ => preset.as_ref().map(|p| p.non_thinking).unwrap_or(false),
    };

    let prefix_hash = CanonicalRequest::compute_prefix_hash(&system, &tools);

    Ok(CanonicalRequest {
        request_id: format!("msg_{}", Uuid::new_v4().simple()),
        model: requested_model,
        system,
        messages,
        tools,
        tool_choice,
        response_format: ResponseFormat::Text,
        sampling: SamplingParams {
            temperature,
            top_p: req.top_p,
            max_tokens: req.max_tokens.or(Some(4096)),
            stop: req.stop_sequences.unwrap_or_default(),
            non_thinking,
        },
        stream: req.stream,
        priority,
        prefix_hash,
        tenant_id: "default".to_string(),
        assigned_slot: None,
    })
}

pub async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<AnthropicCountTokensRequest>,
) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "x-api-key tidak valid atau tidak diberikan",
        );
    }

    let system = parse_anthropic_system(body.system);
    let messages = match parse_anthropic_messages(body.messages) {
        Ok(m) => m,
        Err(msg) => return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", msg),
    };
    let tools = match parse_anthropic_tools(body.tools) {
        Ok(t) => t,
        Err(msg) => return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", msg),
    };

    let req = CanonicalRequest {
        request_id: "count".to_string(),
        model: body.model,
        system,
        messages,
        tools,
        tool_choice: None,
        response_format: ResponseFormat::Text,
        sampling: SamplingParams {
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: Vec::new(),
            non_thinking: false,
        },
        stream: false,
        priority: crate::ir::PriorityLane::Interactive,
        prefix_hash: String::new(),
        tenant_id: "default".to_string(),
        assigned_slot: None,
    };

    let input_tokens = estimate_request_input_tokens(&req);
    Json(json!({ "input_tokens": input_tokens })).into_response()
}

pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<AnthropicMessagesRequest>,
) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "x-api-key tidak valid atau tidak diberikan",
        );
    }

    let verbose = body.verbose
        || headers
            .get("x-cmod-verbose")
            .and_then(|v| v.to_str().ok())
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);
    let priority = extract_priority(&headers);
    let mut canonical =
        match parse_anthropic_to_canonical(body, state.backend.default_model(), priority) {
            Ok(req) => req,
            Err(msg) => {
                return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", msg);
            }
        };
    canonical.tenant_id = resolve_tenant_id(&headers, &state.config.server.api_keys);

    let pruned = auto_prune_old_tool_results(&mut canonical, state.config.backend.max_model_len);
    if pruned > 0 {
        tracing::info!(
            pruned_tool_results = pruned,
            max_model_len = state.config.backend.max_model_len,
            "Auto-pruned old tool_result blocks to fit context window"
        );
    }

    let estimated_input = match check_context_window(&mut canonical, state.config.backend.max_model_len)
    {
        Ok(n) => n,
        Err(msg) => {
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", msg);
        }
    };

    canonical.assigned_slot = Some(state.scheduler.assign_slot(&canonical.prefix_hash));
    let permit = match state.scheduler.acquire(canonical.priority).await {
        Ok(p) => p,
        Err(msg) => {
            return anthropic_error(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", msg);
        }
    };

    if canonical.stream {
        let fallback_id = canonical.request_id.clone();
        let fallback_model = canonical.model.clone();
        let stream = match state.backend.chat_stream(canonical).await {
            Ok(s) => s,
            Err(err) => {
                return anthropic_error(StatusCode::BAD_GATEWAY, "api_error", err.to_string());
            }
        };

        let sse_stream = async_stream::stream! {
            let _permit = permit;
            let mut stream = stream;
            let mut started = false;
            let mut next_block_idx: usize = 0;
            let mut thinking_block_idx: Option<usize> = None;
            let mut text_block_idx: Option<usize> = None;
            let mut tool_block_indices: HashMap<usize, usize> = HashMap::new();
            let mut open_blocks: Vec<usize> = Vec::new();
            let mut final_usage = UsageStats {
                input_tokens: estimated_input,
                output_tokens: 0,
                cached_input_tokens: 0,
            };

            while let Some(item) = stream.next().await {
                match item {
                    Ok(CanonicalStreamEvent::Start { id, model }) => {
                        if !started {
                            started = true;
                            let payload = json!({
                                "type": "message_start",
                                "message": {
                                    "id": if id.starts_with("msg_") { id } else { fallback_id.clone() },
                                    "type": "message",
                                    "role": "assistant",
                                    "model": model,
                                    "content": [],
                                    "stop_reason": Value::Null,
                                    "stop_sequence": Value::Null,
                                    "usage": {
                                        "input_tokens": estimated_input,
                                        "output_tokens": 0,
                                        "cache_read_input_tokens": 0
                                    }
                                }
                            });
                            yield Ok::<Event, Infallible>(Event::default().event("message_start").data(payload.to_string()));
                        }
                    }
                    Ok(CanonicalStreamEvent::ThinkingDelta(delta)) => {
                        if !started {
                            started = true;
                            let payload = json!({
                                "type": "message_start",
                                "message": {
                                    "id": fallback_id.clone(),
                                    "type": "message",
                                    "role": "assistant",
                                    "model": fallback_model.clone(),
                                    "content": [],
                                    "stop_reason": Value::Null,
                                    "stop_sequence": Value::Null,
                                    "usage": { "input_tokens": estimated_input, "output_tokens": 0 }
                                }
                            });
                            yield Ok::<Event, Infallible>(Event::default().event("message_start").data(payload.to_string()));
                        }
                        let idx = match thinking_block_idx {
                            Some(i) => i,
                            None => {
                                let i = next_block_idx;
                                next_block_idx += 1;
                                thinking_block_idx = Some(i);
                                open_blocks.push(i);
                                let start_ev = json!({
                                    "type": "content_block_start",
                                    "index": i,
                                    "content_block": { "type": "thinking", "thinking": "" }
                                });
                                yield Ok::<Event, Infallible>(Event::default().event("content_block_start").data(start_ev.to_string()));
                                i
                            }
                        };
                        let delta_ev = json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": { "type": "thinking_delta", "thinking": delta }
                        });
                        yield Ok::<Event, Infallible>(Event::default().event("content_block_delta").data(delta_ev.to_string()));
                    }
                    Ok(CanonicalStreamEvent::TextDelta(delta)) => {
                        if !started {
                            started = true;
                            let payload = json!({
                                "type": "message_start",
                                "message": {
                                    "id": fallback_id.clone(),
                                    "type": "message",
                                    "role": "assistant",
                                    "model": fallback_model.clone(),
                                    "content": [],
                                    "stop_reason": Value::Null,
                                    "stop_sequence": Value::Null,
                                    "usage": { "input_tokens": estimated_input, "output_tokens": 0 }
                                }
                            });
                            yield Ok::<Event, Infallible>(Event::default().event("message_start").data(payload.to_string()));
                        }
                        let idx = match text_block_idx {
                            Some(i) => i,
                            None => {
                                let i = next_block_idx;
                                next_block_idx += 1;
                                text_block_idx = Some(i);
                                open_blocks.push(i);
                                let start_ev = json!({
                                    "type": "content_block_start",
                                    "index": i,
                                    "content_block": { "type": "text", "text": "" }
                                });
                                yield Ok::<Event, Infallible>(Event::default().event("content_block_start").data(start_ev.to_string()));
                                i
                            }
                        };
                        let delta_ev = json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": { "type": "text_delta", "text": delta }
                        });
                        yield Ok::<Event, Infallible>(Event::default().event("content_block_delta").data(delta_ev.to_string()));
                    }
                    Ok(CanonicalStreamEvent::ToolCallStart { index, id, name }) => {
                        let block_idx = next_block_idx;
                        next_block_idx += 1;
                        tool_block_indices.insert(index, block_idx);
                        open_blocks.push(block_idx);
                        let start_ev = json!({
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": {}
                            }
                        });
                        yield Ok::<Event, Infallible>(Event::default().event("content_block_start").data(start_ev.to_string()));
                    }
                    Ok(CanonicalStreamEvent::ToolCallDelta { index, arguments_delta }) => {
                        if let Some(&block_idx) = tool_block_indices.get(&index) {
                            let delta_ev = json!({
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments_delta
                                }
                            });
                            yield Ok::<Event, Infallible>(Event::default().event("content_block_delta").data(delta_ev.to_string()));
                        }
                    }
                    Ok(CanonicalStreamEvent::Usage(u)) => {
                        final_usage = u;
                    }
                    Ok(CanonicalStreamEvent::Done { stop_reason }) => {
                        for idx in open_blocks.drain(..) {
                            let stop_ev = json!({
                                "type": "content_block_stop",
                                "index": idx
                            });
                            yield Ok::<Event, Infallible>(Event::default().event("content_block_stop").data(stop_ev.to_string()));
                        }
                        let msg_delta = json!({
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason.as_anthropic_str(),
                                "stop_sequence": Value::Null
                            },
                            "usage": {
                                "output_tokens": final_usage.output_tokens,
                                "cache_read_input_tokens": final_usage.cached_input_tokens
                            }
                        });
                        yield Ok::<Event, Infallible>(Event::default().event("message_delta").data(msg_delta.to_string()));
                        yield Ok::<Event, Infallible>(Event::default().event("message_stop").data(json!({ "type": "message_stop" }).to_string()));
                    }
                    Err(err) => {
                        let ev = json!({
                            "type": "error",
                            "error": { "type": "api_error", "message": err.to_string() }
                        });
                        yield Ok::<Event, Infallible>(Event::default().event("error").data(ev.to_string()));
                        break;
                    }
                }
            }
        };

        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let _permit = permit;
        let started_at = std::time::Instant::now();
        let thinking_enabled = !canonical.sampling.non_thinking;
        let assigned_slot = canonical.assigned_slot;
        let prefix_hash = canonical.prefix_hash.clone();
        match state.backend.chat_complete(canonical).await {
            Ok(resp) => {
                let elapsed_ms = started_at.elapsed().as_millis() as u64;
                let mut content = Vec::new();
                if let Some(thinking) = resp.thinking {
                    if !thinking.is_empty() {
                        content.push(json!({ "type": "thinking", "thinking": thinking }));
                    }
                }
                if !resp.text.is_empty() || resp.tool_calls.is_empty() {
                    content.push(json!({ "type": "text", "text": resp.text }));
                }
                for tc in resp.tool_calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }

                let msg_id = if resp.id.starts_with("msg_") {
                    resp.id
                } else {
                    format!("msg_{}", Uuid::new_v4().simple())
                };

                let mut body = json!({
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": resp.model,
                    "content": content,
                    "stop_reason": resp.stop_reason.as_anthropic_str(),
                    "stop_sequence": Value::Null,
                    "usage": {
                        "input_tokens": resp.usage.input_tokens.max(estimated_input),
                        "output_tokens": resp.usage.output_tokens,
                        "cache_read_input_tokens": resp.usage.cached_input_tokens,
                    }
                });
                if verbose {
                    let tps = if elapsed_ms > 0 {
                        (resp.usage.output_tokens as f64) / (elapsed_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                    body["cmod_verbose"] = json!({
                        "backend": state.backend.kind_name(),
                        "elapsed_ms": elapsed_ms,
                        "tokens_per_sec": (tps * 10.0).round() / 10.0,
                        "thinking_enabled": thinking_enabled,
                        "assigned_slot": assigned_slot,
                        "prefix_hash": prefix_hash,
                        "cached_tokens": resp.usage.cached_input_tokens,
                    });
                }

                Json(body).into_response()
            }
            Err(err) => anthropic_error(StatusCode::BAD_GATEWAY, "api_error", err.to_string()),
        }
    }
}
