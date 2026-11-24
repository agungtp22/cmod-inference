use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::auth::{authorize, extract_priority};
use crate::ir::tokens::check_context_window;
use crate::ir::{
    parse_data_image_url, CanonicalMessage, CanonicalRequest, CanonicalStreamEvent, ContentPart,
    ResponseFormat, Role, SamplingParams, SystemBlock, ToolChoice, ToolDefinition,
};
use crate::presets::{builtin_presets, find_preset};
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct OpenAiChatRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub think: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    #[serde(default)]
    pub verbose: bool,
}

fn openai_error(status: StatusCode, err_type: &str, code: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": err_type,
                "code": code,
            }
        })),
    )
        .into_response()
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn parse_openai_to_canonical(
    req: OpenAiChatRequest,
    default_model: &str,
    default_non_thinking: bool,
    priority: crate::ir::PriorityLane,
) -> Result<CanonicalRequest, String> {
    let mut system = Vec::new();
    let mut messages = Vec::new();

    for raw in req.messages {
        let role_str = raw
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");

        match role_str {
            "system" | "developer" => {
                let text = extract_text_content(raw.get("content"));
                if !text.is_empty() {
                    system.push(SystemBlock {
                        text,
                        cacheable: true,
                    });
                }
            }
            "user" => {
                let parts = parse_openai_content_parts(raw.get("content"))?;
                messages.push(CanonicalMessage {
                    role: Role::User,
                    parts,
                });
            }
            "assistant" => {
                let mut parts = Vec::new();
                let text = extract_text_content(raw.get("content"));
                if !text.is_empty() {
                    parts.push(ContentPart::Text { text });
                }
                if let Some(tcs) = raw.get("tool_calls").and_then(Value::as_array) {
                    for tc in tcs {
                        let id = tc
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("call_1")
                            .to_string();
                        let func = tc.get("function").unwrap_or(&Value::Null);
                        let name = func
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let args_str = func
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}");
                        let arguments = serde_json::from_str::<Value>(args_str)
                            .unwrap_or_else(|_| json!({ "raw": args_str }));
                        parts.push(ContentPart::ToolUse {
                            id,
                            name,
                            arguments,
                        });
                    }
                }
                messages.push(CanonicalMessage {
                    role: Role::Assistant,
                    parts,
                });
            }
            "tool" => {
                let tool_call_id = raw
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let content = extract_text_content(raw.get("content"));
                messages.push(CanonicalMessage {
                    role: Role::Tool,
                    parts: vec![ContentPart::ToolResult {
                        tool_call_id,
                        content,
                        is_error: false,
                    }],
                });
            }
            other => {
                return Err(format!("Role OpenAI tidak didukung: {other}"));
            }
        }
    }

    let mut tools = Vec::new();
    if let Some(raw_tools) = req.tools {
        for t in raw_tools {
            let func = t.get("function").unwrap_or(&t);
            let name = func
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "Tool OpenAI wajib memiliki function.name".to_string())?
                .to_string();
            let description = func
                .get("description")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let input_schema = func
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            tools.push(ToolDefinition {
                name,
                description,
                input_schema,
            });
        }
    }

    let tool_choice = req.tool_choice.and_then(|tc| match tc {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        Value::Object(map) => map
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named {
                name: n.to_string(),
            }),
        _ => None,
    });

    let response_format = match req.response_format {
        None => ResponseFormat::Text,
        Some(rf) => match rf.get("type").and_then(Value::as_str) {
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => {
                let js = rf.get("json_schema").unwrap_or(&Value::Null);
                let name = js
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("response")
                    .to_string();
                let schema = js
                    .get("schema")
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object" }));
                let strict = js.get("strict").and_then(Value::as_bool).unwrap_or(true);
                ResponseFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                }
            }
            _ => ResponseFormat::Text,
        },
    };

    let stop = match req.stop {
        Some(Value::String(s)) => vec![s],
        Some(Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    };

    let requested_model = if req.model.trim().is_empty() {
        default_model.to_string()
    } else {
        req.model.trim().to_string()
    };

    let preset = find_preset(&requested_model);
    let temperature = req
        .temperature
        .or_else(|| preset.as_ref().map(|p| p.default_temperature));
    let explicit_enable_thinking = req
        .enable_thinking
        .or(req.think)
        .or_else(|| {
            req.chat_template_kwargs
                .as_ref()
                .and_then(|kw| kw.get("enable_thinking").and_then(Value::as_bool))
        })
        .or_else(|| {
            req.reasoning_effort.as_deref().and_then(|eff| {
                match eff.trim().to_ascii_lowercase().as_str() {
                    "none" | "off" | "disabled" | "0" => Some(false),
                    "low" | "medium" | "high" | "minimal" => Some(true),
                    _ => None,
                }
            })
        });
    let non_thinking = match explicit_enable_thinking {
        Some(enabled) => !enabled,
        None => {
            default_non_thinking
                || preset.as_ref().map(|p| p.non_thinking).unwrap_or(false)
                || !matches!(response_format, ResponseFormat::Text)
        }
    };

    let prefix_hash = CanonicalRequest::compute_prefix_hash(&system, &tools);

    Ok(CanonicalRequest {
        request_id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
        model: requested_model,
        system,
        messages,
        tools,
        tool_choice,
        response_format,
        sampling: SamplingParams {
            temperature,
            top_p: req.top_p,
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            stop,
            non_thinking,
        },
        stream: req.stream,
        priority,
        prefix_hash,
        assigned_slot: None,
    })
}

fn extract_text_content(val: Option<&Value>) -> String {
    match val {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn parse_openai_content_parts(val: Option<&Value>) -> Result<Vec<ContentPart>, String> {
    match val {
        Some(Value::String(s)) => Ok(vec![ContentPart::Text { text: s.clone() }]),
        Some(Value::Array(arr)) => {
            let mut parts = Vec::new();
            for item in arr {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = item
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        parts.push(ContentPart::Text { text });
                    }
                    Some("image_url") => {
                        let url = item
                            .get("image_url")
                            .and_then(|iu| iu.get("url").and_then(Value::as_str).or_else(|| iu.as_str()))
                            .ok_or_else(|| "image_url wajib memiliki field url".to_string())?;
                        let (mime_type, base64) = parse_data_image_url(url)?;
                        parts.push(ContentPart::Image { mime_type, base64 });
                    }
                    _ => {}
                }
            }
            Ok(parts)
        }
        _ => Ok(Vec::new()),
    }
}

pub async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return openai_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "API key tidak valid atau tidak diberikan",
        );
    }

    let created = unix_ts();
    let mut data = vec![json!({
        "id": state.config.backend.model,
        "object": "model",
        "created": created,
        "owned_by": "cmod-inference",
        "context_length": state.config.backend.max_model_len,
    })];

    for p in builtin_presets() {
        if p.id != state.config.backend.model {
            data.push(json!({
                "id": p.id,
                "object": "model",
                "created": created,
                "owned_by": "cmod-inference-preset",
                "context_length": state.config.backend.max_model_len,
                "multimodal": p.multimodal,
            }));
        }
    }

    Json(json!({
        "object": "list",
        "data": data,
    }))
    .into_response()
}

pub async fn get_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return openai_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "API key tidak valid atau tidak diberikan",
        );
    }
    Json(json!({
        "id": id,
        "object": "model",
        "created": unix_ts(),
        "owned_by": "cmod-inference",
        "context_length": state.config.backend.max_model_len,
    }))
    .into_response()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<OpenAiChatRequest>,
) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return openai_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "API key tidak valid atau tidak diberikan",
        );
    }

    let verbose = body.verbose
        || headers
            .get("x-cmod-verbose")
            .and_then(|v| v.to_str().ok())
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);
    let priority = extract_priority(&headers);
    let mut canonical = match parse_openai_to_canonical(
        body,
        state.backend.default_model(),
        state.config.backend.default_non_thinking,
        priority,
    ) {
        Ok(req) => req,
        Err(msg) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_request",
                msg,
            );
        }
    };

    if let Err(msg) = check_context_window(&canonical, state.config.backend.max_model_len) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "context_length_exceeded",
            msg,
        );
    }

    canonical.assigned_slot = Some(state.scheduler.assign_slot(&canonical.prefix_hash));
    let permit = match state.scheduler.acquire(canonical.priority).await {
        Ok(p) => p,
        Err(msg) => {
            return openai_error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "queue_full",
                msg,
            );
        }
    };

    if canonical.stream {
        let model_name = canonical.model.clone();
        let req_id = canonical.request_id.clone();
        let stream = match state.backend.chat_stream(canonical).await {
            Ok(s) => s,
            Err(err) => {
                return openai_error(
                    StatusCode::BAD_GATEWAY,
                    "backend_error",
                    "upstream_error",
                    err.to_string(),
                );
            }
        };

        let created = unix_ts();
        let sse_stream = async_stream::stream! {
            let _permit = permit;
            let mut stream = stream;
            let mut cur_id = req_id;
            let mut cur_model = model_name;
            let mut sent_role = false;

            while let Some(item) = stream.next().await {
                match item {
                    Ok(CanonicalStreamEvent::Start { id, model }) => {
                        cur_id = id;
                        cur_model = model;
                        if !sent_role {
                            sent_role = true;
                            let payload = json!({
                                "id": cur_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": cur_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": { "role": "assistant" },
                                    "finish_reason": Value::Null
                                }]
                            });
                            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                        }
                    }
                    Ok(CanonicalStreamEvent::ThinkingDelta(think)) => {
                        let payload = json!({
                            "id": cur_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": cur_model,
                            "choices": [{
                                "index": 0,
                                "delta": { "reasoning_content": think },
                                "finish_reason": Value::Null
                            }]
                        });
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                    }
                    Ok(CanonicalStreamEvent::TextDelta(text)) => {
                        let payload = json!({
                            "id": cur_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": cur_model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": text },
                                "finish_reason": Value::Null
                            }]
                        });
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                    }
                    Ok(CanonicalStreamEvent::ToolCallStart { index, id, name }) => {
                        let payload = json!({
                            "id": cur_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": cur_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": index,
                                        "id": id,
                                        "type": "function",
                                        "function": { "name": name, "arguments": "" }
                                    }]
                                },
                                "finish_reason": Value::Null
                            }]
                        });
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                    }
                    Ok(CanonicalStreamEvent::ToolCallDelta { index, arguments_delta }) => {
                        let payload = json!({
                            "id": cur_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": cur_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": index,
                                        "function": { "arguments": arguments_delta }
                                    }]
                                },
                                "finish_reason": Value::Null
                            }]
                        });
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                    }
                    Ok(CanonicalStreamEvent::Usage(u)) => {
                        let payload = json!({
                            "id": cur_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": cur_model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": u.input_tokens,
                                "completion_tokens": u.output_tokens,
                                "total_tokens": u.input_tokens + u.output_tokens,
                                "prompt_tokens_details": {
                                    "cached_tokens": u.cached_input_tokens
                                }
                            }
                        });
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                    }
                    Ok(CanonicalStreamEvent::Done { stop_reason }) => {
                        let payload = json!({
                            "id": cur_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": cur_model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": stop_reason.as_openai_str()
                            }]
                        });
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                    }
                    Err(err) => {
                        let payload = json!({ "error": { "message": err.to_string(), "type": "stream_error" } });
                        yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
                        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
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
                let mut msg = serde_json::Map::new();
                msg.insert("role".to_string(), json!("assistant"));
                if resp.text.is_empty() && !resp.tool_calls.is_empty() {
                    msg.insert("content".to_string(), Value::Null);
                } else {
                    msg.insert("content".to_string(), json!(resp.text));
                }
                if let Some(thinking) = resp.thinking {
                    msg.insert("reasoning_content".to_string(), json!(thinking));
                }
                if !resp.tool_calls.is_empty() {
                    let tcs: Vec<Value> = resp
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string()),
                                }
                            })
                        })
                        .collect();
                    msg.insert("tool_calls".to_string(), Value::Array(tcs));
                }

                let mut body = json!({
                    "id": resp.id,
                    "object": "chat.completion",
                    "created": unix_ts(),
                    "model": resp.model,
                    "choices": [{
                        "index": 0,
                        "message": Value::Object(msg),
                        "finish_reason": resp.stop_reason.as_openai_str(),
                    }],
                    "usage": {
                        "prompt_tokens": resp.usage.input_tokens,
                        "completion_tokens": resp.usage.output_tokens,
                        "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": resp.usage.cached_input_tokens,
                        }
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
            Err(err) => openai_error(
                StatusCode::BAD_GATEWAY,
                "backend_error",
                "upstream_error",
                err.to_string(),
            ),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAiEmbeddingsRequest {
    #[serde(default)]
    pub model: String,
    pub input: Value,
}

pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<OpenAiEmbeddingsRequest>,
) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return openai_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "API key tidak valid atau tidak diberikan",
        );
    }

    let inputs: Vec<String> = match body.input {
        Value::String(s) => vec![s],
        Value::Array(arr) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    };
    if inputs.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_input",
            "Field 'input' wajib berupa string atau array of string yang tidak kosong",
        );
    }

    let priority = extract_priority(&headers);
    let _permit = match state.scheduler.acquire(priority).await {
        Ok(p) => p,
        Err(msg) => {
            return openai_error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "queue_full",
                msg,
            );
        }
    };

    match state.backend.embeddings(body.model, inputs).await {
        Ok(resp) => {
            let data: Vec<Value> = resp
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| {
                    json!({
                        "object": "embedding",
                        "index": index,
                        "embedding": embedding,
                    })
                })
                .collect();
            Json(json!({
                "object": "list",
                "data": data,
                "model": resp.model,
                "usage": {
                    "prompt_tokens": resp.prompt_tokens,
                    "total_tokens": resp.prompt_tokens,
                }
            }))
            .into_response()
        }
        Err(err) => openai_error(
            StatusCode::BAD_GATEWAY,
            "backend_error",
            "embeddings_error",
            err.to_string(),
        ),
    }
}
