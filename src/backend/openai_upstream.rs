use std::pin::Pin;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};

use crate::backend::{BackendHealth, EmbeddingResponse, EventStream, InferenceBackend};
use crate::config::{BackendConfig, BackendKind};
use crate::ir::{
    CanonicalRequest, CanonicalResponse, CanonicalStreamEvent, CanonicalToolCall, ContentPart,
    ResponseFormat, Role, StopReason, ToolChoice, UsageStats,
};

pub struct OpenAiUpstreamBackend {
    client: Client,
    kind: BackendKind,
    upstream_url: String,
    upstream_api_key: String,
    default_model: String,
}

impl OpenAiUpstreamBackend {
    pub fn new(cfg: &BackendConfig) -> Result<Self> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("gagal membuat HTTP client upstream")?;

        Ok(Self {
            client,
            kind: cfg.kind,
            upstream_url: cfg.upstream_url.trim_end_matches('/').to_string(),
            upstream_api_key: cfg.upstream_api_key.clone(),
            default_model: cfg.model.clone(),
        })
    }

    fn build_upstream_payload(&self, req: &CanonicalRequest) -> Value {
        let mut messages: Vec<Value> = Vec::new();

        if !req.system.is_empty() {
            let combined = req
                .system
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            messages.push(json!({
                "role": "system",
                "content": combined,
            }));
        }

        for msg in &req.messages {
            match msg.role {
                Role::User => {
                    let has_image = msg
                        .parts
                        .iter()
                        .any(|p| matches!(p, ContentPart::Image { .. }));
                    if has_image {
                        let content: Vec<Value> = msg
                            .parts
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => {
                                    Some(json!({ "type": "text", "text": text }))
                                }
                                ContentPart::Image { mime_type, base64 } => Some(json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!("data:{mime_type};base64,{base64}")
                                    }
                                })),
                                _ => None,
                            })
                            .collect();
                        messages.push(json!({ "role": "user", "content": content }));
                    } else {
                        let text = msg
                            .parts
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        messages.push(json!({ "role": "user", "content": text }));
                    }
                }
                Role::Assistant => {
                    let text = msg
                        .parts
                        .iter()
                        .filter_map(|p| match p {
                            ContentPart::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let tool_calls: Vec<Value> = msg
                        .parts
                        .iter()
                        .filter_map(|p| match p {
                            ContentPart::ToolUse {
                                id,
                                name,
                                arguments,
                            } => Some(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
                                }
                            })),
                            _ => None,
                        })
                        .collect();

                    let mut obj = serde_json::Map::new();
                    obj.insert("role".to_string(), json!("assistant"));
                    if !text.is_empty() || tool_calls.is_empty() {
                        obj.insert("content".to_string(), json!(text));
                    } else {
                        obj.insert("content".to_string(), Value::Null);
                    }
                    if !tool_calls.is_empty() {
                        obj.insert("tool_calls".to_string(), Value::Array(tool_calls));
                    }
                    messages.push(Value::Object(obj));
                }
                Role::Tool => {
                    for part in &msg.parts {
                        if let ContentPart::ToolResult {
                            tool_call_id,
                            content,
                            ..
                        } = part
                        {
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": content,
                            }));
                        }
                    }
                }
            }
        }

        let model_name = if req.model.trim().is_empty() {
            self.default_model.clone()
        } else {
            req.model.clone()
        };

        let mut payload = serde_json::Map::new();
        payload.insert("model".to_string(), json!(model_name));
        payload.insert("messages".to_string(), Value::Array(messages));
        payload.insert("stream".to_string(), json!(req.stream));

        if req.stream {
            payload.insert(
                "stream_options".to_string(),
                json!({ "include_usage": true }),
            );
        }
        if let Some(temp) = req.sampling.temperature {
            payload.insert("temperature".to_string(), json!(temp));
        }
        if let Some(top_p) = req.sampling.top_p {
            payload.insert("top_p".to_string(), json!(top_p));
        }
        if let Some(max_tokens) = req.sampling.max_tokens {
            payload.insert("max_tokens".to_string(), json!(max_tokens));
        }
        if !req.sampling.stop.is_empty() {
            payload.insert("stop".to_string(), json!(req.sampling.stop));
        }

        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description.as_deref().unwrap_or(""),
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            payload.insert("tools".to_string(), Value::Array(tools));
        }

        if let Some(choice) = &req.tool_choice {
            let val = match choice {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::None => json!("none"),
                ToolChoice::Required => json!("required"),
                ToolChoice::Named { name } => json!({
                    "type": "function",
                    "function": { "name": name }
                }),
            };
            payload.insert("tool_choice".to_string(), val);
        }

        match &req.response_format {
            ResponseFormat::Text => {}
            ResponseFormat::JsonObject => {
                payload.insert("response_format".to_string(), json!({ "type": "json_object" }));
            }
            ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => {
                payload.insert(
                    "response_format".to_string(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": name,
                            "schema": schema,
                            "strict": strict,
                        }
                    }),
                );
                if matches!(self.kind, BackendKind::Vllm) {
                    payload.insert("guided_json".to_string(), schema.clone());
                }
            }
        }

        // Backend-specific document optimizations
        if matches!(self.kind, BackendKind::Llamacpp) {
            payload.insert("cache_prompt".to_string(), json!(true));
            if let Some(slot) = req.assigned_slot {
                payload.insert("id_slot".to_string(), json!(slot));
            }
        }
        if req.sampling.non_thinking {
            payload.insert(
                "chat_template_kwargs".to_string(),
                json!({ "enable_thinking": false }),
            );
            payload.insert("think".to_string(), json!(false));
        }

        Value::Object(payload)
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if !self.upstream_api_key.trim().is_empty() {
            builder.bearer_auth(&self.upstream_api_key)
        } else {
            builder
        }
    }
}

fn map_stop_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length") | Some("max_tokens") => StopReason::MaxTokens,
        Some("tool_calls") | Some("function_call") | Some("tool_use") => StopReason::ToolUse,
        Some("stop_sequence") => StopReason::StopSequence,
        _ => StopReason::EndTurn,
    }
}

fn parse_usage(val: &Value) -> UsageStats {
    let input_tokens = val
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = val
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached_input_tokens = val
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    UsageStats {
        input_tokens,
        output_tokens,
        cached_input_tokens,
    }
}

impl InferenceBackend for OpenAiUpstreamBackend {
    fn kind_name(&self) -> &'static str {
        match self.kind {
            BackendKind::Auto => "auto",
            BackendKind::Native => "native",
            BackendKind::Vllm => "vllm",
            BackendKind::Llamacpp => "llamacpp",
            BackendKind::Passthrough => "passthrough",
            BackendKind::Cluster => "cluster-node",
            BackendKind::Mlx => "mlx-lm",
        }
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn chat_complete(
        &self,
        req: CanonicalRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<CanonicalResponse>> + Send + '_>> {
        Box::pin(async move {
            let mut non_stream = req;
            non_stream.stream = false;
            let payload = self.build_upstream_payload(&non_stream);
            let url = format!("{}/chat/completions", self.upstream_url);

            let resp = self
                .apply_auth(self.client.post(&url).json(&payload))
                .send()
                .await
                .with_context(|| format!("gagal menghubungi upstream {url}"))?;

            let status = resp.status();
            let body: Value = resp
                .json()
                .await
                .context("respons upstream bukan JSON valid")?;
            if !status.is_success() {
                bail!("upstream error ({status}): {body}");
            }

            let choice = body
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .ok_or_else(|| anyhow!("upstream tidak mengembalikan choices"))?;

            let message = choice.get("message").unwrap_or(&Value::Null);
            let text = message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let thinking = message
                .get("reasoning_content")
                .or_else(|| message.get("reasoning"))
                .and_then(Value::as_str)
                .map(|s| s.to_string());

            let mut tool_calls = Vec::new();
            if let Some(tcs) = message.get("tool_calls").and_then(Value::as_array) {
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
                    let args_raw = func
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let arguments = serde_json::from_str::<Value>(args_raw)
                        .unwrap_or_else(|_| json!({ "raw": args_raw }));
                    tool_calls.push(CanonicalToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
            }

            let stop_reason = if !tool_calls.is_empty() {
                StopReason::ToolUse
            } else {
                map_stop_reason(choice.get("finish_reason").and_then(Value::as_str))
            };
            let usage = body
                .get("usage")
                .map(parse_usage)
                .unwrap_or_default();

            Ok(CanonicalResponse {
                id: body
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| non_stream.request_id.clone()),
                model: body
                    .get("model")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| non_stream.model.clone()),
                text,
                thinking,
                tool_calls,
                stop_reason,
                usage,
            })
        })
    }

    fn chat_stream(
        &self,
        req: CanonicalRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<EventStream>> + Send + '_>> {
        Box::pin(async move {
            let mut stream_req = req;
            stream_req.stream = true;
            let payload = self.build_upstream_payload(&stream_req);
            let url = format!("{}/chat/completions", self.upstream_url);

            let resp = self
                .apply_auth(self.client.post(&url).json(&payload))
                .send()
                .await
                .with_context(|| format!("gagal membuka stream upstream {url}"))?;

            let status = resp.status();
            if !status.is_success() {
                let err_txt = resp.text().await.unwrap_or_default();
                bail!("upstream stream error ({status}): {err_txt}");
            }

            let req_id = stream_req.request_id.clone();
            let model_fallback = stream_req.model.clone();
            let mut bytes_stream = resp.bytes_stream();

            let s = async_stream::try_stream! {
                let mut buffer = String::new();
                let mut started = false;
                let mut final_stop = StopReason::EndTurn;
                let mut saw_tool_call = false;

                while let Some(chunk_res) = bytes_stream.next().await {
                    let chunk = chunk_res.context("error membaca chunk stream upstream")?;
                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    while let Some(nl_pos) = buffer.find('\n') {
                        let line = buffer[..nl_pos].trim().to_string();
                        buffer.drain(..=nl_pos);

                        if line.is_empty() || line.starts_with(':') {
                            continue;
                        }
                        let Some(data) = line.strip_prefix("data:") else {
                            continue;
                        };
                        let data = data.trim();
                        if data == "[DONE]" {
                            if !started {
                                yield CanonicalStreamEvent::Start {
                                    id: req_id.clone(),
                                    model: model_fallback.clone(),
                                };
                            }
                            yield CanonicalStreamEvent::Done {
                                stop_reason: if saw_tool_call { StopReason::ToolUse } else { final_stop },
                            };
                            return;
                        }

                        let Ok(val) = serde_json::from_str::<Value>(data) else {
                            continue;
                        };

                        if !started {
                            let id = val
                                .get("id")
                                .and_then(Value::as_str)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| req_id.clone());
                            let model = val
                                .get("model")
                                .and_then(Value::as_str)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| model_fallback.clone());
                            yield CanonicalStreamEvent::Start { id, model };
                            started = true;
                        }

                        if let Some(usage_val) = val.get("usage").filter(|u| !u.is_null()) {
                            yield CanonicalStreamEvent::Usage(parse_usage(usage_val));
                        }

                        if let Some(choice) = val
                            .get("choices")
                            .and_then(Value::as_array)
                            .and_then(|arr| arr.first())
                        {
                            if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                                final_stop = map_stop_reason(Some(fr));
                            }
                            if let Some(delta) = choice.get("delta") {
                                if let Some(think) = delta
                                    .get("reasoning_content")
                                    .or_else(|| delta.get("reasoning"))
                                    .and_then(Value::as_str)
                                {
                                    if !think.is_empty() {
                                        yield CanonicalStreamEvent::ThinkingDelta(think.to_string());
                                    }
                                }
                                if let Some(txt) = delta.get("content").and_then(Value::as_str) {
                                    if !txt.is_empty() {
                                        yield CanonicalStreamEvent::TextDelta(txt.to_string());
                                    }
                                }
                                if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                                    for tc in tcs {
                                        saw_tool_call = true;
                                        let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                                        if let Some(id) = tc.get("id").and_then(Value::as_str) {
                                            let name = tc
                                                .get("function")
                                                .and_then(|f| f.get("name"))
                                                .and_then(Value::as_str)
                                                .unwrap_or_default()
                                                .to_string();
                                            yield CanonicalStreamEvent::ToolCallStart {
                                                index: idx,
                                                id: id.to_string(),
                                                name,
                                            };
                                        }
                                        if let Some(arg_delta) = tc
                                            .get("function")
                                            .and_then(|f| f.get("arguments"))
                                            .and_then(Value::as_str)
                                        {
                                            if !arg_delta.is_empty() {
                                                yield CanonicalStreamEvent::ToolCallDelta {
                                                    index: idx,
                                                    arguments_delta: arg_delta.to_string(),
                                                };
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !started {
                    yield CanonicalStreamEvent::Start {
                        id: req_id.clone(),
                        model: model_fallback.clone(),
                    };
                }
                yield CanonicalStreamEvent::Done {
                    stop_reason: if saw_tool_call { StopReason::ToolUse } else { final_stop },
                };
            };

            Ok(Box::pin(s) as EventStream)
        })
    }

    fn health(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = BackendHealth> + Send + '_>> {
        Box::pin(async move {
            let url = format!("{}/models", self.upstream_url);
            let res = self
                .apply_auth(self.client.get(&url))
                .timeout(Duration::from_secs(3))
                .send()
                .await;

            match res {
                Ok(r) if r.status().is_success() => BackendHealth {
                    ready: true,
                    kind: self.kind_name().to_string(),
                    upstream_url: self.upstream_url.clone(),
                    model: self.default_model.clone(),
                    detail: None,
                },
                Ok(r) => BackendHealth {
                    ready: false,
                    kind: self.kind_name().to_string(),
                    upstream_url: self.upstream_url.clone(),
                    model: self.default_model.clone(),
                    detail: Some(format!("HTTP {}", r.status())),
                },
                Err(err) => BackendHealth {
                    ready: false,
                    kind: self.kind_name().to_string(),
                    upstream_url: self.upstream_url.clone(),
                    model: self.default_model.clone(),
                    detail: Some(err.to_string()),
                },
            }
        })
    }

    fn embeddings(
        &self,
        model: String,
        inputs: Vec<String>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<EmbeddingResponse>> + Send + '_>> {
        Box::pin(async move {
            let url = format!("{}/embeddings", self.upstream_url);
            let target_model = if model.trim().is_empty() {
                self.default_model.clone()
            } else {
                model
            };
            let payload = json!({
                "model": target_model,
                "input": inputs,
            });
            let resp = self
                .apply_auth(self.client.post(&url))
                .json(&payload)
                .send()
                .await
                .with_context(|| format!("gagal menghubungi upstream embeddings {url}"))?;
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(anyhow!("Upstream embeddings HTTP {status}: {text}"));
            }
            let raw: Value =
                serde_json::from_str(&text).context("response embeddings upstream bukan JSON")?;
            let mut items: Vec<(usize, Vec<f32>)> = raw
                .get("data")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .enumerate()
                        .map(|(i, item)| {
                            let idx = item
                                .get("index")
                                .and_then(Value::as_u64)
                                .map(|v| v as usize)
                                .unwrap_or(i);
                            let vec = item
                                .get("embedding")
                                .and_then(Value::as_array)
                                .map(|nums| {
                                    nums.iter()
                                        .filter_map(|n| n.as_f64().map(|f| f as f32))
                                        .collect()
                                })
                                .unwrap_or_default();
                            (idx, vec)
                        })
                        .collect()
                })
                .unwrap_or_default();
            items.sort_by_key(|(idx, _)| *idx);
            let embeddings = items.into_iter().map(|(_, v)| v).collect();
            let prompt_tokens = raw
                .get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let resp_model = raw
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(&target_model)
                .to_string();
            Ok(EmbeddingResponse {
                model: resp_model,
                embeddings,
                prompt_tokens,
            })
        })
    }
}
