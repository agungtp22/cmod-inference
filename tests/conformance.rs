use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use reqwest::Client;
use serde_json::{json, Value};

use cmod_inference::backend::openai_upstream::OpenAiUpstreamBackend;
use cmod_inference::cache::TieredKvCache;
use cmod_inference::config::{BackendConfig, BackendKind, InferenceConfig, SchedulerConfig, ServerConfig};
use cmod_inference::hardware::detect_hardware;
use cmod_inference::models::DownloadManager;
use cmod_inference::scheduler::PriorityScheduler;
use cmod_inference::{build_router, AppState};

#[derive(Clone, Default)]
struct MockUpstreamState {
    last_payload: Arc<Mutex<Option<Value>>>,
}

async fn mock_models() -> Response {
    Json(json!({
        "object": "list",
        "data": [{ "id": "mock-qwen-14b", "object": "model" }]
    }))
    .into_response()
}

async fn mock_chat(
    State(st): State<MockUpstreamState>,
    Json(body): Json<Value>,
) -> Response {
    *st.last_payload.lock().unwrap() = Some(body.clone());
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let has_tools = body
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    if stream {
        let chunks = if has_tools {
            vec![
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_benefit_1",
                                "type": "function",
                                "function": { "name": "lookup_benefit", "arguments": "" }
                            }]
                        },
                        "finish_reason": null
                    }]
                }),
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "function": { "arguments": "{\"code\":\"06\"}" }
                            }]
                        },
                        "finish_reason": null
                    }]
                }),
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 18,
                        "total_tokens": 138,
                        "prompt_tokens_details": { "cached_tokens": 64 }
                    }
                }),
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls"
                    }]
                }),
            ]
        } else {
            vec![
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [{
                        "index": 0,
                        "delta": { "content": "Total tagihan " },
                        "finish_reason": null
                    }]
                }),
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [{
                        "index": 0,
                        "delta": { "content": "Rp 350.000 [Halaman 2]." },
                        "finish_reason": null
                    }]
                }),
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 90,
                        "completion_tokens": 12,
                        "total_tokens": 102,
                        "prompt_tokens_details": { "cached_tokens": 40 }
                    }
                }),
                json!({
                    "id": "chatcmpl-mock-1",
                    "model": "mock-qwen-14b",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }),
            ]
        };

        let sse = async_stream::stream! {
            for c in chunks {
                yield Ok::<Event, Infallible>(Event::default().data(c.to_string()));
            }
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };
        Sse::new(sse).into_response()
    } else if has_tools {
        Json(json!({
            "id": "chatcmpl-mock-tool",
            "model": "mock-qwen-14b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_benefit_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_benefit",
                            "arguments": "{\"code\":\"06\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 15,
                "total_tokens": 115,
                "prompt_tokens_details": { "cached_tokens": 50 }
            }
        }))
        .into_response()
    } else {
        Json(json!({
            "id": "chatcmpl-mock-ok",
            "model": "mock-qwen-14b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "{\"benefit_code\":\"06\",\"confidence\":0.94}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 14,
                "total_tokens": 94,
                "prompt_tokens_details": { "cached_tokens": 40 }
            }
        }))
        .into_response()
    }
}

async fn spawn_test_stack(
    api_keys: Vec<String>,
    max_model_len: u32,
) -> (String, MockUpstreamState) {
    let mock_state = MockUpstreamState::default();
    let mock_router = Router::new()
        .route("/v1/models", get(mock_models))
        .route("/v1/chat/completions", post(mock_chat))
        .with_state(mock_state.clone());

    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(mock_listener, mock_router).await.unwrap();
    });

    let cfg = InferenceConfig {
        server: ServerConfig {
            bind: "127.0.0.1".to_string(),
            port: 0,
            api_keys,
            cors_origins: Vec::new(),
            allow_unauthenticated_lan: false,
            log_prompts: false,
        },
        backend: BackendConfig {
            kind: BackendKind::Llamacpp,
            upstream_url: format!("http://{mock_addr}/v1"),
            upstream_api_key: String::new(),
            model: "cmod-doc-extract".to_string(),
            max_model_len,
            parallel_slots: 4,
            ..BackendConfig::default()
        },
        scheduler: SchedulerConfig {
            interactive_slots: 4,
            batch_slots: 2,
            queue_timeout_secs: 10,
        },
    };

    let backend = Arc::new(OpenAiUpstreamBackend::new(&cfg.backend).unwrap());
    let scheduler = PriorityScheduler::new(&cfg.scheduler, cfg.backend.parallel_slots);
    let hardware = Arc::new(detect_hardware());
    let kv_cache = TieredKvCache::new(512, 2048, None);
    let state = AppState {
        config: Arc::new(cfg),
        backend,
        scheduler,
        hardware,
        kv_cache,
        downloads: DownloadManager::new(),
    };

    let gw_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let gw_addr = gw_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(gw_listener, build_router(state)).await.unwrap();
    });

    (format!("http://{gw_addr}"), mock_state)
}

#[tokio::test]
async fn openai_non_stream_json_schema_and_multimodal_base64() {
    let (base, mock) = spawn_test_stack(vec!["cmod-secret".to_string()], 32_768).await;
    let client = Client::new();

    // Unauthorized without key
    let unauth = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "cmod-doc-extract",
            "messages": [{"role": "user", "content": "Halo"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(unauth.status(), 401);

    // Valid OpenAI request with multimodal base64 + strict json_schema
    let res = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("cmod-secret")
        .header("X-Cmod-Priority", "interactive")
        .json(&json!({
            "model": "cmod-doc-extract",
            "messages": [
                { "role": "system", "content": "Ekstrak kode benefit secara deterministik." },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Cek kuitansi ini" },
                        { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBORw0KGgo=" } }
                    ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "benefit_map",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": { "benefit_code": { "type": "string" } },
                        "required": ["benefit_code"]
                    }
                }
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 200);
    let body: Value = res.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "{\"benefit_code\":\"06\",\"confidence\":0.94}"
    );
    assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 40);

    // Verify upstream received llama-server optimizations (cache_prompt, id_slot, non-thinking, image_url)
    let sent = mock.last_payload.lock().unwrap().clone().unwrap();
    assert_eq!(sent["cache_prompt"], json!(true));
    assert!(sent["id_slot"].is_u64());
    assert_eq!(sent["chat_template_kwargs"]["enable_thinking"], json!(false));
    assert_eq!(
        sent["messages"][1]["content"][1]["type"],
        json!("image_url")
    );
}

#[tokio::test]
async fn anthropic_messages_streaming_and_tool_use_and_count_tokens() {
    let (base, _) = spawn_test_stack(vec!["ant-key-99".to_string()], 32_768).await;
    let client = Client::new();

    // 1. count_tokens
    let cnt = client
        .post(format!("{base}/v1/messages/count_tokens"))
        .header("x-api-key", "ant-key-99")
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "cmod-doc-chat",
            "system": [{ "type": "text", "text": "Anda asisten dokumen RS.", "cache_control": { "type": "ephemeral" } }],
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "Berapa total tagihan?" },
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo=" } }
                ]
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(cnt.status(), 200);
    let cnt_json: Value = cnt.json().await.unwrap();
    assert!(cnt_json["input_tokens"].as_u64().unwrap() >= 1024);

    // 2. Streaming text SSE sequence
    let stream_res = client
        .post(format!("{base}/v1/messages"))
        .header("x-api-key", "ant-key-99")
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "cmod-doc-chat",
            "max_tokens": 512,
            "stream": true,
            "system": [{ "type": "text", "text": "Anda asisten dokumen.", "cache_control": { "type": "ephemeral" } }],
            "messages": [{ "role": "user", "content": "Ringkas halaman 2" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(stream_res.status(), 200);
    let sse_text = stream_res.text().await.unwrap();
    assert!(sse_text.contains("event: message_start"));
    assert!(sse_text.contains("event: content_block_start"));
    assert!(sse_text.contains("event: content_block_delta"));
    assert!(sse_text.contains("Total tagihan "));
    assert!(sse_text.contains("Rp 350.000 [Halaman 2]."));
    assert!(sse_text.contains("event: content_block_stop"));
    assert!(sse_text.contains("event: message_delta"));
    assert!(sse_text.contains("event: message_stop"));

    // 3. Non-streaming tool_use roundtrip
    let tool_res = client
        .post(format!("{base}/v1/messages"))
        .header("x-api-key", "ant-key-99")
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "cmod-doc-extract",
            "max_tokens": 256,
            "tools": [{
                "name": "lookup_benefit",
                "description": "Cari kode benefit",
                "input_schema": {
                    "type": "object",
                    "properties": { "code": { "type": "string" } }
                }
            }],
            "messages": [{ "role": "user", "content": "Cari kode konsultasi" }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(tool_res.status(), 200);
    let tool_json: Value = tool_res.json().await.unwrap();
    assert_eq!(tool_json["stop_reason"], "tool_use");
    assert_eq!(tool_json["content"][0]["type"], "tool_use");
    assert_eq!(tool_json["content"][0]["name"], "lookup_benefit");
    assert_eq!(tool_json["content"][0]["input"]["code"], "06");
}

#[tokio::test]
async fn ssrf_guard_and_context_window_guard_and_control_plane() {
    let (base, _) = spawn_test_stack(vec![], 2048).await;
    let client = Client::new();

    // 1. SSRF guard rejects remote image URL
    let ssrf = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "cmod-doc-vision",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": "https://evil.example/image.png" } }
                ]
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ssrf.status(), 400);
    let ssrf_body: Value = ssrf.json().await.unwrap();
    assert!(ssrf_body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("SSRF"));

    // 2. Context window guard prevents silent truncation
    let huge_text = "dokumen ".repeat(3000);
    let ctx = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "cmod-doc-chat",
            "max_tokens": 512,
            "messages": [{ "role": "user", "content": huge_text }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ctx.status(), 400);
    let ctx_body: Value = ctx.json().await.unwrap();
    assert_eq!(ctx_body["error"]["code"], "context_length_exceeded");

    // 3. Control & Telemetry Plane
    let hw: Value = client
        .get(format!("{base}/cmod/v1/hardware"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(hw["recommendedBackend"].is_string());

    let st: Value = client
        .get(format!("{base}/cmod/v1/status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(st["backend"]["ready"], json!(true));
    assert_eq!(st["logPrompts"], json!(false));

    let recs: Value = client
        .get(format!("{base}/cmod/v1/models/recommend"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let arr = recs["recommendations"].as_array().unwrap();
    assert!(!arr.is_empty());
    assert_eq!(arr[0]["status"], json!("recommended"));

    // Prometheus /metrics endpoint
    let prom_resp = client.get(format!("{base}/metrics")).send().await.unwrap();
    assert_eq!(prom_resp.status(), 200);
    let prom_text = prom_resp.text().await.unwrap();
    assert!(prom_text.contains("cmod_inference_ready"));
    assert!(prom_text.contains("cmod_inference_kv_cache_hits_total{tier=\"l1_vram\"}"));
    assert!(prom_text.contains("cmod_inference_hardware_usable_vram_mb"));
}
