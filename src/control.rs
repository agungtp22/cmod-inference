use std::path::{Path, PathBuf};

use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::json;

use crate::auth::authorize;
use crate::models::{list_local_models, recommend_for_hardware};
use crate::presets::builtin_presets;
use crate::AppState;

pub async fn healthz() -> Response {
    Json(json!({
        "status": "ok",
        "service": "cmod-inference",
        "version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

pub async fn readyz(State(state): State<AppState>) -> Response {
    let backend = state.backend.health().await;
    let code = if backend.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        code,
        Json(json!({
            "ready": backend.ready,
            "backend": backend,
        })),
    )
        .into_response()
}

fn unauthorized_json() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "API key tidak valid atau tidak diberikan" })),
    )
        .into_response()
}

pub async fn hardware(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return unauthorized_json();
    }
    Json(state.hardware.as_ref()).into_response()
}

pub async fn status(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return unauthorized_json();
    }
    let backend = state.backend.health().await;
    let scheduler = state.scheduler.status();
    let kv_cache = state.kv_cache.stats();
    Json(json!({
        "service": "cmod-inference",
        "version": env!("CARGO_PKG_VERSION"),
        "protocols": ["openai:/v1/chat/completions", "anthropic:/v1/messages"],
        "maxModelLen": state.config.backend.max_model_len,
        "kvCacheQuant": state.config.backend.kv_cache_quant,
        "logPrompts": state.config.server.log_prompts,
        "backend": backend,
        "scheduler": scheduler,
        "kvCache": kv_cache,
        "hardware": {
            "recommendedBackend": state.hardware.recommended_backend,
            "accelerators": state.hardware.available_accelerators,
            "usableVramMb": state.hardware.total_usable_vram_mb,
        }
    }))
    .into_response()
}

pub async fn presets() -> Response {
    Json(json!({ "presets": builtin_presets() })).into_response()
}

/// Prometheus text exposition endpoint (`GET /metrics`) for Grafana & enterprise monitoring.
pub async fn metrics(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return unauthorized_json();
    }
    let backend = state.backend.health().await;
    let sc = state.scheduler.status();
    let kv = state.kv_cache.stats();
    let hw = &state.hardware;

    let body = format!(
        "# HELP cmod_inference_ready Whether the inference backend is ready (1=ready, 0=not ready)\n\
         # TYPE cmod_inference_ready gauge\n\
         cmod_inference_ready{{backend=\"{}\",model=\"{}\"}} {}\n\
         # HELP cmod_inference_scheduler_interactive_active Active interactive lane requests\n\
         # TYPE cmod_inference_scheduler_interactive_active gauge\n\
         cmod_inference_scheduler_interactive_active {}\n\
         # HELP cmod_inference_scheduler_interactive_queued Queued interactive lane requests\n\
         # TYPE cmod_inference_scheduler_interactive_queued gauge\n\
         cmod_inference_scheduler_interactive_queued {}\n\
         # HELP cmod_inference_scheduler_batch_active Active batch lane requests\n\
         # TYPE cmod_inference_scheduler_batch_active gauge\n\
         cmod_inference_scheduler_batch_active {}\n\
         # HELP cmod_inference_scheduler_batch_queued Queued batch lane requests\n\
         # TYPE cmod_inference_scheduler_batch_queued gauge\n\
         cmod_inference_scheduler_batch_queued {}\n\
         # HELP cmod_inference_requests_completed_total Total completed inference requests\n\
         # TYPE cmod_inference_requests_completed_total counter\n\
         cmod_inference_requests_completed_total {}\n\
         # HELP cmod_inference_requests_rejected_total Total rejected inference requests\n\
         # TYPE cmod_inference_requests_rejected_total counter\n\
         cmod_inference_requests_rejected_total {}\n\
         # HELP cmod_inference_kv_cache_hits_total Prefix KV-Cache hits by tier\n\
         # TYPE cmod_inference_kv_cache_hits_total counter\n\
         cmod_inference_kv_cache_hits_total{{tier=\"l1_vram\"}} {}\n\
         cmod_inference_kv_cache_hits_total{{tier=\"l2_ram\"}} {}\n\
         cmod_inference_kv_cache_hits_total{{tier=\"l3_disk\"}} {}\n\
         # HELP cmod_inference_kv_cache_misses_total Prefix KV-Cache misses\n\
         # TYPE cmod_inference_kv_cache_misses_total counter\n\
         cmod_inference_kv_cache_misses_total {}\n\
         # HELP cmod_inference_kv_cache_l2_ram_bytes Current L2 Host RAM KV-Cache bytes used\n\
         # TYPE cmod_inference_kv_cache_l2_ram_bytes gauge\n\
         cmod_inference_kv_cache_l2_ram_bytes {}\n\
         # HELP cmod_inference_kv_cache_l3_disk_files Current L3 NVMe Disk KV-Cache snapshot files\n\
         # TYPE cmod_inference_kv_cache_l3_disk_files gauge\n\
         cmod_inference_kv_cache_l3_disk_files {}\n\
         # HELP cmod_inference_hardware_usable_vram_mb Detected usable GPU VRAM / Unified Memory in MB\n\
         # TYPE cmod_inference_hardware_usable_vram_mb gauge\n\
         cmod_inference_hardware_usable_vram_mb {}\n",
        backend.kind,
        backend.model,
        if backend.ready { 1 } else { 0 },
        sc.interactive_active,
        sc.interactive_queued,
        sc.batch_active,
        sc.batch_queued,
        sc.total_completed,
        sc.total_rejected,
        kv.l1_vram_hits,
        kv.l2_ram_hits,
        kv.l3_disk_hits,
        kv.misses,
        kv.l2_ram_used_bytes,
        kv.l3_disk_files,
        hw.total_usable_vram_mb,
    );

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

pub async fn recommend_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return unauthorized_json();
    }
    let dir = Path::new(&state.config.backend.models_dir);
    let recs = recommend_for_hardware(&state.hardware, dir);
    Json(json!({
        "hardware": {
            "recommendedBackend": state.hardware.recommended_backend,
            "usableVramMb": state.hardware.total_usable_vram_mb,
            "totalMemoryMb": state.hardware.total_memory_mb,
        },
        "modelsDir": state.config.backend.models_dir,
        "recommendations": recs,
    }))
    .into_response()
}

pub async fn local_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return unauthorized_json();
    }
    let dir = Path::new(&state.config.backend.models_dir);
    Json(json!({
        "modelsDir": state.config.backend.models_dir,
        "models": list_local_models(dir),
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanModelRequest {
    pub filename: String,
    #[serde(default)]
    pub sha256: bool,
    pub expected_sha256: Option<String>,
}

pub async fn scan_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ScanModelRequest>,
) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return unauthorized_json();
    }
    let name = body.filename.trim();
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid model filename" })),
        )
            .into_response();
    }
    let path = PathBuf::from(&state.config.backend.models_dir).join(name);
    let compute_sha = body.sha256;
    let expected = body.expected_sha256.clone();
    match tokio::task::spawn_blocking(move || {
        crate::scanner::scan_gguf_model(&path, compute_sha, expected.as_deref())
    })
    .await
    {
        Ok(Ok(report)) => Json(report).into_response(),
        Ok(Err(err)) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct PullModelRequest {
    pub id: String,
}

pub async fn start_model_pull(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<PullModelRequest>,
) -> Response {
    if !authorize(&headers, &state.config.server.api_keys) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "API key tidak valid" })),
        )
            .into_response();
    }
    let dir = PathBuf::from(&state.config.backend.models_dir);
    match state.downloads.start_download(body.id, dir) {
        Ok(progress) => (StatusCode::ACCEPTED, Json(progress)).into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

pub async fn model_pull_status(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match state.downloads.get(&id) {
        Some(progress) => Json(progress).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Tidak ada job unduhan untuk '{id}'") })),
        )
            .into_response(),
    }
}
