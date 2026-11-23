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

pub async fn hardware(State(state): State<AppState>) -> Response {
    Json(state.hardware.as_ref()).into_response()
}

pub async fn status(State(state): State<AppState>) -> Response {
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

pub async fn recommend_models(State(state): State<AppState>) -> Response {
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

pub async fn local_models(State(state): State<AppState>) -> Response {
    let dir = Path::new(&state.config.backend.models_dir);
    Json(json!({
        "modelsDir": state.config.backend.models_dir,
        "models": list_local_models(dir),
    }))
    .into_response()
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
