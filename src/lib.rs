pub mod auth;
pub mod backend;
pub mod cache;
pub mod config;
pub mod control;
pub mod hardware;
pub mod ir;
pub mod models;
pub mod presets;
pub mod protocol;
pub mod scheduler;
pub mod ui;

use std::sync::Arc;

use axum::http::{HeaderValue, Method};
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::backend::InferenceBackend;
use crate::cache::TieredKvCache;
use crate::config::{is_loopback_bind, InferenceConfig, ServerConfig};
use crate::hardware::HardwareReport;
use crate::models::DownloadManager;
use crate::scheduler::PriorityScheduler;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<InferenceConfig>,
    pub backend: Arc<dyn InferenceBackend>,
    pub scheduler: Arc<PriorityScheduler>,
    pub hardware: Arc<HardwareReport>,
    pub kv_cache: Arc<TieredKvCache>,
    pub downloads: Arc<DownloadManager>,
}

fn build_cors_layer(cfg: &ServerConfig) -> CorsLayer {
    if cfg.cors_origins.iter().any(|o| o.trim() == "*") {
        return CorsLayer::permissive();
    }
    if !cfg.cors_origins.is_empty() {
        let allowed: Vec<HeaderValue> = cfg
            .cors_origins
            .iter()
            .filter_map(|o| HeaderValue::from_str(o.trim()).ok())
            .collect();
        return CorsLayer::new()
            .allow_origin(AllowOrigin::list(allowed))
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(Any);
    }
    if is_loopback_bind(&cfg.bind) {
        CorsLayer::permissive()
    } else {
        // Non-loopback (0.0.0.0 or specific LAN/VPC IP) without --cors-origin:
        // restrict to same-origin only (no Access-Control-Allow-Origin wildcard).
        CorsLayer::new()
    }
}

pub fn build_router(state: AppState) -> Router {
    let cors = build_cors_layer(&state.config.server);
    Router::new()
        // Built-in Web GUI (Chat Playground + Live Monitor + Model Manager + Setup)
        .route("/", get(ui::index))
        // OpenAI-compatible API
        .route("/v1/chat/completions", post(protocol::openai::chat_completions))
        .route("/v1/embeddings", post(protocol::openai::embeddings))
        .route("/v1/models", get(protocol::openai::list_models))
        .route("/v1/models/{id}", get(protocol::openai::get_model))
        // Anthropic Messages API
        .route("/v1/messages", post(protocol::anthropic::messages))
        .route("/v1/messages/count_tokens", post(protocol::anthropic::count_tokens))
        // cmod Control, Telemetry, Prometheus & Model Recommender/Downloader Plane
        .route("/healthz", get(control::healthz))
        .route("/readyz", get(control::readyz))
        .route("/metrics", get(control::metrics))
        .route("/cmod/v1/hardware", get(control::hardware))
        .route("/cmod/v1/status", get(control::status))
        .route("/cmod/v1/presets", get(control::presets))
        .route("/cmod/v1/models/recommend", get(control::recommend_models))
        .route("/cmod/v1/models/local", get(control::local_models))
        .route("/cmod/v1/models/pull", post(control::start_model_pull))
        .route("/cmod/v1/models/pull/{id}", get(control::model_pull_status))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
