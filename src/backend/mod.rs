pub mod cluster;
pub mod native_llama;
pub mod openai_upstream;
pub mod parser;
pub mod supervisor;

use std::pin::Pin;

use anyhow::Result;
use futures::Stream;
use serde::Serialize;

use crate::ir::{CanonicalRequest, CanonicalResponse, CanonicalStreamEvent};

pub type EventStream = Pin<Box<dyn Stream<Item = Result<CanonicalStreamEvent>> + Send>>;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendHealth {
    pub ready: bool,
    pub kind: String,
    pub upstream_url: String,
    pub model: String,
    pub detail: Option<String>,
}

pub trait InferenceBackend: Send + Sync {
    fn kind_name(&self) -> &'static str;

    fn default_model(&self) -> &str;

    fn chat_complete(
        &self,
        req: CanonicalRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<CanonicalResponse>> + Send + '_>>;

    fn chat_stream(
        &self,
        req: CanonicalRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<EventStream>> + Send + '_>>;

    fn health(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = BackendHealth> + Send + '_>>;
}
