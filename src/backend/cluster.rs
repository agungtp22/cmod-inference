use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use tracing::{info, warn};

use crate::backend::openai_upstream::OpenAiUpstreamBackend;
use crate::backend::{BackendHealth, EventStream, InferenceBackend};
use crate::config::{BackendConfig, ClusterNodeConfig};
use crate::ir::{CanonicalRequest, CanonicalResponse};

struct ClusterNode {
    name: String,
    url: String,
    models: Vec<String>,
    backend: OpenAiUpstreamBackend,
    healthy: AtomicBool,
    requests: AtomicU64,
}

impl ClusterNode {
    fn matches_model(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|m| m.eq_ignore_ascii_case(model))
    }

    fn is_exact_model_match(&self, model: &str) -> bool {
        !self.models.is_empty() && self.models.iter().any(|m| m.eq_ignore_ascii_case(model))
    }
}

/// Pattern 1: Multi-Node Horizontal Cluster Router.
/// Routes requests across multiple remote `cmod-inference` / OpenAI-compatible nodes using:
/// 1) Model-based routing (e.g. text node vs VLM node),
/// 2) Prefix-hash consistent affinity (preserves L1 VRAM KV-Cache hits per node),
/// 3) Automatic failover across remaining nodes on error.
pub struct ClusterBackend {
    default_model: String,
    nodes: Vec<Arc<ClusterNode>>,
}

impl ClusterBackend {
    pub fn new(cfg: &BackendConfig) -> Result<Self> {
        if cfg.cluster_nodes.is_empty() {
            bail!(
                "Backend 'cluster' membutuhkan minimal 1 node pada backend.cluster_nodes \
                 (atau env CMOD_INFERENCE_CLUSTER_NODES)"
            );
        }

        let mut nodes = Vec::with_capacity(cfg.cluster_nodes.len());
        for (idx, ncfg) in cfg.cluster_nodes.iter().enumerate() {
            let url = normalize_node_url(&ncfg.url);
            let mut node_backend_cfg = cfg.clone();
            node_backend_cfg.upstream_url = url.clone();
            if !ncfg.api_key.trim().is_empty() {
                node_backend_cfg.upstream_api_key = ncfg.api_key.trim().to_string();
            }
            let backend = OpenAiUpstreamBackend::new(&node_backend_cfg)?;
            let name = if ncfg.name.trim().is_empty() {
                format!("node-{}", idx + 1)
            } else {
                ncfg.name.trim().to_string()
            };
            info!(
                node = %name,
                url = %url,
                models = ?ncfg.models,
                "Mendaftarkan node cluster cmod-inference"
            );
            nodes.push(Arc::new(ClusterNode {
                name,
                url,
                models: ncfg.models.clone(),
                backend,
                healthy: AtomicBool::new(true),
                requests: AtomicU64::new(0),
            }));
        }

        Ok(Self {
            default_model: cfg.model.clone(),
            nodes,
        })
    }

    pub fn from_node_configs(default_model: &str, configs: Vec<ClusterNodeConfig>) -> Result<Self> {
        let mut cfg = BackendConfig::default();
        cfg.model = default_model.to_string();
        cfg.cluster_nodes = configs;
        Self::new(&cfg)
    }

    fn ordered_candidates(&self, req: &CanonicalRequest) -> Vec<Arc<ClusterNode>> {
        // Prefer exact model matches if any exist; otherwise all wildcard/matching nodes.
        let exact: Vec<Arc<ClusterNode>> = self
            .nodes
            .iter()
            .filter(|n| n.is_exact_model_match(&req.model))
            .cloned()
            .collect();
        let pool: Vec<Arc<ClusterNode>> = if !exact.is_empty() {
            exact
        } else {
            let matching: Vec<Arc<ClusterNode>> = self
                .nodes
                .iter()
                .filter(|n| n.matches_model(&req.model))
                .cloned()
                .collect();
            if matching.is_empty() {
                self.nodes.clone()
            } else {
                matching
            }
        };

        let len = pool.len();
        if len <= 1 {
            return pool;
        }

        // Consistent affinity index from prefix_hash (falls back to model name)
        let hash_seed = if req.prefix_hash.is_empty() {
            req.model.as_bytes()
        } else {
            req.prefix_hash.as_bytes()
        };
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in hash_seed {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let start = (h as usize) % len;

        let mut healthy_ordered = Vec::with_capacity(len);
        let mut unhealthy_ordered = Vec::with_capacity(len);
        for offset in 0..len {
            let node = pool[(start + offset) % len].clone();
            if node.healthy.load(Ordering::Relaxed) {
                healthy_ordered.push(node);
            } else {
                unhealthy_ordered.push(node);
            }
        }
        healthy_ordered.extend(unhealthy_ordered);
        healthy_ordered
    }
}

fn normalize_node_url(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

impl InferenceBackend for ClusterBackend {
    fn kind_name(&self) -> &'static str {
        "cluster-router"
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn chat_complete(
        &self,
        req: CanonicalRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<CanonicalResponse>> + Send + '_>> {
        Box::pin(async move {
            let candidates = self.ordered_candidates(&req);
            let mut last_err = anyhow!("tidak ada node cluster yang tersedia");
            for node in candidates {
                match node.backend.chat_complete(req.clone()).await {
                    Ok(resp) => {
                        node.healthy.store(true, Ordering::Relaxed);
                        node.requests.fetch_add(1, Ordering::Relaxed);
                        return Ok(resp);
                    }
                    Err(err) => {
                        warn!(
                            node = %node.name,
                            url = %node.url,
                            "Node cluster gagal merespons, mencoba failover ke node berikutnya: {err}"
                        );
                        node.healthy.store(false, Ordering::Relaxed);
                        last_err = err;
                    }
                }
            }
            Err(anyhow!("Semua node cluster gagal merespons: {last_err}"))
        })
    }

    fn chat_stream(
        &self,
        req: CanonicalRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<EventStream>> + Send + '_>> {
        Box::pin(async move {
            let candidates = self.ordered_candidates(&req);
            let mut last_err = anyhow!("tidak ada node cluster yang tersedia");
            for node in candidates {
                match node.backend.chat_stream(req.clone()).await {
                    Ok(stream) => {
                        node.healthy.store(true, Ordering::Relaxed);
                        node.requests.fetch_add(1, Ordering::Relaxed);
                        return Ok(stream);
                    }
                    Err(err) => {
                        warn!(
                            node = %node.name,
                            url = %node.url,
                            "Node cluster gagal streaming, mencoba failover ke node berikutnya: {err}"
                        );
                        node.healthy.store(false, Ordering::Relaxed);
                        last_err = err;
                    }
                }
            }
            Err(anyhow!("Semua node cluster gagal streaming: {last_err}"))
        })
    }

    fn health(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = BackendHealth> + Send + '_>> {
        Box::pin(async move {
            let mut healthy_count = 0usize;
            let mut summaries = Vec::with_capacity(self.nodes.len());

            for node in &self.nodes {
                let h = node.backend.health().await;
                node.healthy.store(h.ready, Ordering::Relaxed);
                if h.ready {
                    healthy_count += 1;
                }
                let reqs = node.requests.load(Ordering::Relaxed);
                summaries.push(format!(
                    "{}:{}(reqs={})",
                    node.name,
                    if h.ready { "ok" } else { "down" },
                    reqs
                ));
            }

            BackendHealth {
                ready: healthy_count > 0,
                kind: "cluster-router".to_string(),
                upstream_url: format!("cluster://{}nodes", self.nodes.len()),
                model: self.default_model.clone(),
                detail: Some(format!(
                    "nodes={}/{} healthy [{}]",
                    healthy_count,
                    self.nodes.len(),
                    summaries.join(", ")
                )),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{PriorityLane, ResponseFormat, SamplingParams};

    fn dummy_req(model: &str, prefix_hash: &str) -> CanonicalRequest {
        CanonicalRequest {
            request_id: "req-1".to_string(),
            model: model.to_string(),
            system: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            response_format: ResponseFormat::Text,
            sampling: SamplingParams {
                temperature: Some(0.0),
                top_p: None,
                max_tokens: Some(64),
                stop: Vec::new(),
                non_thinking: true,
            },
            stream: false,
            priority: PriorityLane::Interactive,
            prefix_hash: prefix_hash.to_string(),
            assigned_slot: None,
        }
    }

    #[test]
    fn routes_by_model_and_prefix_affinity_with_failover_order() {
        let cluster = ClusterBackend::from_node_configs(
            "qwen3-14b-doc",
            vec![
                ClusterNodeConfig {
                    name: "text-1".to_string(),
                    url: "http://10.0.0.1:8790".to_string(),
                    api_key: String::new(),
                    models: vec!["qwen3-14b-doc".to_string()],
                },
                ClusterNodeConfig {
                    name: "text-2".to_string(),
                    url: "http://10.0.0.2:8790".to_string(),
                    api_key: String::new(),
                    models: vec!["qwen3-14b-doc".to_string()],
                },
                ClusterNodeConfig {
                    name: "vlm-1".to_string(),
                    url: "http://10.0.0.3:8790".to_string(),
                    api_key: String::new(),
                    models: vec!["gemma-4-e4b-vlm".to_string()],
                },
            ],
        )
        .unwrap();

        // Model-based routing: VLM request goes straight to vlm-1
        let vlm_cands = cluster.ordered_candidates(&dummy_req("gemma-4-e4b-vlm", "abc123"));
        assert_eq!(vlm_cands.len(), 1);
        assert_eq!(vlm_cands[0].name, "vlm-1");

        // Consistent prefix affinity: same prefix_hash always picks the same primary node
        let r1 = cluster.ordered_candidates(&dummy_req("qwen3-14b-doc", "invoice-prefix-01"));
        let r2 = cluster.ordered_candidates(&dummy_req("qwen3-14b-doc", "invoice-prefix-01"));
        assert_eq!(r1.len(), 2);
        assert_eq!(r1[0].name, r2[0].name);

        // Failover: marking primary unhealthy moves it behind the healthy node
        r1[0].healthy.store(false, Ordering::Relaxed);
        let r3 = cluster.ordered_candidates(&dummy_req("qwen3-14b-doc", "invoice-prefix-01"));
        assert_ne!(r3[0].name, r1[0].name);
        assert_eq!(r3[1].name, r1[0].name);
    }
}
