use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tracing::{info, warn};

use crate::config::{BackendConfig, BackendKind};

/// Optional process supervisor that manages a local `llama-server` instance
/// (e.g. Apple Silicon Metal, Vulkan, CUDA, or CPU fallback) with cmod document defaults.
pub struct ManagedLlamaServer {
    child: Option<Child>,
}

impl ManagedLlamaServer {
    pub async fn start_if_enabled(cfg: &BackendConfig) -> Result<Option<Self>> {
        if !cfg.manage_local_process {
            return Ok(None);
        }

        let port = extract_port(&cfg.upstream_url).unwrap_or(8080);

        if matches!(cfg.kind, BackendKind::Mlx) {
            let model_ref = if !cfg.model_path.trim().is_empty() {
                cfg.model_path.trim()
            } else {
                cfg.model.trim()
            };
            let mut cmd = Command::new(&cfg.mlx_server_bin);
            cmd.arg("--model")
                .arg(model_ref)
                .arg("--host")
                .arg("127.0.0.1")
                .arg("--port")
                .arg(port.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit());

            info!(
                bin = %cfg.mlx_server_bin,
                model = %model_ref,
                port,
                "Menjalankan managed Apple MLX server (mlx_lm.server)"
            );
            let child = cmd.spawn().with_context(|| {
                format!(
                    "gagal menjalankan '{}'. Pastikan mlx-lm terpasang (pip install mlx-lm) atau set CMOD_INFERENCE_MLX_BIN",
                    cfg.mlx_server_bin
                )
            })?;

            let client = reqwest::Client::new();
            let health_url = format!("http://127.0.0.1:{port}/v1/models");
            for _ in 0..60 {
                if let Ok(r) = client.get(&health_url).timeout(Duration::from_secs(1)).send().await {
                    if r.status().is_success() {
                        info!("Managed mlx_lm.server siap menerima request");
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            return Ok(Some(Self { child: Some(child) }));
        }

        let mut cmd = Command::new(&cfg.llama_server_bin);
        cmd.arg("-m")
            .arg(&cfg.model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(cfg.max_model_len.to_string())
            .arg("--parallel")
            .arg(cfg.parallel_slots.max(1).to_string())
            .arg("-ngl")
            .arg(cfg.gpu_layers.to_string())
            .arg("--cache-type-k")
            .arg(&cfg.kv_cache_quant)
            .arg("--cache-type-v")
            .arg(&cfg.kv_cache_quant)
            .arg("--jinja");

        if !cfg.mmproj_path.trim().is_empty() {
            cmd.arg("--mmproj").arg(cfg.mmproj_path.trim());
        }
        if cfg.moe_cpu_offload {
            cmd.arg("-ot").arg(".ffn_.*_exps.=CPU");
        }
        if !cfg.split_mode.trim().is_empty() {
            cmd.arg("--split-mode").arg(cfg.split_mode.trim());
        }
        if !cfg.tensor_split.is_empty() {
            let split_csv = cfg
                .tensor_split
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(",");
            cmd.arg("--tensor-split").arg(split_csv);
        }
        if cfg.main_gpu > 0 {
            cmd.arg("--main-gpu").arg(cfg.main_gpu.to_string());
        }
        if !cfg.rpc_servers.is_empty() {
            // Pattern 2a: Distributed GGUF Sharding across LAN/Thunderbolt via ggml-rpc
            cmd.arg("--rpc").arg(cfg.rpc_servers.join(","));
        }

        cmd.stdout(Stdio::null()).stderr(Stdio::inherit());
        info!(
            bin = %cfg.llama_server_bin,
            port,
            ctx = cfg.max_model_len,
            kv_quant = %cfg.kv_cache_quant,
            "Menjalankan managed llama-server lokal"
        );
        let child = cmd
            .spawn()
            .with_context(|| format!("gagal menjalankan {}", cfg.llama_server_bin))?;

        // Wait briefly for port readiness
        let client = reqwest::Client::new();
        let health_url = format!("http://127.0.0.1:{port}/health");
        for _ in 0..30 {
            if let Ok(r) = client.get(&health_url).timeout(Duration::from_secs(1)).send().await {
                if r.status().is_success() {
                    info!("Managed llama-server siap menerima request");
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Ok(Some(Self { child: Some(child) }))
    }
}

impl Drop for ManagedLlamaServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if let Err(err) = child.start_kill() {
                warn!("gagal menghentikan managed llama-server: {err}");
            }
        }
    }
}

fn extract_port(url: &str) -> Option<u16> {
    let without_scheme = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let host_port = without_scheme.split('/').next()?;
    let (_, port_str) = host_port.rsplit_once(':')?;
    port_str.parse::<u16>().ok()
}
