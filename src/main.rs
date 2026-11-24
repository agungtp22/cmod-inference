use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

use cmod_inference::backend::openai_upstream::OpenAiUpstreamBackend;
use cmod_inference::backend::supervisor::ManagedLlamaServer;
use cmod_inference::backend::InferenceBackend;
use cmod_inference::cache::TieredKvCache;
use cmod_inference::config::{default_models_dir, expand_tilde, BackendKind, InferenceConfig};
use cmod_inference::hardware::detect_hardware;
use cmod_inference::models::{list_local_models, pull_model_cli, recommend_for_hardware, DownloadManager};
use cmod_inference::presets::builtin_presets;
use cmod_inference::scheduler::PriorityScheduler;
use cmod_inference::{build_router, AppState};

#[derive(Parser)]
#[command(
    name = "cmod-inference",
    version,
    about = "Standalone multi-GPU & multi-machine AI inference server (OpenAI + Anthropic + 3-Tier KV-Cache + llguidance)",
    after_help = "\
CONTOH PENGGUNAAN:
  # 1. Cek hardware GPU/VRAM & rekomendasi 30 model GGUF terbaru
  cmod-inference hardware
  cmod-inference recommend

  # 2. Unduh model dari katalog (contoh 4B non-thinking atau 0.6B mini)
  cmod-inference pull qwen3-4b-doc
  cmod-inference list

  # 3. Jalankan Native Metal/CUDA/ROCm/Vulkan (GGUF in-process + llguidance + 3-Tier KV-Cache)
  cmod-inference serve --backend native --model-path ./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \\
    --model qwen3-4b-doc --ctx 8192 --slots 2 --kv-cache-dir ./kvcache

  # 4. Jalankan MoE 30B dengan CPU Expert Offload / Multi-GPU
  cmod-inference serve --backend native --model-path ./models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf \\
    --moe-cpu-offload --split-mode layer --tensor-split 24,12

  # 5. Jalankan Multi-Machine Cluster Router (Pola 1: Prefix-Aware + Failover)
  cmod-inference serve --backend cluster \\
    --cluster-nodes 'http://192.168.1.10:8790/v1|qwen3-14b-doc,http://192.168.1.11:8790/v1|gemma-4-e4b-vlm'

  # 6. Jalankan Managed Apple MLX di macOS (butuh `pip install mlx-lm`)
  cmod-inference serve --backend mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit

Gunakan `cmod-inference serve --help` untuk melihat seluruh opsi server, offloading, multi-GPU, & cluster."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Jalankan server HTTP cmod-inference (OpenAI /v1/chat/completions + Anthropic /v1/messages)
    #[command(after_help = "\
ENDPOINT UTAMA:
  OpenAI    : POST /v1/chat/completions, GET /v1/models
  Anthropic : POST /v1/messages, POST /v1/messages/count_tokens
  Control   : GET /healthz, GET /readyz, GET /cmod/v1/status, GET /cmod/v1/hardware
  Models    : GET /cmod/v1/models/recommend, GET /cmod/v1/models/local, POST /cmod/v1/models/pull

TIPS:
  - Toggle thinking per request : kirim \"enable_thinking\": false|true (OpenAI) atau \"thinking\": {\"type\":\"disabled\"|\"enabled\"} (Anthropic)
  - Mode verbose per request    : kirim \"verbose\": true atau header X-Cmod-Verbose: 1
  - Bind LAN (0.0.0.0)          : wajib sertakan --api-key (atau CMOD_INFERENCE_API_KEYS)")]
    Serve {
        /// Path file konfigurasi TOML (opsional; flag CLI di bawah akan menimpa nilai TOML/env)
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Alamat IP bind server (127.0.0.1 untuk lokal, 0.0.0.0 untuk LAN dengan --api-key)
        #[arg(long)]
        bind: Option<String>,

        /// Port HTTP server (default: 8790)
        #[arg(short, long)]
        port: Option<u16>,

        /// API key untuk autentikasi Bearer & x-api-key (pisah koma jika lebih dari satu)
        #[arg(long)]
        api_key: Option<String>,

        /// Jenis backend: native | mlx | cluster | llamacpp | vllm | passthrough
        #[arg(short, long)]
        backend: Option<String>,

        /// Nama model default (atau repo HuggingFace untuk backend mlx)
        #[arg(short, long)]
        model: Option<String>,

        /// Path file model .gguf (native/llamacpp) atau folder model MLX
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Direktori penyimpanan model lokal .gguf untuk Web UI & /cmod/v1/models/* (default: ~/.cmod/models)
        #[arg(long, env = "CMOD_INFERENCE_MODELS_DIR")]
        models_dir: Option<PathBuf>,

        /// Path file projector multimodal mmproj-*.gguf untuk model VLM/gambar
        #[arg(long)]
        mmproj: Option<PathBuf>,

        /// Path file LoRA adapter .gguf (opsional, format: PATH atau PATH:SCALE, mis. ./adapter.gguf:0.8)
        #[arg(long)]
        lora: Option<String>,

        /// Matikan blok <think> secara default untuk semua request (setara enable_thinking=false)
        #[arg(long)]
        no_think: bool,

        /// Aktifkan same-model Multi-Token Prediction (MTP) speculative decoding (Gemma 4 / DeepSeek / Qwen3.5)
        #[arg(long)]
        mtp: bool,

        /// Path file draft model .gguf untuk speculative decoding (mis. Qwen3-0.6B / Gemma-4-E2B / assistant)
        #[arg(long)]
        draft_model: Option<PathBuf>,

        /// Jumlah maksimal token draft per langkah speculative decoding (1-16, default: 4)
        #[arg(long)]
        draft_max: Option<u32>,

        /// Panjang context window maksimal dalam token (default: 32768, min: 1024)
        #[arg(long)]
        ctx: Option<u32>,

        /// Jumlah persistent VRAM worker slot paralel (1-16, default: 4)
        #[arg(long)]
        slots: Option<u32>,

        /// Jumlah layer offload ke GPU (-1 = semua di GPU, 0 = full CPU, N = partial offload)
        #[arg(long)]
        gpu_layers: Option<i32>,

        /// Offload bobot expert MoE ke CPU RAM sambil menjaga Attention & KV-Cache tetap di VRAM
        #[arg(long)]
        moe_cpu_offload: bool,

        /// Mode pembagian Multi-GPU: layer | row | tensor | none
        #[arg(long)]
        split_mode: Option<String>,

        /// Rasio VRAM Multi-GPU dipisah koma (contoh: 24,12 atau 0.67,0.33)
        #[arg(long)]
        tensor_split: Option<String>,

        /// Indeks GPU utama (0, 1, dst.)
        #[arg(long)]
        main_gpu: Option<i32>,

        /// Kuantisasi KV-Cache: q8_0 | f16 | q5_0 | q4_0 | f32 (default: q8_0)
        #[arg(long)]
        kv_cache_quant: Option<String>,

        /// Anggaran L2 Host RAM KV-Cache dalam MB (default: 2048)
        #[arg(long)]
        kv_cache_ram_mb: Option<u64>,

        /// Anggaran L3 NVMe Disk KV-Cache dalam MB (default: 16384)
        #[arg(long)]
        kv_cache_disk_mb: Option<u64>,

        /// Direktori penyimpanan persisten L3 NVMe Disk KV-Cache (kosong = nonaktif)
        #[arg(long)]
        kv_cache_dir: Option<PathBuf>,

        /// Daftar node Pola 1 Cluster Router dipisah koma (format: http://ip:8790/v1[|model])
        #[arg(long)]
        cluster_nodes: Option<String>,

        /// Daftar worker Pola 2a ggml-rpc dipisah koma (format: host1:50052,host2:50052)
        #[arg(long)]
        rpc_servers: Option<String>,

        /// URL upstream OpenAI-compatible untuk mode passthrough/vllm/llamacpp
        #[arg(long)]
        upstream_url: Option<String>,

        /// Tampilkan log mentah C++ llama.cpp / Metal / CUDA di stderr
        #[arg(long)]
        llama_logs: bool,
    },
    /// Deteksi GPU/CPU/VRAM dan rekomendasikan backend terbaik
    Hardware,
    /// Rekomendasikan 30 model GGUF terbaru (2025-2026) yang cocok untuk hardware mesin ini
    Recommend {
        /// Folder penyimpanan model lokal (default: ~/.cmod/models)
        #[arg(long, env = "CMOD_INFERENCE_MODELS_DIR")]
        models_dir: Option<PathBuf>,
    },
    /// Unduh model GGUF (+ mmproj jika multimodal) dari katalog terkurasi
    Pull {
        /// ID model dari `cmod-inference recommend` (contoh: qwen3-0.6b-tiny, qwen3-4b-doc, gemma-4-e4b-vlm)
        model_id: String,
        /// Folder tujuan penyimpanan (default: ~/.cmod/models)
        #[arg(long, env = "CMOD_INFERENCE_MODELS_DIR")]
        models_dir: Option<PathBuf>,
    },
    /// Lihat daftar file model .gguf yang sudah terunduh di lokal
    List {
        /// Folder penyimpanan model lokal (default: ~/.cmod/models)
        #[arg(long, env = "CMOD_INFERENCE_MODELS_DIR")]
        models_dir: Option<PathBuf>,
    },
    /// Tampilkan daftar preset model dokumen cmod (JSON)
    Presets,
    /// Lihat status server & proses cmod-inference yang sedang berjalan (seperti `ollama ps`)
    #[command(visible_aliases = ["status", "process"])]
    Ps {
        /// URL server cmod-inference yang ingin dicek (default: http://127.0.0.1:8790)
        #[arg(long, default_value = "http://127.0.0.1:8790")]
        url: String,
        /// Tampilkan output mentah JSON dari /cmod/v1/status
        #[arg(long)]
        json: bool,
    },
    /// Hentikan proses server `cmod serve` yang sedang berjalan di mesin ini
    Stop {
        /// Hentikan hanya PID tertentu (opsional; default hentikan semua proses `cmod serve`)
        #[arg(long)]
        pid: Option<u32>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Hardware => {
            let report = detect_hardware();
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::Recommend { models_dir } => {
            let models_dir = models_dir
                .map(|p| expand_tilde(&p))
                .unwrap_or_else(default_models_dir);
            let hw = detect_hardware();
            let recs = recommend_for_hardware(&hw, &models_dir);
            println!(
                "Hardware: {} | Backend: {} | Usable VRAM: {} MB | RAM: {} MB\n",
                hw.cpu_brand, hw.recommended_backend, hw.total_usable_vram_mb, hw.total_memory_mb
            );
            for r in recs {
                let badge = match r.status.as_str() {
                    "recommended" => "[REKOMENDASI]",
                    "compatible" => "[COCOK]      ",
                    "conditional" => "[BERSYARAT]  ",
                    _ => "[TIDAK MUAT] ",
                };
                let inst = if r.installed { " (terpasang)" } else { "" };
                let tag_badges = r
                    .tags
                    .iter()
                    .map(|t| format!("[{t}]"))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "{badge} {:<24} {:<46} ~{:.1} GB{}",
                    r.model.id,
                    r.model.name,
                    r.model.disk_mb as f64 / 1024.0,
                    inst
                );
                println!("              ├─ Fitur: {tag_badges}");
                println!("              ├─ {}", r.reason);
                println!("              └─ Cocok untuk: {}", r.model.best_for);
            }
            println!("\nUntuk mengunduh: cmod-inference pull <model_id>");
        }
        Commands::Pull {
            model_id,
            models_dir,
        } => {
            let models_dir = models_dir
                .map(|p| expand_tilde(&p))
                .unwrap_or_else(default_models_dir);
            let mut last_pct: i32 = -1;
            let (path, mmproj) = pull_model_cli(&model_id, &models_dir, |file, dl, tot| {
                if let Some(t) = tot.filter(|&t| t > 0) {
                    let pct = ((dl as f64 / t as f64) * 100.0) as i32;
                    if pct != last_pct && (pct % 5 == 0 || pct == 100) {
                        last_pct = pct;
                        eprint!(
                            "\rMengunduh {file}: {pct}% ({:.1} / {:.1} MB)",
                            dl as f64 / 1_048_576.0,
                            t as f64 / 1_048_576.0
                        );
                        let _ = std::io::stderr().flush();
                    }
                }
            })
            .await?;
            eprintln!();
            println!("Selesai! Model tersimpan di: {}", path.display());
            if let Some(mm) = mmproj {
                println!("Multimodal projector (mmproj): {}", mm.display());
            }
        }
        Commands::List { models_dir } => {
            let models_dir = models_dir
                .map(|p| expand_tilde(&p))
                .unwrap_or_else(default_models_dir);
            let files = list_local_models(&models_dir);
            if files.is_empty() {
                println!("Belum ada file .gguf di {}", models_dir.display());
            } else {
                println!("Direktori model: {}", models_dir.display());
                for f in files {
                    println!("{:<45} {:>6} MB  ({})", f.filename, f.size_mb, f.path);
                }
            }
        }
        Commands::Presets => {
            let presets = builtin_presets();
            println!("{}", serde_json::to_string_pretty(&presets)?);
        }
        Commands::Ps { url, json } => {
            run_ps(&url, json).await?;
        }
        Commands::Stop { pid } => {
            run_stop(pid)?;
        }
        Commands::Serve {
            config,
            bind,
            port,
            api_key,
            backend,
            model,
            model_path,
            models_dir,
            mmproj,
            lora,
            no_think,
            mtp,
            draft_model,
            draft_max,
            ctx,
            slots,
            gpu_layers,
            moe_cpu_offload,
            split_mode,
            tensor_split,
            main_gpu,
            kv_cache_quant,
            kv_cache_ram_mb,
            kv_cache_disk_mb,
            kv_cache_dir,
            cluster_nodes,
            rpc_servers,
            upstream_url,
            llama_logs,
        } => {
            if llama_logs {
                std::env::set_var("CMOD_INFERENCE_LLAMA_LOGS", "1");
            }
            let mut cfg = if let Some(path) = config.as_deref() {
                let raw = std::fs::read_to_string(path)?;
                toml::from_str::<InferenceConfig>(&raw)?
            } else {
                InferenceConfig::default()
            };
            cfg.apply_env_overrides();

            if let Some(b) = bind {
                cfg.server.bind = b;
            }
            if let Some(p) = port {
                cfg.server.port = p;
            }
            if let Some(k) = api_key {
                let keys: Vec<String> = k
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !keys.is_empty() {
                    cfg.server.api_keys = keys;
                }
            }
            if let Some(bk) = backend {
                match bk.trim().to_ascii_lowercase().as_str() {
                    "auto" => cfg.backend.kind = BackendKind::Auto,
                    "native" | "embedded" | "in-process" => cfg.backend.kind = BackendKind::Native,
                    "cluster" | "router" | "multi-node" => cfg.backend.kind = BackendKind::Cluster,
                    "mlx" | "mlx-lm" | "mlx_lm" => {
                        cfg.backend.kind = BackendKind::Mlx;
                        cfg.backend.manage_local_process = true;
                    }
                    "vllm" => cfg.backend.kind = BackendKind::Vllm,
                    "llamacpp" | "llama" | "llama-server" => {
                        cfg.backend.kind = BackendKind::Llamacpp;
                        cfg.backend.manage_local_process = true;
                    }
                    "passthrough" => cfg.backend.kind = BackendKind::Passthrough,
                    other => anyhow::bail!(
                        "Backend tidak dikenal: '{other}'. Pilihan: native, mlx, cluster, llamacpp, vllm, passthrough"
                    ),
                }
            }
            if let Some(m) = model {
                cfg.backend.model = m;
            }
            if let Some(md) = models_dir.as_ref() {
                cfg.backend.models_dir = expand_tilde(md).display().to_string();
            }
            let effective_models_dir = expand_tilde(std::path::Path::new(&cfg.backend.models_dir));
            let resolve_in_models_dir = |raw: &std::path::Path| -> PathBuf {
                let expanded = expand_tilde(raw);
                if expanded.exists() {
                    return expanded;
                }
                let candidate = effective_models_dir.join(&expanded);
                if candidate.exists() {
                    return candidate;
                }
                expanded
            };
            if let Some(mp) = model_path {
                let resolved = resolve_in_models_dir(&mp);
                let default_dir_str = default_models_dir().display().to_string();
                if models_dir.is_none()
                    && (cfg.backend.models_dir == default_dir_str
                        || cfg.backend.models_dir == "./models")
                {
                    if let Some(parent) = resolved.parent().filter(|p| !p.as_os_str().is_empty()) {
                        cfg.backend.models_dir = parent.display().to_string();
                    }
                }
                cfg.backend.model_path = resolved.display().to_string();
                if matches!(cfg.backend.kind, BackendKind::Llamacpp) {
                    cfg.backend.manage_local_process = true;
                }
            }
            if let Some(mm) = mmproj {
                cfg.backend.mmproj_path = resolve_in_models_dir(&mm).display().to_string();
            }
            if let Some(lora_spec) = lora {
                let spec = lora_spec.trim();
                if let Some((path_part, scale_part)) = spec.rsplit_once(':') {
                    if let Ok(scale) = scale_part.trim().parse::<f32>() {
                        cfg.backend.lora_path = path_part.trim().to_string();
                        cfg.backend.lora_scale = scale;
                    } else {
                        cfg.backend.lora_path = spec.to_string();
                        cfg.backend.lora_scale = 1.0;
                    }
                } else {
                    cfg.backend.lora_path = spec.to_string();
                    cfg.backend.lora_scale = 1.0;
                }
            }
            if no_think {
                cfg.backend.default_non_thinking = true;
            }
            if mtp {
                cfg.backend.enable_mtp = true;
            }
            if let Some(dm) = draft_model {
                cfg.backend.draft_model_path = resolve_in_models_dir(&dm).display().to_string();
            }
            if let Some(dmax) = draft_max {
                cfg.backend.draft_max = dmax.clamp(1, 16);
            }
            if let Some(c) = ctx {
                cfg.backend.max_model_len = c;
            }
            if let Some(s) = slots {
                cfg.backend.parallel_slots = s.clamp(1, 16);
            }
            if let Some(gl) = gpu_layers {
                cfg.backend.gpu_layers = gl;
            }
            if moe_cpu_offload {
                cfg.backend.moe_cpu_offload = true;
            }
            if let Some(sm) = split_mode {
                let mode = sm.trim().to_ascii_lowercase();
                if matches!(mode.as_str(), "none" | "layer" | "row" | "tensor") {
                    cfg.backend.split_mode = mode;
                } else {
                    anyhow::bail!(
                        "split-mode tidak dikenal: '{sm}'. Pilihan: layer, row, tensor, none"
                    );
                }
            }
            if let Some(ts) = tensor_split {
                let splits: Vec<f32> = ts
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .filter(|&x| x >= 0.0)
                    .collect();
                if !splits.is_empty() {
                    cfg.backend.tensor_split = splits;
                }
            }
            if let Some(mg) = main_gpu {
                cfg.backend.main_gpu = mg.max(0);
            }
            if let Some(kq) = kv_cache_quant {
                cfg.backend.kv_cache_quant = kq;
            }
            if let Some(kr) = kv_cache_ram_mb {
                cfg.backend.kv_cache_ram_mb = kr;
            }
            if let Some(kd) = kv_cache_disk_mb {
                cfg.backend.kv_cache_disk_mb = kd;
            }
            if let Some(dir) = kv_cache_dir {
                cfg.backend.kv_cache_dir = dir.display().to_string();
            }
            if let Some(nodes_raw) = cluster_nodes {
                let nodes: Vec<cmod_inference::config::ClusterNodeConfig> = nodes_raw
                    .split(',')
                    .enumerate()
                    .filter_map(|(i, raw)| {
                        let s = raw.trim();
                        if s.is_empty() {
                            return None;
                        }
                        let mut parts = s.splitn(2, '|');
                        let url = parts.next()?.trim().trim_end_matches('/').to_string();
                        let models = parts
                            .next()
                            .map(|m| {
                                m.split('+')
                                    .map(|x| x.trim().to_string())
                                    .filter(|x| !x.is_empty())
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some(cmod_inference::config::ClusterNodeConfig {
                            name: format!("node-{}", i + 1),
                            url,
                            api_key: cfg.backend.upstream_api_key.clone(),
                            models,
                        })
                    })
                    .collect();
                if !nodes.is_empty() {
                    cfg.backend.cluster_nodes = nodes;
                }
            }
            if let Some(rpcs) = rpc_servers {
                let list: Vec<String> = rpcs
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !list.is_empty() {
                    cfg.backend.rpc_servers = list;
                }
            }
            if let Some(u) = upstream_url {
                cfg.backend.upstream_url = u.trim().trim_end_matches('/').to_string();
            }

            cfg.validate()?;
            let hw = Arc::new(detect_hardware());
            let kv_dir = if cfg.backend.kv_cache_dir.trim().is_empty() {
                None
            } else {
                Some(PathBuf::from(cfg.backend.kv_cache_dir.trim()))
            };
            let kv_cache = TieredKvCache::new(
                cfg.backend.kv_cache_ram_mb,
                cfg.backend.kv_cache_disk_mb,
                kv_dir,
            );

            let (_managed, backend): (Option<ManagedLlamaServer>, Arc<dyn InferenceBackend>) =
                if matches!(cfg.backend.kind, BackendKind::Native) {
                    #[cfg(feature = "native")]
                    {
                        let native =
                            cmod_inference::backend::native_llama::NativeLlamaBackend::load(
                                &cfg.backend,
                                kv_cache.clone(),
                            )?;
                        (None, Arc::new(native))
                    }
                    #[cfg(not(feature = "native"))]
                    {
                        anyhow::bail!(
                            "Backend 'native' memerlukan build dengan feature native/metal/cuda/rocm/vulkan \
                             (contoh: cargo build --release --features metal)"
                        );
                    }
                } else if matches!(cfg.backend.kind, BackendKind::Cluster) {
                    let cluster =
                        cmod_inference::backend::cluster::ClusterBackend::new(&cfg.backend)?;
                    (None, Arc::new(cluster))
                } else {
                    let managed = ManagedLlamaServer::start_if_enabled(&cfg.backend).await?;
                    let upstream = Arc::new(OpenAiUpstreamBackend::new(&cfg.backend)?);
                    (managed, upstream)
                };

            let scheduler = PriorityScheduler::new(&cfg.scheduler, cfg.backend.parallel_slots);

            let addr: SocketAddr = format!("{}:{}", cfg.server.bind, cfg.server.port).parse()?;
            info!(
                %addr,
                backend = backend.kind_name(),
                upstream = %cfg.backend.upstream_url,
                recommended_hw = %hw.recommended_backend,
                max_model_len = cfg.backend.max_model_len,
                "cmod-inference aktif (OpenAI /v1/chat/completions + Anthropic /v1/messages)"
            );

            let state = AppState {
                config: Arc::new(cfg),
                backend,
                scheduler,
                hardware: hw,
                kv_cache,
                downloads: DownloadManager::new(),
            };

            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, build_router(state))
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("Menerima sinyal berhenti, menutup cmod-inference...");
}

fn find_cmod_serve_processes() -> Vec<(u32, u64, String)> {
    use sysinfo::{ProcessesToUpdate, System};
    let my_pid = std::process::id();
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, true);

    let mut found = Vec::new();
    for (pid, proc_) in sys.processes() {
        let pid_u32 = pid.as_u32();
        if pid_u32 == my_pid {
            continue;
        }
        let name = proc_.name().to_string_lossy().to_string();
        let cmd_str = proc_
            .cmd()
            .iter()
            .map(|s| s.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ");
        let is_cmod = name.contains("cmod-inference")
            || name == "cmod"
            || cmd_str.contains("cmod-inference serve")
            || cmd_str.contains("cmod serve");
        if is_cmod && cmd_str.contains("serve") {
            let rss_mb = proc_.memory() / (1024 * 1024);
            found.push((pid_u32, rss_mb, cmd_str));
        }
    }
    found.sort_by_key(|(pid, _, _)| *pid);
    found
}

async fn run_ps(base_url: &str, raw_json: bool) -> Result<()> {
    let procs = find_cmod_serve_processes();
    let status_url = format!("{}/cmod/v1/status", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let status_val: Option<serde_json::Value> = match client.get(&status_url).send().await {
        Ok(resp) if resp.status().is_success() => resp.json().await.ok(),
        _ => None,
    };

    if raw_json {
        if let Some(val) = status_val {
            println!("{}", serde_json::to_string_pretty(&val)?);
        } else {
            println!("{{\"running\": false, \"url\": \"{}\"}}", status_url);
        }
        return Ok(());
    }

    if procs.is_empty() && status_val.is_none() {
        println!("Tidak ada proses atau server cmod-inference yang aktif di {base_url}.");
        println!("Jalankan dengan: cmod serve --backend native --model-path <file.gguf>");
        return Ok(());
    }

    if !procs.is_empty() {
        println!("PROSES LOKAL AKTIF:");
        println!("{:<8} {:>8}  COMMAND", "PID", "RAM");
        for (pid, rss_mb, cmd) in &procs {
            println!("{:<8} {:>5} MB  {}", pid, rss_mb, cmd);
        }
        println!();
    }

    if let Some(st) = status_val {
        let bk = &st["backend"];
        let sc = &st["scheduler"];
        let kv = &st["kvCache"];
        let hw = &st["hardware"];
        println!("SERVER AKTIF ({base_url}):");
        println!(
            "  Model      : {} ({})",
            bk["model"].as_str().unwrap_or("-"),
            bk["kind"].as_str().unwrap_or("-")
        );
        println!(
            "  Ready      : {} | Context: {} | KV Quant: {}",
            bk["ready"].as_bool().unwrap_or(false),
            st["maxModelLen"].as_u64().unwrap_or(0),
            st["kvCacheQuant"].as_str().unwrap_or("-")
        );
        if let Some(detail) = bk["detail"].as_str() {
            println!("  Engine     : {detail}");
        }
        println!(
            "  Scheduler  : interactive {}/{} (queue {}), batch {}/{} (queue {}), selesai {}",
            sc["interactiveActive"].as_u64().unwrap_or(0),
            sc["interactiveSlots"].as_u64().unwrap_or(0),
            sc["interactiveQueued"].as_u64().unwrap_or(0),
            sc["batchActive"].as_u64().unwrap_or(0),
            sc["batchSlots"].as_u64().unwrap_or(0),
            sc["batchQueued"].as_u64().unwrap_or(0),
            sc["totalCompleted"].as_u64().unwrap_or(0),
        );
        println!(
            "  KV-Cache   : L1 VRAM hits={} | L2 RAM={} entries ({:.1} MB, hits={}) | L3 Disk={} files (hits={})",
            kv["l1VramHits"].as_u64().unwrap_or(0),
            kv["l2Entries"].as_u64().unwrap_or(0),
            kv["l2RamUsedBytes"].as_u64().unwrap_or(0) as f64 / 1_048_576.0,
            kv["l2RamHits"].as_u64().unwrap_or(0),
            kv["l3DiskFiles"].as_u64().unwrap_or(0),
            kv["l3DiskHits"].as_u64().unwrap_or(0),
        );
        println!(
            "  Hardware   : {} | Usable VRAM: {} MB",
            hw["recommendedBackend"].as_str().unwrap_or("-"),
            hw["usableVramMb"].as_u64().unwrap_or(0)
        );
    } else {
        println!("Catatan: endpoint {status_url} tidak merespons (mungkin berjalan di port lain; gunakan --url http://127.0.0.1:<PORT>).");
    }

    Ok(())
}

fn run_stop(target_pid: Option<u32>) -> Result<()> {
    use sysinfo::{Pid, ProcessesToUpdate, Signal, System};
    let procs = find_cmod_serve_processes();
    let targets: Vec<(u32, String)> = match target_pid {
        Some(p) => procs
            .into_iter()
            .filter(|(pid, _, _)| *pid == p)
            .map(|(pid, _, cmd)| (pid, cmd))
            .collect(),
        None => procs.into_iter().map(|(pid, _, cmd)| (pid, cmd)).collect(),
    };

    if targets.is_empty() {
        println!("Tidak ada proses `cmod serve` yang sedang berjalan.");
        return Ok(());
    }

    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, true);
    for (pid, cmd) in targets {
        if let Some(p) = sys.process(Pid::from_u32(pid)) {
            if p.kill_with(Signal::Term).unwrap_or(false) || p.kill() {
                println!("Menghentikan PID {pid} ({cmd})");
            } else {
                println!("Gagal menghentikan PID {pid}");
            }
        }
    }
    Ok(())
}
