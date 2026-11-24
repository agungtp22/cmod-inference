#[cfg(feature = "native")]
mod imp {
    use std::num::NonZeroU32;
    use std::path::Path;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use anyhow::{anyhow, Context, Result};
    use llama_cpp_2::context::params::{
        KvCacheType, LlamaContextParams, LlamaContextType, LlamaPoolingType,
    };
    use llama_cpp_2::context::session::LlamaStateSeqFlags;
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
    use llama_cpp_2::model::{LlamaChatMessage, LlamaLoraAdapter, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};
    use llama_cpp_2::token::LlamaToken;
    use tokio::sync::mpsc;
    use tracing::{info, warn};

    use crate::backend::parser::{
        format_gemma4_tools_block, format_tools_system_block, NativeOutputParser,
    };
    use crate::backend::{BackendHealth, EmbeddingResponse, EventStream, InferenceBackend};
    use crate::cache::{compute_model_fingerprint, NativeKvSnapshot, TieredKvCache};
    use crate::config::BackendConfig;
    use crate::ir::{
        CanonicalRequest, CanonicalResponse, CanonicalStreamEvent, CanonicalToolCall, ContentPart,
        ResponseFormat, Role, StopReason, ToolChoice, ToolDefinition, UsageStats,
    };

    struct GenerationJob {
        req: CanonicalRequest,
        tx: mpsc::UnboundedSender<Result<CanonicalStreamEvent>>,
    }

    struct NativeEngineInner {
        backend: LlamaBackend,
        model: LlamaModel,
        model_id: String,
        model_path: String,
        model_fingerprint: String,
        max_model_len: u32,
        gpu_layers: u32,
        split_mode: String,
        moe_cpu_offload: bool,
        kv_cache_quant: KvCacheType,
        n_threads: i32,
        kv_cache: Arc<TieredKvCache>,
        #[cfg(feature = "grammar")]
        tok_env: toktrie::TokEnv,
        #[cfg(feature = "multimodal")]
        mmproj_path: Option<String>,
        lora_adapter: Option<(std::sync::Mutex<LlamaLoraAdapter>, f32, String)>,
        enable_mtp: bool,
        draft_model: Option<LlamaModel>,
        draft_max: usize,
    }

    unsafe impl Send for NativeEngineInner {}
    unsafe impl Sync for NativeEngineInner {}

    /// Persistent VRAM Context Pool: spawns `parallel_slots` dedicated worker threads,
    /// each owning ONE pre-allocated `LlamaContext` to prevent per-request VRAM OOM spikes
    /// and enable zero-copy L1 VRAM prefix reuse.
    pub struct NativeLlamaBackend {
        inner: Arc<NativeEngineInner>,
        worker_senders: Vec<std::sync::mpsc::Sender<GenerationJob>>,
        rr: AtomicUsize,
    }

    fn parse_kv_cache_type(s: &str) -> KvCacheType {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" => KvCacheType::F32,
            "f16" => KvCacheType::F16,
            "q4_0" => KvCacheType::Q4_0,
            "q5_0" => KvCacheType::Q5_0,
            _ => KvCacheType::Q8_0,
        }
    }

    fn active_accelerator_label() -> &'static str {
        #[cfg(feature = "cuda")]
        {
            return "native-cuda";
        }
        #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
        {
            return "native-rocm";
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "metal"))]
        {
            return "native-metal";
        }
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "rocm"),
            not(feature = "metal"),
            feature = "vulkan"
        ))]
        {
            return "native-vulkan";
        }
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "rocm"),
            not(feature = "metal"),
            not(feature = "vulkan")
        ))]
        {
            "native-cpu"
        }
    }

    impl NativeLlamaBackend {
        pub fn load(cfg: &BackendConfig, kv_cache: Arc<TieredKvCache>) -> Result<Self> {
            let path = Path::new(cfg.model_path.trim());
            if !path.exists() {
                return Err(anyhow!(
                    "File model GGUF tidak ditemukan: {}",
                    path.display()
                ));
            }

            let mut backend = LlamaBackend::init().context("gagal inisialisasi LlamaBackend")?;
            if std::env::var("CMOD_INFERENCE_LLAMA_LOGS").is_err() {
                backend.void_logs();
            }

            let gpu_layers: u32 = if cfg.gpu_layers < 0 {
                999
            } else {
                cfg.gpu_layers as u32
            };
            let split_mode_enum = match cfg.split_mode.trim().to_ascii_lowercase().as_str() {
                "none" => LlamaSplitMode::None,
                "row" => LlamaSplitMode::Row,
                "tensor" => LlamaSplitMode::Tensor,
                _ => LlamaSplitMode::Layer,
            };
            let mut model_params = LlamaModelParams::default()
                .with_n_gpu_layers(gpu_layers)
                .with_split_mode(split_mode_enum)
                .with_main_gpu(cfg.main_gpu);
            if !cfg.tensor_split.is_empty() {
                model_params = model_params
                    .with_tensor_split(&cfg.tensor_split)
                    .map_err(|e| anyhow!("gagal mengatur tensor_split multi-GPU: {e:?}"))?;
            }
            let mut model_params = std::pin::pin!(model_params);
            if cfg.moe_cpu_offload {
                // Keep attention & KV-cache in GPU VRAM while offloading MoE expert FFNs to host RAM.
                model_params
                    .as_mut()
                    .add_cpu_buft_override(c"\\.ffn_(up|down|gate)_(ch|)exps");
            }
            let pool_size = cfg.parallel_slots.clamp(1, 16) as usize;
            let fingerprint =
                compute_model_fingerprint(&cfg.model_path, &cfg.kv_cache_quant, cfg.max_model_len);

            info!(
                path = %path.display(),
                gpu_layers,
                split_mode = %cfg.split_mode,
                tensor_split = ?cfg.tensor_split,
                main_gpu = cfg.main_gpu,
                moe_cpu_offload = cfg.moe_cpu_offload,
                pool_size,
                fingerprint = %fingerprint,
                accel = active_accelerator_label(),
                ctx = cfg.max_model_len,
                kv = %cfg.kv_cache_quant,
                "Memuat model GGUF langsung ke in-process native engine dengan VRAM Context Pool"
            );

            let mut model = LlamaModel::load_from_file(&backend, path, &model_params)
                .with_context(|| format!("gagal memuat model GGUF {}", path.display()))?;

            let lora_adapter = if !cfg.lora_path.trim().is_empty() {
                let lora_p = Path::new(cfg.lora_path.trim());
                if !lora_p.exists() {
                    return Err(anyhow!(
                        "File LoRA adapter GGUF tidak ditemukan: {}",
                        lora_p.display()
                    ));
                }
                let adapter = model
                    .lora_adapter_init(lora_p)
                    .map_err(|e| anyhow!("gagal memuat LoRA adapter {}: {e:?}", lora_p.display()))?;
                info!(
                    lora_path = %lora_p.display(),
                    lora_scale = cfg.lora_scale,
                    "LoRA adapter GGUF berhasil dimuat in-process"
                );
                Some((
                    std::sync::Mutex::new(adapter),
                    cfg.lora_scale,
                    cfg.lora_path.trim().to_string(),
                ))
            } else {
                None
            };

            #[cfg(feature = "grammar")]
            let tok_env = {
                info!("Membangun TokTrie llguidance untuk constrained JSON decoding in-process");
                LlamaSampler::llguidance_tok_env(&model)
            };

            let n_threads = std::thread::available_parallelism()
                .map(|n| (n.get() as i32).min(8))
                .unwrap_or(4);

            #[cfg(feature = "multimodal")]
            let mmproj_path = if cfg.mmproj_path.trim().is_empty() {
                None
            } else {
                Some(cfg.mmproj_path.trim().to_string())
            };

            let draft_model = if !cfg.draft_model_path.trim().is_empty() {
                let dm_p = Path::new(cfg.draft_model_path.trim());
                if !dm_p.exists() {
                    return Err(anyhow!(
                        "File draft model GGUF tidak ditemukan: {}",
                        dm_p.display()
                    ));
                }
                let dm_params = LlamaModelParams::default()
                    .with_n_gpu_layers(gpu_layers)
                    .with_main_gpu(cfg.main_gpu);
                let dm = LlamaModel::load_from_file(&backend, dm_p, &dm_params)
                    .with_context(|| format!("gagal memuat draft model GGUF {}", dm_p.display()))?;
                info!(
                    draft_model = %dm_p.display(),
                    draft_max = cfg.draft_max,
                    "Draft model GGUF untuk speculative decoding berhasil dimuat in-process"
                );
                Some(dm)
            } else {
                None
            };

            let inner = Arc::new(NativeEngineInner {
                backend,
                model,
                model_id: cfg.model.clone(),
                model_path: cfg.model_path.clone(),
                model_fingerprint: fingerprint,
                max_model_len: cfg.max_model_len,
                gpu_layers,
                split_mode: cfg.split_mode.clone(),
                moe_cpu_offload: cfg.moe_cpu_offload,
                kv_cache_quant: parse_kv_cache_type(&cfg.kv_cache_quant),
                n_threads,
                kv_cache,
                #[cfg(feature = "grammar")]
                tok_env,
                #[cfg(feature = "multimodal")]
                mmproj_path,
                lora_adapter,
                enable_mtp: cfg.enable_mtp,
                draft_model,
                draft_max: cfg.draft_max.clamp(1, 16) as usize,
            });

            // Spawn `pool_size` persistent worker threads sequentially so GPU backends
            // (notably Apple Metal) never initialize multiple LlamaContexts concurrently.
            let mut worker_senders = Vec::with_capacity(pool_size);
            for slot_idx in 0..pool_size {
                let (job_tx, job_rx) = std::sync::mpsc::channel::<GenerationJob>();
                let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<()>>(1);
                let w_inner = inner.clone();
                std::thread::Builder::new()
                    .name(format!("cmod-vram-slot-{slot_idx}"))
                    .spawn(move || worker_loop(slot_idx, w_inner, job_rx, ready_tx))
                    .context("gagal membuat thread worker VRAM pool")?;
                ready_rx
                    .recv()
                    .context("thread worker VRAM pool berhenti sebelum siap")??;
                worker_senders.push(job_tx);
            }

            Ok(Self {
                inner,
                worker_senders,
                rr: AtomicUsize::new(0),
            })
        }
    }

    fn worker_loop(
        slot_idx: usize,
        inner: Arc<NativeEngineInner>,
        rx: std::sync::mpsc::Receiver<GenerationJob>,
        ready_tx: std::sync::mpsc::SyncSender<Result<()>>,
    ) {
        let batch_cap_u32 = inner.max_model_len.min(2048);
        let make_target_params = |with_mtp: bool| {
            let mut p = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(inner.max_model_len))
                .with_n_batch(batch_cap_u32)
                .with_n_threads(inner.n_threads)
                .with_n_threads_batch(inner.n_threads)
                .with_type_k(inner.kv_cache_quant)
                .with_type_v(inner.kv_cache_quant);
            if with_mtp {
                p = p.with_n_rs_seq(2);
            }
            p
        };

        // Optional same-model MTP speculative context (when `--mtp` is enabled and no external draft model is set)
        let mut mtp_ctx: Option<MtpSpeculative<'_>> = None;
        let mut plain_ctx: Option<LlamaContext<'_>> = None;

        if inner.enable_mtp && inner.draft_model.is_none() {
            let target_try = inner
                .model
                .new_context(&inner.backend, make_target_params(true));
            let draft_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(inner.max_model_len))
                .with_n_batch(batch_cap_u32)
                .with_n_threads(inner.n_threads)
                .with_n_threads_batch(inner.n_threads)
                .with_context_type(LlamaContextType::Mtp);
            let draft_try = inner.model.new_context(&inner.backend, draft_params);
            match (target_try, draft_try) {
                (Ok(t_ctx), Ok(d_ctx)) => {
                    let params = MtpSpeculativeParams {
                        n_max: inner.draft_max as i32,
                        n_min: 0,
                        p_min: 0.0,
                    };
                    match MtpSpeculative::new(t_ctx, d_ctx, params) {
                        Ok(spec) => {
                            info!(
                                slot_idx,
                                draft_max = inner.draft_max,
                                "Same-model MTP speculative decoding aktif di worker slot"
                            );
                            mtp_ctx = Some(spec);
                        }
                        Err(e) => {
                            warn!(
                                slot_idx,
                                "Model GGUF ini tidak memiliki head MTP native ({e}); fallback ke decode standar"
                            );
                        }
                    }
                }
                _ => {
                    warn!(
                        slot_idx,
                        "Gagal inisialisasi context MTP; fallback ke decode standar"
                    );
                }
            }
        }

        if mtp_ctx.is_none() {
            match inner
                .model
                .new_context(&inner.backend, make_target_params(false))
            {
                Ok(c) => plain_ctx = Some(c),
                Err(err) => {
                    warn!(slot_idx, "gagal pre-alokasi LlamaContext VRAM pool: {err}");
                    let _ = ready_tx.send(Err(anyhow!(
                        "gagal pre-alokasi LlamaContext slot {slot_idx}: {err}"
                    )));
                    return;
                }
            }
        }

        if let Some((ref adapter_mtx, scale, ref path_label)) = inner.lora_adapter {
            let mut guard = adapter_mtx.lock().unwrap();
            let target_mut = if let Some(ref mut m) = mtp_ctx {
                m.target_context_mut()
            } else {
                plain_ctx.as_mut().unwrap()
            };
            if let Err(err) = target_mut.lora_adapter_set(&mut guard, scale) {
                warn!(slot_idx, lora = %path_label, "gagal memasang LoRA adapter ke slot: {err:?}");
                let _ = ready_tx.send(Err(anyhow!(
                    "gagal memasang LoRA adapter {} ke slot {slot_idx}: {err:?}",
                    path_label
                )));
                return;
            }
            info!(slot_idx, lora = %path_label, scale, "LoRA adapter aktif di worker slot");
        }

        let mut ext_draft_ctx = if let Some(ref dm) = inner.draft_model {
            let d_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(inner.max_model_len))
                .with_n_batch(batch_cap_u32)
                .with_n_threads(inner.n_threads)
                .with_n_threads_batch(inner.n_threads)
                .with_type_k(inner.kv_cache_quant)
                .with_type_v(inner.kv_cache_quant);
            match dm.new_context(&inner.backend, d_params) {
                Ok(c) => {
                    info!(slot_idx, "External draft model context siap di worker slot");
                    Some(c)
                }
                Err(e) => {
                    warn!(slot_idx, "gagal membuat draft context: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Quick Win #5: pre-load persistent MtmdContext once per worker slot (avoids reloading mmproj per image request)
        #[cfg(feature = "multimodal")]
        let mut mtmd_ctx = if let Some(ref mmproj) = inner.mmproj_path {
            use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};
            let mut mtmd_params = MtmdContextParams::default();
            mtmd_params.n_threads = inner.n_threads;
            match MtmdContext::init_from_file(mmproj, &inner.model, &mtmd_params) {
                Ok(m) => {
                    info!(slot_idx, mmproj = %mmproj, "Persistent MtmdContext siap di worker slot");
                    Some(m)
                }
                Err(e) => {
                    warn!(slot_idx, "gagal pre-load MtmdContext mmproj: {e:?}");
                    None
                }
            }
        } else {
            None
        };

        let _ = ready_tx.send(Ok(()));

        let mut l1_tokens: Vec<LlamaToken> = Vec::new();
        let mut l1_valid = false;

        while let Ok(job) = rx.recv() {
            let err_tx = job.tx.clone();
            let res = if let Some(ref mut mtp) = mtp_ctx {
                execute_on_context(
                    &inner,
                    TargetSlot::Mtp(mtp),
                    None,
                    #[cfg(feature = "multimodal")]
                    mtmd_ctx.as_mut(),
                    &mut l1_tokens,
                    &mut l1_valid,
                    job.req,
                    job.tx,
                )
            } else {
                execute_on_context(
                    &inner,
                    TargetSlot::Plain(plain_ctx.as_mut().unwrap()),
                    ext_draft_ctx.as_mut(),
                    #[cfg(feature = "multimodal")]
                    mtmd_ctx.as_mut(),
                    &mut l1_tokens,
                    &mut l1_valid,
                    job.req,
                    job.tx,
                )
            };
            if let Err(err) = res {
                l1_valid = false;
                let _ = err_tx.send(Err(err));
            }
        }
    }

    fn format_prompt(model: &LlamaModel, req: &CanonicalRequest) -> Result<String> {
        let mut chat_msgs: Vec<LlamaChatMessage> = Vec::new();
        let tmpl_res = model.chat_template(None);
        let tmpl_str = tmpl_res
            .as_ref()
            .ok()
            .and_then(|t| t.to_str().ok())
            .unwrap_or("");
        let is_gemma4_tmpl = tmpl_str.contains("<|turn>");

        let mut sys_parts: Vec<String> = req.system.iter().map(|b| b.text.clone()).collect();
        let tool_block = if is_gemma4_tmpl {
            format_gemma4_tools_block(&req.tools, req.tool_choice.as_ref())
        } else {
            format_tools_system_block(&req.tools, req.tool_choice.as_ref())
        };
        if let Some(tb) = tool_block {
            sys_parts.push(tb);
        }
        if !sys_parts.is_empty() {
            chat_msgs.push(
                LlamaChatMessage::new("system".to_string(), sys_parts.join("\n\n"))
                    .context("gagal membuat system chat message")?,
            );
        }

        for msg in &req.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            let mut text_parts = Vec::new();
            for p in &msg.parts {
                match p {
                    ContentPart::Text { text } => text_parts.push(text.clone()),
                    ContentPart::ToolResult {
                        tool_call_id,
                        content,
                        ..
                    } => text_parts.push(format!("[tool_result:{tool_call_id}] {content}")),
                    ContentPart::ToolUse {
                        id: _,
                        name,
                        arguments,
                    } => text_parts.push(format!(
                        "<tool_call>\n{{\"name\":\"{name}\",\"arguments\":{}}}\n</tool_call>",
                        serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string())
                    )),
                    ContentPart::Image { mime_type, .. } => {
                        #[cfg(feature = "multimodal")]
                        {
                            let _ = mime_type;
                            text_parts.push(llama_cpp_2::mtmd::mtmd_default_marker().to_string());
                        }
                        #[cfg(not(feature = "multimodal"))]
                        {
                            text_parts.push(format!("[image:{mime_type}]"));
                        }
                    }
                }
            }
            chat_msgs.push(
                LlamaChatMessage::new(role.to_string(), text_parts.join("\n"))
                    .context("gagal membuat chat message")?,
            );
        }

        let is_mistral_tmpl =
            tmpl_str.contains("[SYSTEM_PROMPT]") || tmpl_str.contains("[INST]");
        let is_gemma3_tmpl = !is_gemma4_tmpl && tmpl_str.contains("<start_of_turn>");
        let is_llama3_tmpl = tmpl_str.contains("<|start_header_id|>");

        let mut rendered = if let Ok(ref tmpl) = tmpl_res {
            if let Ok(r) = model.apply_chat_template(tmpl, &chat_msgs, true) {
                r
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if rendered.is_empty() {
            if is_gemma4_tmpl {
                // Native Google Gemma 4 turn format (`<|turn>role\n...<turn|>\n`)
                if !sys_parts.is_empty() {
                    rendered.push_str("<|turn>system\n");
                    rendered.push_str(&sys_parts.join("\n\n"));
                    rendered.push_str("<turn|>\n");
                }
                for msg in &req.messages {
                    let role = match msg.role {
                        Role::User => "user",
                        Role::Assistant => "model",
                        Role::Tool => "user",
                    };
                    rendered.push_str(&format!("<|turn>{role}\n"));
                    for p in &msg.parts {
                        match p {
                            ContentPart::Text { text } => rendered.push_str(text),
                            ContentPart::ToolResult {
                                tool_call_id,
                                content,
                                ..
                            } => rendered.push_str(&format!(
                                "<|tool_response>[{tool_call_id}] {content}"
                            )),
                            ContentPart::ToolUse {
                                name, arguments, ..
                            } => rendered.push_str(&format!(
                                "<tool_call>\n{{\"name\":\"{name}\",\"arguments\":{}}}\n</tool_call>",
                                serde_json::to_string(arguments)
                                    .unwrap_or_else(|_| "{}".to_string())
                            )),
                            ContentPart::Image { .. } => {
                                #[cfg(feature = "multimodal")]
                                rendered.push_str(llama_cpp_2::mtmd::mtmd_default_marker());
                            }
                        }
                    }
                    rendered.push_str("<turn|>\n");
                }
                rendered.push_str("<|turn>model\n");
                if req.sampling.non_thinking {
                    rendered.push_str("<|channel>thought\n<channel|>");
                }
            } else if is_gemma3_tmpl {
                // Native Google Gemma 2 / Gemma 3 format (`<start_of_turn>role\n...<end_of_turn>\n`)
                let mut sys_prefix = if !sys_parts.is_empty() {
                    format!("{}\n\n", sys_parts.join("\n\n"))
                } else {
                    String::new()
                };
                for msg in &req.messages {
                    let role = match msg.role {
                        Role::Assistant => "model",
                        _ => "user",
                    };
                    rendered.push_str(&format!("<start_of_turn>{role}\n"));
                    if role == "user" && !sys_prefix.is_empty() {
                        rendered.push_str(&sys_prefix);
                        sys_prefix.clear();
                    }
                    for p in &msg.parts {
                        match p {
                            ContentPart::Text { text } => rendered.push_str(text),
                            ContentPart::ToolResult {
                                tool_call_id,
                                content,
                                ..
                            } => rendered.push_str(&format!("[tool_result:{tool_call_id}] {content}")),
                            ContentPart::ToolUse {
                                name, arguments, ..
                            } => rendered.push_str(&format!(
                                "<tool_call>\n{{\"name\":\"{name}\",\"arguments\":{}}}\n</tool_call>",
                                serde_json::to_string(arguments)
                                    .unwrap_or_else(|_| "{}".to_string())
                            )),
                            ContentPart::Image { .. } => {
                                #[cfg(feature = "multimodal")]
                                rendered.push_str(llama_cpp_2::mtmd::mtmd_default_marker());
                            }
                        }
                    }
                    rendered.push_str("<end_of_turn>\n");
                }
                rendered.push_str("<start_of_turn>model\n");
            } else if is_llama3_tmpl {
                // Native Meta Llama 3 / 3.1 / 3.2 / 3.3 / 4 format
                if !sys_parts.is_empty() {
                    rendered.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    rendered.push_str(&sys_parts.join("\n\n"));
                    rendered.push_str("<|eot_id|>");
                }
                for msg in &req.messages {
                    let role = match msg.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "ipython",
                    };
                    rendered.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n"));
                    for p in &msg.parts {
                        match p {
                            ContentPart::Text { text } => rendered.push_str(text),
                            ContentPart::ToolResult {
                                tool_call_id,
                                content,
                                ..
                            } => rendered.push_str(&format!("[tool_result:{tool_call_id}] {content}")),
                            ContentPart::ToolUse {
                                name, arguments, ..
                            } => rendered.push_str(&format!(
                                "<tool_call>\n{{\"name\":\"{name}\",\"arguments\":{}}}\n</tool_call>",
                                serde_json::to_string(arguments)
                                    .unwrap_or_else(|_| "{}".to_string())
                            )),
                            _ => {}
                        }
                    }
                    rendered.push_str("<|eot_id|>");
                }
                rendered.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            } else if is_mistral_tmpl {
                // Native Mistral v3 / Ministral-3 Tekken chat format when llama_chat_apply_template
                // cannot evaluate complex Jinja2 templates.
                rendered.push_str("<s>");
                if !sys_parts.is_empty() {
                    rendered.push_str("[SYSTEM_PROMPT]");
                    rendered.push_str(&sys_parts.join("\n\n"));
                    rendered.push_str("[/SYSTEM_PROMPT]");
                }
                for msg in &req.messages {
                    match msg.role {
                        Role::User => {
                            rendered.push_str("[INST]");
                            for p in &msg.parts {
                                if let ContentPart::Text { text } = p {
                                    rendered.push_str(text);
                                }
                            }
                            rendered.push_str("[/INST]");
                        }
                        Role::Assistant => {
                            for p in &msg.parts {
                                match p {
                                    ContentPart::Text { text } => rendered.push_str(text),
                                    ContentPart::ToolUse {
                                        name, arguments, ..
                                    } => {
                                        rendered.push_str(&format!(
                                            "[TOOL_CALLS]{name}[ARGS]{}",
                                            serde_json::to_string(arguments)
                                                .unwrap_or_else(|_| "{}".to_string())
                                        ));
                                    }
                                    _ => {}
                                }
                            }
                            rendered.push_str("</s>");
                        }
                        Role::Tool => {
                            for p in &msg.parts {
                                if let ContentPart::ToolResult { content, .. } = p {
                                    rendered.push_str("[TOOL_RESULTS]");
                                    rendered.push_str(content);
                                    rendered.push_str("[/TOOL_RESULTS]");
                                }
                            }
                        }
                    }
                }
            } else {
                if !sys_parts.is_empty() {
                    rendered.push_str("<|im_start|>system\n");
                    rendered.push_str(&sys_parts.join("\n\n"));
                    rendered.push_str("<|im_end|>\n");
                }
                for msg in &req.messages {
                    let role = match msg.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    };
                    rendered.push_str(&format!("<|im_start|>{role}\n"));
                    for p in &msg.parts {
                        match p {
                            ContentPart::Text { text } => rendered.push_str(text),
                            ContentPart::ToolResult {
                                tool_call_id,
                                content,
                                ..
                            } => rendered.push_str(&format!("[tool_result:{tool_call_id}] {content}")),
                            ContentPart::ToolUse {
                                name, arguments, ..
                            } => rendered.push_str(&format!(
                                "<tool_call>\n{{\"name\":\"{name}\",\"arguments\":{}}}\n</tool_call>",
                                serde_json::to_string(arguments)
                                    .unwrap_or_else(|_| "{}".to_string())
                            )),
                            _ => {}
                        }
                    }
                    rendered.push_str("<|im_end|>\n");
                }
                rendered.push_str("<|im_start|>assistant\n");
            }
        }

        // Honor explicit non_thinking (enable_thinking=false / thinking.type=disabled)
        // by pre-closing the empty <think> block on ChatML/Qwen3 thinking templates.
        let has_forced_tool = !req.tools.is_empty()
            && matches!(
                req.tool_choice,
                Some(ToolChoice::Required | ToolChoice::Named { .. })
            );
        if req.sampling.non_thinking
            && matches!(req.response_format, ResponseFormat::Text)
            && !has_forced_tool
        {
            if rendered.ends_with("<think>\n") {
                rendered.push_str("\n</think>\n\n");
            } else if rendered.ends_with("<|channel>thought\n") {
                rendered.push_str("<channel|>");
            } else if rendered.ends_with("<|im_start|>assistant\n") {
                rendered.push_str("<think>\n\n</think>\n\n");
            }
        }

        Ok(rendered)
    }

    /// Quick Win #6: build a strict JSON Schema for forced tool calling (`required` / `named`)
    /// so `llguidance` guarantees 100% valid tool-call JSON across all model families (Gemma/Llama/Mistral/Qwen).
    fn build_forced_tool_schema(
        tools: &[ToolDefinition],
        tool_choice: Option<&ToolChoice>,
    ) -> Option<serde_json::Value> {
        if tools.is_empty() {
            return None;
        }
        let make_entry = |t: &ToolDefinition| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "const": t.name },
                    "arguments": t.input_schema,
                },
                "required": ["name", "arguments"],
                "additionalProperties": false,
            })
        };
        match tool_choice {
            Some(ToolChoice::Named { name }) => tools
                .iter()
                .find(|t| t.name == *name)
                .map(make_entry),
            Some(ToolChoice::Required) => {
                if tools.len() == 1 {
                    Some(make_entry(&tools[0]))
                } else {
                    Some(serde_json::json!({
                        "oneOf": tools.iter().map(make_entry).collect::<Vec<_>>()
                    }))
                }
            }
            _ => None,
        }
    }

    #[cfg(feature = "grammar")]
    fn build_grammar_sampler(
        tok_env: &toktrie::TokEnv,
        rf: &ResponseFormat,
        forced_tool_schema: Option<&serde_json::Value>,
    ) -> Option<LlamaSampler> {
        let schema_val = match rf {
            ResponseFormat::Text => forced_tool_schema.cloned()?,
            ResponseFormat::JsonObject => serde_json::json!({ "type": "object" }),
            ResponseFormat::JsonSchema { schema, .. } => schema.clone(),
        };

        let factory = match llguidance::ParserFactory::new_simple(tok_env) {
            Ok(f) => f,
            Err(e) => {
                warn!("llguidance ParserFactory gagal dibuat: {e}");
                return None;
            }
        };
        let grammar = llguidance::api::TopLevelGrammar::from_json_schema(schema_val);
        let parser = match factory.create_parser(grammar) {
            Ok(p) => p,
            Err(e) => {
                warn!("llguidance create_parser gagal: {e}");
                return None;
            }
        };
        let matcher = llguidance::Matcher::new(Ok(parser));
        Some(LlamaSampler::from(matcher))
    }

    fn build_sampler(
        inner: &NativeEngineInner,
        req: &CanonicalRequest,
        forced_tool_schema: Option<&serde_json::Value>,
    ) -> LlamaSampler {
        let mut chain = Vec::new();

        #[cfg(feature = "grammar")]
        if let Some(llg) =
            build_grammar_sampler(&inner.tok_env, &req.response_format, forced_tool_schema)
        {
            chain.push(llg);
        }
        #[cfg(not(feature = "grammar"))]
        let _ = (&inner.model_id, &req.response_format, forced_tool_schema);

        let temp = req.sampling.temperature.unwrap_or(0.0) as f32;
        if temp <= 0.001 {
            chain.push(LlamaSampler::greedy());
        } else {
            // Always filter long-tail garbage tokens (prevents multilingual token drift
            // when clients like Zed send temperature=1.0 without top_p/top_k).
            chain.push(LlamaSampler::top_k(40));
            chain.push(LlamaSampler::min_p(0.05, 1));
            if let Some(top_p) = req.sampling.top_p {
                chain.push(LlamaSampler::top_p(top_p as f32, 1));
            }
            chain.push(LlamaSampler::temp(temp));
            chain.push(LlamaSampler::dist(42));
        }

        LlamaSampler::chain_simple(chain)
    }

    fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    enum TargetSlot<'a, 'model> {
        Plain(&'a mut LlamaContext<'model>),
        Mtp(&'a mut MtpSpeculative<'model>),
    }

    impl<'model> TargetSlot<'_, 'model> {
        fn ctx_mut(&mut self) -> &mut LlamaContext<'model> {
            match self {
                Self::Plain(c) => c,
                Self::Mtp(m) => m.target_context_mut(),
            }
        }
    }

    fn execute_on_context(
        inner: &NativeEngineInner,
        mut slot: TargetSlot<'_, '_>,
        mut draft_ctx_opt: Option<&mut LlamaContext<'_>>,
        #[cfg(feature = "multimodal")] mut mtmd_ctx_opt: Option<&mut llama_cpp_2::mtmd::MtmdContext>,
        l1_tokens: &mut Vec<LlamaToken>,
        l1_valid: &mut bool,
        req: CanonicalRequest,
        tx: mpsc::UnboundedSender<Result<CanonicalStreamEvent>>,
    ) -> Result<()> {
        let req_started = std::time::Instant::now();
        let forced_tool_schema = if cfg!(feature = "grammar")
            && matches!(req.response_format, ResponseFormat::Text)
        {
            build_forced_tool_schema(&req.tools, req.tool_choice.as_ref())
        } else {
            None
        };
        let mut req = req;
        let mut prompt = format_prompt(&inner.model, &req)?;
        let vocab = inner.model.vocab();
        let mut add_bos = !prompt.starts_with("<s>");
        let mut prompt_tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);
        let max_ctx_usize = inner.max_model_len as usize;
        let safe_prompt_limit = max_ctx_usize.saturating_sub(2048).max(max_ctx_usize / 2);

        // Exact-tokenizer auto-compacting: if actual token count exceeds the safe budget
        // (even after the fast chars/4 pre-check), prune old tool results and slide-compact
        // oldest turns while keeping system prompt + the most recent 4 turns intact.
        if prompt_tokens.len() > safe_prompt_limit && req.messages.len() > 4 {
            let before_toks = prompt_tokens.len();
            // Stage 1: aggressively prune old ToolResult payloads outside the last 4 messages
            let cutoff = req.messages.len().saturating_sub(4);
            let mut pruned_tools = 0usize;
            for idx in 0..cutoff {
                for part in &mut req.messages[idx].parts {
                    if let ContentPart::ToolResult { content, .. } = part {
                        if content.len() > 160 {
                            let orig = content.chars().count();
                            let head: String = content.chars().take(100).collect();
                            *content = format!("{head}\n...[auto-compacted tool_result: {orig} chars]");
                            pruned_tools += 1;
                        }
                    }
                }
            }
            if pruned_tools > 0 {
                prompt = format_prompt(&inner.model, &req)?;
                add_bos = !prompt.starts_with("<s>");
                prompt_tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);
            }

            // Stage 2: sliding-window drop of oldest turns if still above safe_prompt_limit
            let mut dropped_turns = 0usize;
            while prompt_tokens.len() > safe_prompt_limit && req.messages.len() > 4 {
                req.messages.remove(0);
                dropped_turns += 1;
                if dropped_turns == 1 || dropped_turns % 2 == 0 || req.messages.len() <= 4 {
                    let mut tmp_req = req.clone();
                    tmp_req.messages.insert(
                        0,
                        crate::ir::CanonicalMessage {
                            role: Role::User,
                            parts: vec![ContentPart::Text {
                                text: format!(
                                    "[Auto-compacted {dropped_turns} earlier messages to stay within {max_ctx_usize} token context window. Continue from the recent turns below.]"
                                ),
                            }],
                        },
                    );
                    prompt = format_prompt(&inner.model, &tmp_req)?;
                    add_bos = !prompt.starts_with("<s>");
                    prompt_tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);
                }
            }
            if dropped_turns > 0 {
                req.messages.insert(
                    0,
                    crate::ir::CanonicalMessage {
                        role: Role::User,
                        parts: vec![ContentPart::Text {
                            text: format!(
                                "[Auto-compacted {dropped_turns} earlier messages to stay within {max_ctx_usize} token context window. Continue from the recent turns below.]"
                            ),
                        }],
                    },
                );
                prompt = format_prompt(&inner.model, &req)?;
                add_bos = !prompt.starts_with("<s>");
                prompt_tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);
            }
            info!(
                before_toks,
                after_toks = prompt_tokens.len(),
                pruned_tools,
                dropped_turns,
                max_ctx = max_ctx_usize,
                "Auto-compacted conversation using actual tokenizer count"
            );
        }

        let input_count = prompt_tokens.len() as u64;
        if prompt_tokens.len() + 1 >= max_ctx_usize {
            return Err(anyhow!(
                "context_length_exceeded: jumlah token aktual ({}) melampaui batas context server ({}). Mulai thread baru atau naikkan --ctx.",
                prompt_tokens.len(),
                inner.max_model_len
            ));
        }
        let available_out = max_ctx_usize.saturating_sub(prompt_tokens.len() + 1).max(1);
        let max_new = (req.sampling.max_tokens.unwrap_or(1024) as usize).min(available_out);

        let _ = tx.send(Ok(CanonicalStreamEvent::Start {
            id: req.request_id.clone(),
            model: req.model.clone(),
        }));

        let raw_images: Vec<String> = req
            .messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                ContentPart::Image { base64, .. } => Some(base64.clone()),
                _ => None,
            })
            .collect();

        let mut n_cur: i32;
        let mut last_logit_idx: i32;
        let mut cached_prefix_tokens: usize = 0;
        let batch_cap = (inner.max_model_len as usize).min(2048);
        let mut batch = LlamaBatch::new(batch_cap, 1);

        #[cfg(feature = "multimodal")]
        let mut used_mtmd = false;
        #[cfg(not(feature = "multimodal"))]
        let used_mtmd = false;

        #[cfg(feature = "multimodal")]
        if !raw_images.is_empty() {
            if let Some(mtmd_ctx) = mtmd_ctx_opt.as_deref_mut() {
                use base64::Engine;
                use llama_cpp_2::mtmd::{MtmdBitmap, MtmdInputText};
                slot.ctx_mut().clear_kv_cache();
                *l1_valid = false;
                l1_tokens.clear();

                let mut bitmaps = Vec::with_capacity(raw_images.len());
                for b64 in &raw_images {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .context("gagal decode base64 gambar untuk mtmd")?;
                    let bmp = MtmdBitmap::from_buffer(mtmd_ctx, &bytes, false)
                        .map_err(|e| anyhow!("gagal memproses bitmap gambar mtmd: {e:?}"))?;
                    bitmaps.push(bmp);
                }
                let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
                let chunks = mtmd_ctx
                    .tokenize(
                        MtmdInputText {
                            text: prompt.clone(),
                            add_special: true,
                            parse_special: true,
                        },
                        &bitmap_refs,
                    )
                    .map_err(|e| anyhow!("gagal tokenize multimodal mtmd: {e:?}"))?;

                let new_n_past = chunks
                    .eval_chunks(mtmd_ctx, slot.ctx_mut(), 0, 0, batch_cap as i32, true)
                    .map_err(|e| anyhow!("gagal evaluasi multimodal mtmd: {e:?}"))?;
                n_cur = new_n_past;
                last_logit_idx = -1;
                used_mtmd = true;
            } else {
                n_cur = 0;
                last_logit_idx = 0;
            }
        } else {
            let _ = mtmd_ctx_opt;
            n_cur = 0;
            last_logit_idx = 0;
        }
        #[cfg(not(feature = "multimodal"))]
        {
            let _ = &raw_images;
            n_cur = 0;
            last_logit_idx = 0;
        }

        let cache_key = format!("{}_{}", inner.model_fingerprint, req.prefix_hash);
        let mut should_persist_kv = false;

        if !used_mtmd {

            // Tier 1 (L1 VRAM Hot Hit): reuse prefix already resident in this worker's VRAM.
            // Note: `clear_kv_cache_seq` returns `Ok(false)` on hybrid recurrent models (e.g. Qwen3.5/MiMo DeltaNet)
            // when partial rollback isn't supported; we MUST check `Ok(true)` (or exact continuation).
            if *l1_valid && !l1_tokens.is_empty() && prompt_tokens.len() > 16 {
                let matched = common_prefix_len(l1_tokens, &prompt_tokens);
                if matched > 16 && matched < prompt_tokens.len() {
                    let can_reuse = matched == l1_tokens.len()
                        || matches!(
                            slot.ctx_mut()
                                .clear_kv_cache_seq(Some(0), Some(matched as u32), None),
                            Ok(true)
                        );
                    if can_reuse {
                        cached_prefix_tokens = matched;
                        inner.kv_cache.record_l1_hit();
                    }
                }
            }

            // Tier 2 (L2 Host RAM) & Tier 3 (L3 NVMe Disk)
            if cached_prefix_tokens == 0 {
                slot.ctx_mut().clear_kv_cache();
                if !req.prefix_hash.is_empty() && prompt_tokens.len() > 16 {
                    if let Some(snap) = inner.kv_cache.get_ram(&cache_key) {
                        let matched = common_prefix_len(&snap.tokens, &prompt_tokens);
                        if matched > 16 && matched < prompt_tokens.len() {
                            if slot.ctx_mut().state_seq_set(&snap.state, 0).is_ok()
                                && (matched == snap.tokens.len()
                                    || matches!(
                                        slot.ctx_mut().clear_kv_cache_seq(
                                            Some(0),
                                            Some(matched as u32),
                                            None
                                        ),
                                        Ok(true)
                                    ))
                            {
                                cached_prefix_tokens = matched;
                                inner.kv_cache.record_l2_hit();
                            } else {
                                slot.ctx_mut().clear_kv_cache();
                            }
                        }
                    } else if let Some(disk_path) = inner.kv_cache.disk_path_for_key(&cache_key) {
                        if disk_path.exists() {
                            if let Ok((disk_tokens, _)) = slot.ctx_mut().state_seq_load_file(
                                &disk_path,
                                0,
                                inner.max_model_len as usize,
                            ) {
                                let matched = common_prefix_len(&disk_tokens, &prompt_tokens);
                                if matched > 16
                                    && matched < prompt_tokens.len()
                                    && (matched == disk_tokens.len()
                                        || matches!(
                                            slot.ctx_mut().clear_kv_cache_seq(
                                                Some(0),
                                                Some(matched as u32),
                                                None
                                            ),
                                            Ok(true)
                                        ))
                                {
                                    cached_prefix_tokens = matched;
                                    inner.kv_cache.record_l3_hit();
                                    if let Ok(state) = slot
                                        .ctx_mut()
                                        .state_seq_get(0, LlamaStateSeqFlags::empty())
                                    {
                                        inner.kv_cache.put_ram(
                                            cache_key.clone(),
                                            NativeKvSnapshot {
                                                state,
                                                tokens: disk_tokens[..matched].to_vec(),
                                            },
                                        );
                                    }
                                } else {
                                    slot.ctx_mut().clear_kv_cache();
                                }
                            }
                        }
                    }
                    if cached_prefix_tokens == 0 {
                        inner.kv_cache.record_miss();
                    }
                }
            }

            let total_prompt = prompt_tokens.len();
            let mut prefill_start = cached_prefix_tokens;
            let mut retried_clean = false;
            loop {
                let mut prefill_ok = true;
                for chunk_start in (prefill_start..total_prompt).step_by(batch_cap) {
                    batch.clear();
                    let chunk_end = (chunk_start + batch_cap).min(total_prompt);
                    for (offset, &tok) in prompt_tokens[chunk_start..chunk_end].iter().enumerate() {
                        let pos = (chunk_start + offset) as i32;
                        let is_last = pos as usize == total_prompt - 1;
                        batch.add(tok, pos, &[0], is_last)?;
                    }
                    if let Err(err) = slot.ctx_mut().decode(&mut batch) {
                        if prefill_start > 0 && !retried_clean {
                            warn!(
                                cached = prefill_start,
                                "Prefill gagal setelah reuse KV-cache ({err}); retry clean prefill dari 0"
                            );
                            slot.ctx_mut().clear_kv_cache();
                            cached_prefix_tokens = 0;
                            prefill_start = 0;
                            retried_clean = true;
                            prefill_ok = false;
                            break;
                        }
                        return Err(err).context("gagal prefill prompt");
                    }
                    if let TargetSlot::Mtp(ref mut mtp) = slot {
                        if chunk_start == prefill_start {
                            let _ = mtp.begin(&prompt_tokens[..chunk_end]);
                        }
                        let _ = mtp.process(&batch);
                    }
                }
                if prefill_ok {
                    break;
                }
            }

            // Defer L2 RAM & L3 Disk persistence until AFTER response Done is flushed to client
            // (Quick Win #4: eliminates Run 1 disk-write latency penalty on the user-facing path).
            if cached_prefix_tokens == 0 && !req.prefix_hash.is_empty() && total_prompt > 32 {
                should_persist_kv = true;
            }

            *l1_tokens = prompt_tokens.clone();
            *l1_valid = true;
            n_cur = total_prompt as i32;
            last_logit_idx = batch.n_tokens() - 1;
        }

        let prefill_ms = req_started.elapsed().as_millis();
        let decode_started = std::time::Instant::now();
        let mut sampler = build_sampler(inner, &req, forced_tool_schema.as_ref());
        let starts_in_think =
            prompt.ends_with("<think>\n") || prompt.ends_with("<|channel>thought\n");
        let mut parser = if starts_in_think && !req.sampling.non_thinking {
            NativeOutputParser::new_in_think()
        } else {
            NativeOutputParser::new()
        };
        let mut output_count: u64 = 0;
        let mut utf8_buf: Vec<u8> = Vec::new();
        let mut accumulated = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let is_forced_tool = forced_tool_schema.is_some();
        let use_ext_draft = draft_ctx_opt.is_some()
            && !is_forced_tool
            && matches!(req.response_format, ResponseFormat::Text)
            && raw_images.is_empty();

        // Prefill external draft model context when speculative decoding is active
        let mut draft_n_cur: i32 = 0;
        let mut draft_batch = LlamaBatch::new(batch_cap, 1);
        let mut draft_sampler = LlamaSampler::greedy();
        let mut spec_drafted: u64 = 0;
        let mut spec_accepted: u64 = 0;

        if use_ext_draft {
            if let Some(d_ctx) = draft_ctx_opt.as_deref_mut() {
                d_ctx.clear_kv_cache();
                let total_prompt = prompt_tokens.len();
                for chunk_start in (0..total_prompt).step_by(batch_cap) {
                    draft_batch.clear();
                    let chunk_end = (chunk_start + batch_cap).min(total_prompt);
                    for (offset, &tok) in prompt_tokens[chunk_start..chunk_end].iter().enumerate() {
                        let pos = (chunk_start + offset) as i32;
                        let is_last = pos as usize == total_prompt - 1;
                        draft_batch.add(tok, pos, &[0], is_last)?;
                    }
                    d_ctx
                        .decode(&mut draft_batch)
                        .context("gagal prefill prompt draft model")?;
                }
                draft_n_cur = total_prompt as i32;
            }
        }

        let emit_token = |token: LlamaToken,
                              utf8_buf: &mut Vec<u8>,
                              accumulated: &mut String,
                              parser: &mut NativeOutputParser,
                              output_count: &mut u64,
                              stop_reason: &mut StopReason|
         -> Option<bool> {
            let piece = vocab.token_to_piece(token, true, None);
            utf8_buf.extend_from_slice(&piece);
            *output_count += 1;

            let valid_len = match std::str::from_utf8(utf8_buf) {
                Ok(_) => utf8_buf.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_len > 0 {
                let text = String::from_utf8_lossy(&utf8_buf[..valid_len]).to_string();
                utf8_buf.drain(..valid_len);
                accumulated.push_str(&text);

                if req
                    .sampling
                    .stop
                    .iter()
                    .any(|s| !s.is_empty() && accumulated.ends_with(s))
                {
                    *stop_reason = StopReason::StopSequence;
                    return Some(true);
                }

                if !is_forced_tool {
                    for ev in parser.push(&text) {
                        if tx.send(Ok(ev)).is_err() {
                            return None;
                        }
                    }
                }
            }
            if *output_count as usize >= max_new {
                *stop_reason = StopReason::MaxTokens;
                return Some(true);
            }
            Some(false)
        };

        let use_mtp_spec = matches!(slot, TargetSlot::Mtp(_))
            && !is_forced_tool
            && matches!(req.response_format, ResponseFormat::Text)
            && raw_images.is_empty();

        if use_mtp_spec {
            if let TargetSlot::Mtp(ref mut mtp) = slot {
                let mut hist_tokens = prompt_tokens.clone();
                let mut cur_token = sampler.sample(mtp.target_context(), last_logit_idx);
                while !vocab.is_eog(cur_token) {
                    if *l1_valid {
                        l1_tokens.push(cur_token);
                    }
                    hist_tokens.push(cur_token);
                    match emit_token(
                        cur_token,
                        &mut utf8_buf,
                        &mut accumulated,
                        &mut parser,
                        &mut output_count,
                        &mut stop_reason,
                    ) {
                        None => return Ok(()),
                        Some(true) => break,
                        Some(false) => {}
                    }

                    let drafted = mtp
                        .draft(n_cur, cur_token, &hist_tokens)
                        .unwrap_or_default();
                    spec_drafted += drafted.len() as u64;

                    batch.clear();
                    let target_base_pos = n_cur;
                    batch.add(cur_token, n_cur, &[0], true)?;
                    n_cur += 1;
                    for &dt in &drafted {
                        if n_cur as usize >= max_ctx_usize {
                            break;
                        }
                        batch.add(dt, n_cur, &[0], true)?;
                        n_cur += 1;
                    }
                    mtp.target_context_mut()
                        .decode(&mut batch)
                        .context("gagal verifikasi batch MTP")?;
                    let _ = mtp.process(&batch);

                    let mut n_accepted = 0usize;
                    let mut next_token = sampler.sample(mtp.target_context(), 0);
                    let mut hit_stop = false;

                    for (i, &dt) in drafted.iter().enumerate() {
                        if (i + 1) >= batch.n_tokens() as usize {
                            break;
                        }
                        if next_token == dt {
                            n_accepted += 1;
                            spec_accepted += 1;
                            if *l1_valid {
                                l1_tokens.push(dt);
                            }
                            hist_tokens.push(dt);
                            match emit_token(
                                dt,
                                &mut utf8_buf,
                                &mut accumulated,
                                &mut parser,
                                &mut output_count,
                                &mut stop_reason,
                            ) {
                                None => return Ok(()),
                                Some(true) => {
                                    hit_stop = true;
                                    break;
                                }
                                Some(false) => {}
                            }
                            next_token = sampler.sample(mtp.target_context(), (i + 1) as i32);
                        } else {
                            break;
                        }
                    }

                    let _ = mtp.accept(n_accepted as u16);
                    let kept_end = target_base_pos + 1 + n_accepted as i32;
                    if kept_end < n_cur {
                        if !matches!(
                            mtp.target_context_mut()
                                .clear_kv_cache_seq(Some(0), Some(kept_end as u32), None),
                            Ok(true)
                        ) {
                            *l1_valid = false;
                        }
                        n_cur = kept_end;
                    }

                    if hit_stop || n_cur as usize >= max_ctx_usize {
                        break;
                    }
                    cur_token = next_token;
                }
                if vocab.is_eog(cur_token) {
                    stop_reason = StopReason::EndTurn;
                }
            }
        } else if use_ext_draft {
            let ctx = slot.ctx_mut();
            let d_ctx = draft_ctx_opt.as_deref_mut().unwrap();
            let draft_k = inner.draft_max.min(16);

            // First token comes directly from target model
            let mut cur_token = sampler.sample(ctx, last_logit_idx);
            while !vocab.is_eog(cur_token) {
                if *l1_valid {
                    l1_tokens.push(cur_token);
                }
                match emit_token(
                    cur_token,
                    &mut utf8_buf,
                    &mut accumulated,
                    &mut parser,
                    &mut output_count,
                    &mut stop_reason,
                ) {
                    None => return Ok(()),
                    Some(true) => break,
                    Some(false) => {}
                }

                // 1. Propose up to `draft_k` speculative tokens using the fast draft model
                let mut drafted: Vec<LlamaToken> = Vec::with_capacity(draft_k);
                let mut d_tok = cur_token;
                let draft_base_pos = draft_n_cur;
                for _ in 0..draft_k {
                    if draft_n_cur as usize >= max_ctx_usize {
                        break;
                    }
                    draft_batch.clear();
                    draft_batch.add(d_tok, draft_n_cur, &[0], true)?;
                    draft_n_cur += 1;
                    d_ctx.decode(&mut draft_batch).context("gagal decode draft")?;
                    let proposed = draft_sampler.sample(d_ctx, draft_batch.n_tokens() - 1);
                    if vocab.is_eog(proposed) {
                        break;
                    }
                    drafted.push(proposed);
                    d_tok = proposed;
                }
                spec_drafted += drafted.len() as u64;

                // 2. Verify `[cur_token, drafted...]` in ONE parallel forward pass on the target model
                batch.clear();
                let target_base_pos = n_cur;
                batch.add(cur_token, n_cur, &[0], true)?;
                n_cur += 1;
                for &dt in &drafted {
                    if n_cur as usize >= max_ctx_usize {
                        break;
                    }
                    batch.add(dt, n_cur, &[0], true)?;
                    n_cur += 1;
                }
                ctx.decode(&mut batch)
                    .context("gagal verifikasi batch speculative")?;

                // 3. Accept matching prefix + 1 bonus token from target model
                let mut n_accepted = 0usize;
                let mut next_token = sampler.sample(ctx, 0);
                let mut hit_stop = false;

                for (i, &dt) in drafted.iter().enumerate() {
                    if (i + 1) >= batch.n_tokens() as usize {
                        break;
                    }
                    if next_token == dt {
                        n_accepted += 1;
                        spec_accepted += 1;
                        if *l1_valid {
                            l1_tokens.push(dt);
                        }
                        match emit_token(
                            dt,
                            &mut utf8_buf,
                            &mut accumulated,
                            &mut parser,
                            &mut output_count,
                            &mut stop_reason,
                        ) {
                            None => return Ok(()),
                            Some(true) => {
                                hit_stop = true;
                                break;
                            }
                            Some(false) => {}
                        }
                        next_token = sampler.sample(ctx, (i + 1) as i32);
                    } else {
                        break;
                    }
                }

                // Roll back rejected tail in both target & draft KV caches
                let kept_target_end = target_base_pos + 1 + n_accepted as i32;
                if kept_target_end < n_cur {
                    if !matches!(
                        ctx.clear_kv_cache_seq(Some(0), Some(kept_target_end as u32), None),
                        Ok(true)
                    ) {
                        *l1_valid = false;
                        break;
                    }
                    n_cur = kept_target_end;
                }
                let kept_draft_end = draft_base_pos + 1 + n_accepted as i32;
                if kept_draft_end < draft_n_cur {
                    let _ = d_ctx.clear_kv_cache_seq(Some(0), Some(kept_draft_end as u32), None);
                    draft_n_cur = kept_draft_end;
                }

                if hit_stop || n_cur as usize >= max_ctx_usize {
                    break;
                }
                cur_token = next_token;
            }
            if vocab.is_eog(cur_token) {
                stop_reason = StopReason::EndTurn;
            }
        } else {
            let ctx = slot.ctx_mut();
            for _ in 0..max_new {
                // Note: LlamaSampler::sample() already calls accept(token) internally!
                // Calling sampler.accept(token) again would double-feed stateful samplers like llguidance.
                let token = sampler.sample(ctx, last_logit_idx);

                if vocab.is_eog(token) {
                    stop_reason = StopReason::EndTurn;
                    break;
                }

                if *l1_valid {
                    l1_tokens.push(token);
                }

                match emit_token(
                    token,
                    &mut utf8_buf,
                    &mut accumulated,
                    &mut parser,
                    &mut output_count,
                    &mut stop_reason,
                ) {
                    None => return Ok(()),
                    Some(true) => break,
                    Some(false) => {}
                }

                if n_cur as usize >= max_ctx_usize {
                    stop_reason = StopReason::MaxTokens;
                    break;
                }

                batch.clear();
                batch.add(token, n_cur, &[0], true)?;
                n_cur += 1;
                ctx.decode(&mut batch).context("gagal decode token")?;
                last_logit_idx = batch.n_tokens() - 1;
            }
        }

        if is_forced_tool {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(accumulated.trim()) {
                let name = val
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let args = val
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                if !name.is_empty() {
                    let call_id = format!("call_{}", uuid::Uuid::new_v4().simple());
                    let _ = tx.send(Ok(CanonicalStreamEvent::ToolCallStart {
                        index: 0,
                        id: call_id,
                        name,
                    }));
                    let _ = tx.send(Ok(CanonicalStreamEvent::ToolCallDelta {
                        index: 0,
                        arguments_delta: serde_json::to_string(&args)
                            .unwrap_or_else(|_| "{}".to_string()),
                    }));
                    stop_reason = StopReason::ToolUse;
                }
            } else if !accumulated.is_empty() {
                let _ = tx.send(Ok(CanonicalStreamEvent::TextDelta(accumulated)));
            }
        } else {
            for ev in parser.finish() {
                let _ = tx.send(Ok(ev));
            }
            if parser.had_tool_calls() {
                stop_reason = StopReason::ToolUse;
            } else if !req.tools.is_empty() {
                let allowed: Vec<&str> = req.tools.iter().map(|t| t.name.as_str()).collect();
                let recovered = crate::backend::parser::extract_bare_tool_calls(
                    &accumulated,
                    &allowed,
                );
                if !recovered.is_empty() {
                    for (idx, (name, args_str)) in recovered.into_iter().enumerate() {
                        let call_id = format!("call_native_{}", idx + 1);
                        let _ = tx.send(Ok(CanonicalStreamEvent::ToolCallStart {
                            index: idx,
                            id: call_id,
                            name,
                        }));
                        let _ = tx.send(Ok(CanonicalStreamEvent::ToolCallDelta {
                            index: idx,
                            arguments_delta: args_str,
                        }));
                    }
                    stop_reason = StopReason::ToolUse;
                }
            }
        }

        let decode_secs = decode_started.elapsed().as_secs_f64().max(0.001);
        let decode_tps = output_count as f64 / decode_secs;
        let spec_acc_pct = if spec_drafted > 0 {
            format!("{:.0}%", (spec_accepted as f64 / spec_drafted as f64) * 100.0)
        } else {
            "off".to_string()
        };
        info!(
            model = %inner.model_id,
            in_tok = input_count,
            cached_tok = cached_prefix_tokens,
            out_tok = output_count,
            prefill_ms,
            decode_tps = format!("{decode_tps:.1}"),
            spec_acc = %spec_acc_pct,
            "Request selesai"
        );

        let _ = tx.send(Ok(CanonicalStreamEvent::Usage(UsageStats {
            input_tokens: input_count,
            output_tokens: output_count,
            cached_input_tokens: cached_prefix_tokens as u64,
        })));
        let _ = tx.send(Ok(CanonicalStreamEvent::Done { stop_reason }));
        // Drop sender NOW so the HTTP response finishes immediately before L2/L3 KV-Cache flush!
        drop(tx);

        // Post-response L2 Host RAM & L3 NVMe Disk KV-Cache persistence (off the client critical path).
        // Only trim & save when `clear_kv_cache_seq` returns `Ok(true)` (pure attention KV-cache);
        // on hybrid recurrent models (DeltaNet/Mamba), save the full sequence without partial trim.
        if should_persist_kv && *l1_valid {
            let ctx = slot.ctx_mut();
            let prompt_len = prompt_tokens.len();
            let trimmed = matches!(
                ctx.clear_kv_cache_seq(Some(0), Some(prompt_len as u32), None),
                Ok(true)
            );
            let saved_tokens = if trimmed {
                l1_tokens.truncate(prompt_len);
                prompt_tokens.clone()
            } else {
                l1_tokens.clone()
            };
            if let Ok(state) = ctx.state_seq_get(0, LlamaStateSeqFlags::empty()) {
                inner.kv_cache.put_ram(
                    cache_key.clone(),
                    NativeKvSnapshot {
                        state,
                        tokens: saved_tokens.clone(),
                    },
                );
            }
            if let Some(disk_path) = inner.kv_cache.disk_path_for_key(&cache_key) {
                if let Err(err) = ctx.state_seq_save_file(&disk_path, 0, &saved_tokens) {
                    warn!("gagal menyimpan L3 KV-cache disk: {err}");
                } else {
                    inner.kv_cache.enforce_disk_budget();
                }
            }
        }
        Ok(())
    }

    impl InferenceBackend for NativeLlamaBackend {
        fn kind_name(&self) -> &'static str {
            active_accelerator_label()
        }

        fn default_model(&self) -> &str {
            &self.inner.model_id
        }

        fn chat_complete(
            &self,
            req: CanonicalRequest,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<CanonicalResponse>> + Send + '_>> {
            Box::pin(async move {
                let mut stream = self.chat_stream(req.clone()).await?;
                use futures::StreamExt;
                let mut text = String::new();
                let mut thinking = String::new();
                let mut usage = UsageStats::default();
                let mut stop_reason = StopReason::EndTurn;
                let mut id = req.request_id.clone();
                let mut model = req.model.clone();
                let mut pending_tools: Vec<(String, String, String)> = Vec::new();

                while let Some(ev) = stream.next().await {
                    match ev? {
                        CanonicalStreamEvent::Start { id: sid, model: sm } => {
                            id = sid;
                            model = sm;
                        }
                        CanonicalStreamEvent::ThinkingDelta(d) => thinking.push_str(&d),
                        CanonicalStreamEvent::TextDelta(d) => text.push_str(&d),
                        CanonicalStreamEvent::ToolCallStart {
                            index,
                            id: tid,
                            name,
                        } => {
                            while pending_tools.len() <= index {
                                pending_tools.push((String::new(), String::new(), String::new()));
                            }
                            pending_tools[index].0 = tid;
                            pending_tools[index].1 = name;
                        }
                        CanonicalStreamEvent::ToolCallDelta {
                            index,
                            arguments_delta,
                        } => {
                            if let Some(entry) = pending_tools.get_mut(index) {
                                entry.2.push_str(&arguments_delta);
                            }
                        }
                        CanonicalStreamEvent::Usage(u) => usage = u,
                        CanonicalStreamEvent::Done { stop_reason: sr } => stop_reason = sr,
                    }
                }

                let tool_calls: Vec<CanonicalToolCall> = pending_tools
                    .into_iter()
                    .filter(|(_, name, _)| !name.is_empty())
                    .map(|(tid, name, args)| {
                        let arguments = serde_json::from_str(&args)
                            .unwrap_or_else(|_| serde_json::json!({ "raw": args }));
                        CanonicalToolCall {
                            id: tid,
                            name,
                            arguments,
                        }
                    })
                    .collect();

                Ok(CanonicalResponse {
                    id,
                    model,
                    text,
                    thinking: if thinking.is_empty() {
                        None
                    } else {
                        Some(thinking)
                    },
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
            let slot = req
                .assigned_slot
                .map(|s| s as usize % self.worker_senders.len())
                .unwrap_or_else(|| self.rr.fetch_add(1, Ordering::Relaxed) % self.worker_senders.len());
            let sender = self.worker_senders[slot].clone();

            Box::pin(async move {
                let (tx, mut rx) = mpsc::unbounded_channel();
                sender
                    .send(GenerationJob { req, tx })
                    .map_err(|_| anyhow!("VRAM worker slot {slot} tidak aktif"))?;

                let stream = async_stream::stream! {
                    while let Some(item) = rx.recv().await {
                        yield item;
                    }
                };
                Ok(Box::pin(stream) as EventStream)
            })
        }

        fn health(
            &self,
        ) -> Pin<Box<dyn std::future::Future<Output = BackendHealth> + Send + '_>> {
            let inner = self.inner.clone();
            let slots = self.worker_senders.len();
            Box::pin(async move {
                #[cfg(feature = "multimodal")]
                let mm = inner.mmproj_path.is_some();
                #[cfg(not(feature = "multimodal"))]
                let mm = false;

                let spec_label = if inner.draft_model.is_some() {
                    format!("draft(max={})", inner.draft_max)
                } else if inner.enable_mtp {
                    format!("mtp(max={})", inner.draft_max)
                } else {
                    "off".to_string()
                };

                BackendHealth {
                    ready: true,
                    kind: active_accelerator_label().to_string(),
                    upstream_url: format!("in-process://{}", inner.model_path),
                    model: inner.model_id.clone(),
                    detail: Some(format!(
                        "vram_slots={}, ctx={}, gpu_layers={}, split={}, moe_offload={}, fp={}, grammar={}, multimodal={}, lora={}, spec={}",
                        slots,
                        inner.max_model_len,
                        inner.gpu_layers,
                        inner.split_mode,
                        inner.moe_cpu_offload,
                        inner.model_fingerprint,
                        cfg!(feature = "grammar"),
                        mm,
                        inner.lora_adapter.is_some(),
                        spec_label
                    )),
                }
            })
        }

        fn embeddings(
            &self,
            model: String,
            inputs: Vec<String>,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<EmbeddingResponse>> + Send + '_>> {
            let inner = self.inner.clone();
            let resp_model = if model.trim().is_empty() {
                inner.model_id.clone()
            } else {
                model
            };
            Box::pin(async move {
                tokio::task::spawn_blocking(move || {
                    let max_ctx = inner.max_model_len.min(4096);
                    let ctx_params = LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(max_ctx))
                        .with_n_batch(512)
                        .with_n_threads(inner.n_threads)
                        .with_n_threads_batch(inner.n_threads)
                        .with_embeddings(true)
                        .with_pooling_type(LlamaPoolingType::Mean);

                    let mut ctx = inner
                        .model
                        .new_context(&inner.backend, ctx_params)
                        .context("gagal membuat LlamaContext embeddings")?;
                    let vocab = inner.model.vocab();
                    let mut batch = LlamaBatch::new(512, 1);
                    let mut embeddings = Vec::with_capacity(inputs.len());
                    let mut total_tokens: u64 = 0;

                    for text in &inputs {
                        ctx.clear_kv_cache();
                        let mut tokens = vocab.tokenize(text.as_bytes(), true, true);
                        if tokens.is_empty() {
                            tokens = vocab.tokenize(b" ", true, true);
                        }
                        if tokens.len() > max_ctx as usize {
                            tokens.truncate(max_ctx as usize);
                        }
                        total_tokens += tokens.len() as u64;

                        for chunk_start in (0..tokens.len()).step_by(512) {
                            batch.clear();
                            let chunk_end = (chunk_start + 512).min(tokens.len());
                            for (offset, &tok) in tokens[chunk_start..chunk_end].iter().enumerate() {
                                let pos = (chunk_start + offset) as i32;
                                let is_last = pos as usize == tokens.len() - 1;
                                batch.add(tok, pos, &[0], is_last)?;
                            }
                            ctx.decode(&mut batch)
                                .context("gagal menghitung embeddings")?;
                        }

                        let raw_vec: Vec<f32> = match ctx.embeddings_seq_ith(0) {
                            Ok(slice) => slice.to_vec(),
                            Err(_) => ctx
                                .embeddings_ith(batch.n_tokens() - 1)
                                .map(|s| s.to_vec())
                                .map_err(|e| anyhow!("gagal mengambil vektor embeddings: {e:?}"))?,
                        };
                        // L2 normalize for cosine similarity compatibility
                        let norm = raw_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let normalized = if norm > 1e-12 {
                            raw_vec.into_iter().map(|x| x / norm).collect()
                        } else {
                            raw_vec
                        };
                        embeddings.push(normalized);
                    }

                    Ok(EmbeddingResponse {
                        model: resp_model,
                        embeddings,
                        prompt_tokens: total_tokens,
                    })
                })
                .await
                .context("task embeddings panic")?
            })
        }
    }
}

#[cfg(feature = "native")]
pub use imp::NativeLlamaBackend;
