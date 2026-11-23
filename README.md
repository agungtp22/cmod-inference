# cmod-inference

Server inferensi lokal **mandiri** (repo & proses terpisah dari `cmod` Core) yang dirancang untuk beban kerja **Local Document Intelligence** pada jaringan internal (*on-premise / LAN / VPC*).

## Fitur Utama

- **Dual Protocol Native (OpenAI + Anthropic):**
  - OpenAI: `POST /v1/chat/completions` (streaming SSE & non-streaming, `tools`, `response_format` / `json_schema`, multimodal `image_url` base64), `GET /v1/models`, `GET /v1/models/{id}`.
  - Anthropic: `POST /v1/messages` (streaming SSE penuh `message_start` → `content_block_*` → `message_delta` → `message_stop`, `x-api-key`, `system` + `cache_control`, `tool_use` / `tool_result`, multimodal `image` base64), `POST /v1/messages/count_tokens`.
- **Multi-GPU & Cross-Vendor:**
  - Mendukung backend `vLLM` (**NVIDIA CUDA** & **AMD ROCm**) serta `llama-server` (**Apple Metal**, **Vulkan** lintas Intel/AMD/NVIDIA, **CUDA**, **ROCm**, dan **CPU fallback**).
  - Deteksi hardware otomatis (`GET /cmod/v1/hardware`) dan dukungan model **Multimodal (VLM)**, **MoE (*Mixture-of-Experts*)**, serta **Kuantisasi (GGUF Q4–Q8, FP8, AWQ/GPTQ + KV-Cache Q8/FP8)**.
- **Optimalisasi Khusus Dokumen `cmod`:**
  - *Dual-lane priority scheduler* (`X-Cmod-Priority: interactive | batch`) agar chat analis tidak macet oleh *batch workflow*.
  - *Prefix / KV-cache slot affinity* untuk system prompt & katalog berulang.
  - *Context window guard* eksplisit (tidak memotong halaman dokumen diam-diam).
  - *SSRF guard*: menolak URL gambar eksternal, hanya mengizinkan payload base64 lokal.
  - *Zero prompt logging* secara default (`log_prompts = false`) untuk kepatuhan UU PDP & OJK.

## Menjalankan Cepat

```bash
# 1. Cek deteksi hardware & rekomendasi backend GPU
cargo run -- hardware

# 2. Mode Level 2 (Native In-Process Engine — tanpa Ollama / tanpa llama-server / tanpa vLLM)
#    Pilih feature sesuai GPU target: metal | cuda | rocm | vulkan | native (CPU)
CMOD_INFERENCE_BACKEND=native \
CMOD_INFERENCE_MODEL_PATH=/path/ke/model-q4_k_m.gguf \
CMOD_INFERENCE_MODEL=cmod-doc-extract \
cargo run --release --features metal -- serve

# 3. Mode Opsional Passthrough (bila klien sudah punya vLLM / Ollama eksternal)
CMOD_INFERENCE_BACKEND=passthrough \
CMOD_INFERENCE_UPSTREAM_URL=http://127.0.0.1:8080/v1 \
CMOD_INFERENCE_MODEL=qwen2.5-14b-instruct \
cargo run -- serve
```

## Deployment Multi-GPU (Docker Compose)

- NVIDIA CUDA (`vLLM`): `docker compose -f deploy/docker-compose.cuda.yml up -d`
- AMD ROCm (`vLLM`): `docker compose -f deploy/docker-compose.rocm.yml up -d`
- Intel / AMD / Mixed GPU (`llama-server` Vulkan): `docker compose -f deploy/docker-compose.vulkan.yml up -d`
- CPU Fallback (`llama-server` CPU): `docker compose -f deploy/docker-compose.cpu.yml up -d`
