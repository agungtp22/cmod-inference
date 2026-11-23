use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::hardware::HardwareReport;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub family: String,
    pub architecture: String, // "dense" | "moe" | "vlm"
    pub total_params_b: f32,
    pub active_params_b: f32,
    pub quant: String,
    pub multimodal: bool,
    pub non_thinking: bool,
    pub disk_mb: u64,
    pub min_vram_mb: u64,
    pub recommended_vram_mb: u64,
    pub min_ram_for_offload_mb: Option<u64>,
    pub filename: String,
    pub download_url: String,
    pub mmproj_filename: Option<String>,
    pub mmproj_url: Option<String>,
    pub best_for: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapabilities {
    pub vision: bool,
    /// "none" | "hybrid" | "always"
    pub thinking: String,
    pub ocr: bool,
    pub coding: bool,
    pub moe: bool,
    pub tools: bool,
    pub json_schema: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecommendedModel {
    pub model: CatalogModel,
    pub capabilities: ModelCapabilities,
    pub tags: Vec<String>,
    /// "recommended" | "compatible" | "conditional" | "incompatible"
    pub status: String,
    pub reason: String,
    pub installed: bool,
    pub local_path: Option<String>,
    pub mmproj_local_path: Option<String>,
}

pub fn derive_capabilities(m: &CatalogModel) -> (ModelCapabilities, Vec<String>) {
    let thinking = if !m.non_thinking {
        "always"
    } else if matches!(
        m.id.as_str(),
        "qwen3-0.6b-tiny"
            | "qwen3-1.7b-mini"
            | "qwen3-8b-doc"
            | "qwen3-14b-doc"
            | "qwen3-32b-doc"
            | "smollm3-3b"
    ) {
        "hybrid"
    } else {
        "none"
    };

    let vision = m.multimodal;
    let ocr = m.multimodal;
    let coding = m.id.contains("coder") || m.id.contains("devstral");
    let moe = m.architecture == "moe";
    let tools = true;
    let json_schema = true;

    let mut tags = Vec::new();
    if vision {
        tags.push("Multimodal/Vision".to_string());
        tags.push("OCR".to_string());
    } else {
        tags.push("Text".to_string());
    }
    match thinking {
        "always" => tags.push("Thinking".to_string()),
        "hybrid" => tags.push("Hybrid-Think (/no_think)".to_string()),
        _ => tags.push("Non-Thinking".to_string()),
    }
    if coding {
        tags.push("Coding/Agent".to_string());
    }
    if moe {
        tags.push(format!("MoE ({:.1}B aktif)", m.active_params_b));
    } else {
        tags.push("Dense".to_string());
    }
    tags.push("Tools".to_string());
    tags.push("JSON".to_string());

    (
        ModelCapabilities {
            vision,
            thinking: thinking.to_string(),
            ocr,
            coding,
            moe,
            tools,
            json_schema,
        },
        tags,
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalModelFile {
    pub filename: String,
    pub path: String,
    pub size_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub model_id: String,
    /// "downloading" | "completed" | "failed"
    pub status: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub percent: Option<f32>,
    pub local_path: Option<String>,
    pub mmproj_path: Option<String>,
    pub error: Option<String>,
}

pub fn curated_catalog() -> Vec<CatalogModel> {
    vec![
        CatalogModel {
            id: "qwen3-0.6b-tiny".to_string(),
            name: "Qwen3 0.6B (Q4_K_M)".to_string(),
            family: "Qwen3".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 0.6,
            active_params_b: 0.6,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 378,
            min_vram_mb: 1200,
            recommended_vram_mb: 2000,
            min_ram_for_offload_mb: None,
            filename: "Qwen3-0.6B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Ultra-ringan (378 MB) untuk uji coba cepat, CI/smoke test, & CPU-only".to_string(),
        },
        CatalogModel {
            id: "qwen3.5-0.8b-vlm".to_string(),
            name: "Qwen3.5 0.8B Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3.5".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 0.8,
            active_params_b: 0.8,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: false,
            disk_mb: 703,
            min_vram_mb: 2000,
            recommended_vram_mb: 3500,
            min_ram_for_offload_mb: None,
            filename: "Qwen3.5-0.8B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3.5-0.8B-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "VLM multimodal paling ringan (~703 MB) untuk uji coba gambar/scan lokal".to_string(),
        },
        CatalogModel {
            id: "qwen3-1.7b-mini".to_string(),
            name: "Qwen3 1.7B (Q4_K_M)".to_string(),
            family: "Qwen3".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 1.7,
            active_params_b: 1.7,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 1110,
            min_vram_mb: 2200,
            recommended_vram_mb: 3500,
            min_ram_for_offload_mb: None,
            filename: "Qwen3-1.7B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Model mini 1.1 GB sangat cepat untuk laptop RAM 8 GB, routing, & ekstraksi singkat".to_string(),
        },
        CatalogModel {
            id: "qwen3.5-2b-vlm".to_string(),
            name: "Qwen3.5 2B Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3.5".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 2.0,
            active_params_b: 2.0,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: false,
            disk_mb: 1920,
            min_vram_mb: 3500,
            recommended_vram_mb: 5000,
            min_ram_for_offload_mb: None,
            filename: "Qwen3.5-2B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3.5-2B-mmproj-BF16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-BF16.gguf".to_string()),
            best_for: "VLM multimodal 2B ringan untuk OCR struk & formulir di GPU 4-6 GB".to_string(),
        },
        CatalogModel {
            id: "smollm3-3b".to_string(),
            name: "SmolLM3 3B (Q4_K_M)".to_string(),
            family: "SmolLM3".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 3.0,
            active_params_b: 3.0,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 1920,
            min_vram_mb: 3500,
            recommended_vram_mb: 5000,
            min_ram_for_offload_mb: None,
            filename: "SmolLM3-3B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Model open-source HuggingFace 3B dual-mode (no_think/think) untuk edge & laptop tipis".to_string(),
        },
        CatalogModel {
            id: "phi-4-mini-3.8b".to_string(),
            name: "Phi-4 Mini Instruct 3.8B (Q4_K_M)".to_string(),
            family: "Phi-4".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 3.8,
            active_params_b: 3.8,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 2490,
            min_vram_mb: 4000,
            recommended_vram_mb: 6000,
            min_ram_for_offload_mb: None,
            filename: "Phi-4-mini-instruct-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Microsoft Phi-4 Mini 3.8B kuat di logika, math, & function calling ringan".to_string(),
        },
        CatalogModel {
            id: "gemma-4-e2b-vlm".to_string(),
            name: "Gemma 4 E2B Instruct Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Gemma 4".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 5.1,
            active_params_b: 2.3,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 4050,
            min_vram_mb: 6000,
            recommended_vram_mb: 8500,
            min_ram_for_offload_mb: None,
            filename: "gemma-4-E2B-it-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("gemma-4-E2B-it-mmproj-BF16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-BF16.gguf".to_string()),
            best_for: "Gemma 4 multimodal paling ringan (2.3B efektif) untuk laptop 8-16 GB & OCR cepat".to_string(),
        },
        CatalogModel {
            id: "qwen3-vl-4b-doc".to_string(),
            name: "Qwen3-VL 4B Instruct Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3-VL".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 4.4,
            active_params_b: 4.4,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 3300,
            min_vram_mb: 5500,
            recommended_vram_mb: 8000,
            min_ram_for_offload_mb: None,
            filename: "Qwen3-VL-4B-Instruct-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3-VL-4B-Instruct-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3-VL-4B-mmproj-BF16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-BF16.gguf".to_string()),
            best_for: "Spesialis OCR dokumen & tabel ringkas untuk GPU 8 GB / Mac 16 GB".to_string(),
        },
        CatalogModel {
            id: "qwen3-4b-doc".to_string(),
            name: "Qwen3 4B Instruct 2507 (Q4_K_M)".to_string(),
            family: "Qwen3".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 4.0,
            active_params_b: 4.0,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 2382,
            min_vram_mb: 4000,
            recommended_vram_mb: 6000,
            min_ram_for_offload_mb: None,
            filename: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Pilot cepat non-thinking, laptop 8-16 GB RAM, & ekstraksi JSON ringan".to_string(),
        },
        CatalogModel {
            id: "qwen3.5-4b-vlm".to_string(),
            name: "Qwen3.5 4B Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3.5".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 4.2,
            active_params_b: 4.2,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: false,
            disk_mb: 3255,
            min_vram_mb: 5500,
            recommended_vram_mb: 8000,
            min_ram_for_offload_mb: None,
            filename: "Qwen3.5-4B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3.5-4B-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "VLM multimodal ringkas untuk scan formulir & foto dokumen di GPU 8 GB".to_string(),
        },
        CatalogModel {
            id: "gemma-4-e4b-vlm".to_string(),
            name: "Gemma 4 E4B Instruct Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Gemma 4".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 8.0,
            active_params_b: 4.5,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 5691,
            min_vram_mb: 8500,
            recommended_vram_mb: 12000,
            min_ram_for_offload_mb: None,
            filename: "gemma-4-E4B-it-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("gemma-4-E4B-it-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "Gemma 4 multimodal terbaru dengan visual token budget fleksibel untuk kantor cabang".to_string(),
        },
        CatalogModel {
            id: "qwen3-8b-doc".to_string(),
            name: "Qwen3 8B (Q4_K_M)".to_string(),
            family: "Qwen3".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 8.2,
            active_params_b: 8.2,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 5030,
            min_vram_mb: 7500,
            recommended_vram_mb: 10500,
            min_ram_for_offload_mb: None,
            filename: "Qwen3-8B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "All-rounder 8B cepat & akurat untuk GPU 12 GB / Mac 16 GB dengan JSON strict".to_string(),
        },
        CatalogModel {
            id: "deepseek-r1-0528-qwen3-8b".to_string(),
            name: "DeepSeek-R1-0528 Qwen3 8B Reasoning (Q4_K_M)".to_string(),
            family: "DeepSeek-R1".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 8.2,
            active_params_b: 8.2,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: false,
            disk_mb: 5030,
            min_vram_mb: 7500,
            recommended_vram_mb: 10500,
            min_ram_for_offload_mb: None,
            filename: "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Reasoning mendalam versi R1-0528 untuk audit klaim, analisis kebijakan, & verifikasi logika".to_string(),
        },
        CatalogModel {
            id: "glm-4-9b-0414".to_string(),
            name: "GLM-4 9B 0414 (Q4_K_M)".to_string(),
            family: "GLM-4".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 9.4,
            active_params_b: 9.4,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 6170,
            min_vram_mb: 9000,
            recommended_vram_mb: 12000,
            min_ram_for_offload_mb: None,
            filename: "GLM-4-9B-0414-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/GLM-4-9B-0414-GGUF/resolve/main/GLM-4-9B-0414-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Kuat di penulisan laporan panjang, multilingual Asia, & tool calling di GPU 12 GB".to_string(),
        },
        CatalogModel {
            id: "gemma-3-12b-vlm".to_string(),
            name: "Gemma 3 12B Instruct Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Gemma 3".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 12.2,
            active_params_b: 12.2,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 8114,
            min_vram_mb: 11500,
            recommended_vram_mb: 14500,
            min_ram_for_offload_mb: None,
            filename: "gemma-3-12b-it-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("gemma-3-12b-it-mmproj-BF16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/mmproj-BF16.gguf".to_string()),
            best_for: "Multimodal 12B stabil untuk GPU 16 GB / Mac 24 GB dengan bahasa Indonesia sangat natural".to_string(),
        },
        CatalogModel {
            id: "phi-4-14b".to_string(),
            name: "Microsoft Phi-4 14B (Q4_K_M)".to_string(),
            family: "Phi-4".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 14.7,
            active_params_b: 14.7,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 8890,
            min_vram_mb: 12500,
            recommended_vram_mb: 16000,
            min_ram_for_offload_mb: None,
            filename: "phi-4-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Reasoning presisi tinggi & ekstraksi data terstruktur tanpa bloat thinking token".to_string(),
        },
        CatalogModel {
            id: "qwen3-vl-8b-doc".to_string(),
            name: "Qwen3-VL 8B Instruct Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3-VL".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 8.8,
            active_params_b: 8.8,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 5900,
            min_vram_mb: 9000,
            recommended_vram_mb: 12500,
            min_ram_for_offload_mb: None,
            filename: "Qwen3-VL-8B-Instruct-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3-VL-8B-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "Spesialis baca tabel rumit, faktur padat, dan lampiran scan visual lokal".to_string(),
        },
        CatalogModel {
            id: "qwen3.5-9b-doc".to_string(),
            name: "Qwen3.5 9B Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3.5".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 9.0,
            active_params_b: 9.0,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: false,
            disk_mb: 6293,
            min_vram_mb: 9500,
            recommended_vram_mb: 13000,
            min_ram_for_offload_mb: None,
            filename: "Qwen3.5-9B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3.5-9B-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "Model multimodal terbaru 2026 untuk analisis dokumen + gambar di GPU 12-16 GB".to_string(),
        },
        CatalogModel {
            id: "qwen3-14b-doc".to_string(),
            name: "Qwen3 14B (Q4_K_M)".to_string(),
            family: "Qwen3".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 14.8,
            active_params_b: 14.8,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 8585,
            min_vram_mb: 12500,
            recommended_vram_mb: 15500,
            min_ram_for_offload_mb: None,
            filename: "Qwen3-14B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Sweet-spot akurasi tinggi untuk klaim, faktur PBF, & kontrak di GPU 16 GB / Mac 24 GB+".to_string(),
        },
        CatalogModel {
            id: "gpt-oss-20b-moe".to_string(),
            name: "gpt-oss 20B MoE (UD-Q4_K_XL)".to_string(),
            family: "gpt-oss".to_string(),
            architecture: "moe".to_string(),
            total_params_b: 21.0,
            active_params_b: 3.6,
            quant: "UD-Q4_K_XL".to_string(),
            multimodal: false,
            non_thinking: false,
            disk_mb: 11322,
            min_vram_mb: 14500,
            recommended_vram_mb: 17000,
            min_ram_for_offload_mb: Some(24_000),
            filename: "gpt-oss-20b-UD-Q4_K_XL.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-UD-Q4_K_XL.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Open-weight MoE kencang (3.6B aktif) untuk tool calling & analisis mendalam di 16-24 GB".to_string(),
        },
        CatalogModel {
            id: "mistral-small-3.2-24b".to_string(),
            name: "Mistral Small 3.2 24B Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Mistral Small 3.2".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 24.0,
            active_params_b: 24.0,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 14507,
            min_vram_mb: 18500,
            recommended_vram_mb: 24000,
            min_ram_for_offload_mb: Some(32_000),
            filename: "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/resolve/main/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Mistral-Small-3.2-24B-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "Non-thinking multimodal 24B kuat di function calling, JSON strict, & kontrak korporat".to_string(),
        },
        CatalogModel {
            id: "devstral-small-24b".to_string(),
            name: "Devstral Small 2505 24B Agent/Code (Q4_K_M + mmproj)".to_string(),
            family: "Mistral Devstral".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 24.0,
            active_params_b: 24.0,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 15167,
            min_vram_mb: 19000,
            recommended_vram_mb: 24000,
            min_ram_for_offload_mb: Some(32_000),
            filename: "Devstral-Small-2505-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Devstral-Small-2505-GGUF/resolve/main/Devstral-Small-2505-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Devstral-Small-2505-mmproj-BF16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Devstral-Small-2505-GGUF/resolve/main/mmproj-BF16.gguf".to_string()),
            best_for: "Spesialis coding agentic & tool-use multi-langkah (SWE-bench) di GPU 24-32 GB".to_string(),
        },
        CatalogModel {
            id: "gemma-3-27b-vlm".to_string(),
            name: "Gemma 3 27B Instruct Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Gemma 3".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 27.4,
            active_params_b: 27.4,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 17368,
            min_vram_mb: 22000,
            recommended_vram_mb: 28000,
            min_ram_for_offload_mb: Some(36_000),
            filename: "gemma-3-27b-it-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("gemma-3-27b-it-mmproj-BF16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/mmproj-BF16.gguf".to_string()),
            best_for: "Dense 27B multimodal matang untuk ringkasan dokumen medis & hukum panjang".to_string(),
        },
        CatalogModel {
            id: "qwen3-coder-30b-a3b".to_string(),
            name: "Qwen3-Coder 30B-A3B MoE (Q4_K_M)".to_string(),
            family: "Qwen3-Coder".to_string(),
            architecture: "moe".to_string(),
            total_params_b: 30.5,
            active_params_b: 3.3,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 18560,
            min_vram_mb: 22000,
            recommended_vram_mb: 28000,
            min_ram_for_offload_mb: Some(32_000),
            filename: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "MoE coding & tool-calling super cepat (3.3B aktif) untuk integrasi agentic & otomasi".to_string(),
        },
        CatalogModel {
            id: "gemma-4-31b-vlm".to_string(),
            name: "Gemma 4 31B Instruct Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Gemma 4".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 31.0,
            active_params_b: 31.0,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 19465,
            min_vram_mb: 24000,
            recommended_vram_mb: 32000,
            min_ram_for_offload_mb: Some(40_000),
            filename: "gemma-4-31B-it-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("gemma-4-31B-it-mmproj-BF16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/mmproj-BF16.gguf".to_string()),
            best_for: "Flagship dense Gemma 4 31B multimodal untuk akurasi puncak di server 32-48 GB VRAM".to_string(),
        },
        CatalogModel {
            id: "qwen3-32b-doc".to_string(),
            name: "Qwen3 32B (Q4_K_M)".to_string(),
            family: "Qwen3".to_string(),
            architecture: "dense".to_string(),
            total_params_b: 32.8,
            active_params_b: 32.8,
            quant: "Q4_K_M".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 19760,
            min_vram_mb: 24500,
            recommended_vram_mb: 32000,
            min_ram_for_offload_mb: Some(40_000),
            filename: "Qwen3-32B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "Flagship dense 32B untuk analisis hukum, audit kontrak, & ekstraksi kompleks di server 32-48 GB".to_string(),
        },
        CatalogModel {
            id: "gemma-4-26b-a4b-moe".to_string(),
            name: "Gemma 4 26B-A4B MoE Multimodal (UD-Q4_K_M + mmproj)".to_string(),
            family: "Gemma 4".to_string(),
            architecture: "moe".to_string(),
            total_params_b: 26.0,
            active_params_b: 4.0,
            quant: "UD-Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: true,
            disk_mb: 17300,
            min_vram_mb: 21000,
            recommended_vram_mb: 26000,
            min_ram_for_offload_mb: Some(32_000),
            filename: "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("gemma-4-26B-A4B-it-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "Gemma 4 MoE multimodal (4B aktif): sangat cepat untuk server 24-48 GB VRAM".to_string(),
        },
        CatalogModel {
            id: "qwen3-30b-a3b-doc".to_string(),
            name: "Qwen3 30B-A3B Instruct 2507 MoE (UD-Q4_K_XL)".to_string(),
            family: "Qwen3-MoE".to_string(),
            architecture: "moe".to_string(),
            total_params_b: 30.5,
            active_params_b: 3.3,
            quant: "UD-Q4_K_XL".to_string(),
            multimodal: false,
            non_thinking: true,
            disk_mb: 16871,
            min_vram_mb: 21000,
            recommended_vram_mb: 26000,
            min_ram_for_offload_mb: Some(32_000),
            filename: "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf".to_string(),
            mmproj_filename: None,
            mmproj_url: None,
            best_for: "MoE non-thinking tercepat (3.3B aktif) untuk batch besar klaim & invoice di server 24-48 GB".to_string(),
        },
        CatalogModel {
            id: "qwen3.5-27b-doc".to_string(),
            name: "Qwen3.5 27B Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3.5".to_string(),
            architecture: "vlm".to_string(),
            total_params_b: 27.0,
            active_params_b: 27.0,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: false,
            disk_mb: 16850,
            min_vram_mb: 22000,
            recommended_vram_mb: 30000,
            min_ram_for_offload_mb: Some(36_000),
            filename: "Qwen3.5-27B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3.5-27B-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "Flagship dense 27B multimodal untuk server 32-48 GB (L40S/A6000/Mac Studio 64 GB)".to_string(),
        },
        CatalogModel {
            id: "qwen3.5-35b-a3b-moe".to_string(),
            name: "Qwen3.5 35B-A3B MoE Multimodal (Q4_K_M + mmproj)".to_string(),
            family: "Qwen3.5-MoE".to_string(),
            architecture: "moe".to_string(),
            total_params_b: 35.0,
            active_params_b: 3.0,
            quant: "Q4_K_M".to_string(),
            multimodal: true,
            non_thinking: false,
            disk_mb: 21854,
            min_vram_mb: 26000,
            recommended_vram_mb: 34000,
            min_ram_for_offload_mb: Some(48_000),
            filename: "Qwen3.5-35B-A3B-Q4_K_M.gguf".to_string(),
            download_url: "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf".to_string(),
            mmproj_filename: Some("Qwen3.5-35B-A3B-mmproj-F16.gguf".to_string()),
            mmproj_url: Some("https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/mmproj-F16.gguf".to_string()),
            best_for: "Flagship MoE multimodal 2026 (3B aktif) untuk server produksi 48 GB VRAM / Mac Studio 64 GB+".to_string(),
        },
    ]
}

pub fn find_catalog_model(id: &str) -> Option<CatalogModel> {
    curated_catalog().into_iter().find(|m| m.id == id)
}

pub fn recommend_for_hardware(hw: &HardwareReport, models_dir: &Path) -> Vec<RecommendedModel> {
    let usable_vram = hw.total_usable_vram_mb;
    let total_ram = hw.total_memory_mb;

    let catalog = curated_catalog();
    // Pick highest-capacity dense non-thinking model that fits comfortably inside recommended_vram_mb
    let top_dense_id = catalog
        .iter()
        .filter(|m| m.non_thinking && usable_vram >= m.recommended_vram_mb)
        .max_by(|a, b| a.total_params_b.total_cmp(&b.total_params_b))
        .map(|m| m.id.clone())
        .or_else(|| {
            catalog
                .iter()
                .filter(|m| usable_vram >= m.min_vram_mb)
                .max_by(|a, b| a.total_params_b.total_cmp(&b.total_params_b))
                .map(|m| m.id.clone())
        });

    let mut out: Vec<RecommendedModel> = catalog
        .into_iter()
        .map(|m| {
            let local_file = models_dir.join(&m.filename);
            let mmproj_file = m.mmproj_filename.as_ref().map(|f| models_dir.join(f));
            let installed = local_file.exists()
                && mmproj_file.as_ref().map(|p| p.exists()).unwrap_or(true);

            let (status, reason) = if top_dense_id.as_deref() == Some(m.id.as_str()) {
                (
                    "recommended".to_string(),
                    format!(
                        "Paling optimal untuk VRAM/Unified Memory {} MB dengan ruang aman KV-Cache 32k.",
                        usable_vram
                    ),
                )
            } else if usable_vram >= m.recommended_vram_mb {
                (
                    "compatible".to_string(),
                    format!(
                        "Berjalan lancar penuh di GPU ({} MB tersedia vs {} MB rekomendasi).",
                        usable_vram, m.recommended_vram_mb
                    ),
                )
            } else if usable_vram >= m.min_vram_mb {
                (
                    "conditional".to_string(),
                    format!(
                        "Bisa jalan, tetapi headroom KV-Cache terbatas ({} MB tersedia; disarankan context <= 16k).",
                        usable_vram
                    ),
                )
            } else if m
                .min_ram_for_offload_mb
                .map(|req_ram| total_ram >= req_ram)
                .unwrap_or(false)
            {
                (
                    "conditional".to_string(),
                    format!(
                        "MoE dapat berjalan via CPU Expert Offload (RAM {} MB memadai, meski VRAM {} MB < {} MB).",
                        total_ram, usable_vram, m.min_vram_mb
                    ),
                )
            } else {
                (
                    "incompatible".to_string(),
                    format!(
                        "Membutuhkan minimal {} MB VRAM (tersedia {} MB).",
                        m.min_vram_mb, usable_vram
                    ),
                )
            };

            let (capabilities, tags) = derive_capabilities(&m);

            RecommendedModel {
                local_path: if local_file.exists() {
                    Some(local_file.display().to_string())
                } else {
                    None
                },
                mmproj_local_path: mmproj_file
                    .filter(|p| p.exists())
                    .map(|p| p.display().to_string()),
                installed,
                status,
                reason,
                capabilities,
                tags,
                model: m,
            }
        })
        .collect();

    // Sort: recommended -> compatible -> conditional -> incompatible
    out.sort_by_key(|r| match r.status.as_str() {
        "recommended" => 0,
        "compatible" => 1,
        "conditional" => 2,
        _ => 3,
    });
    out
}

pub fn list_local_models(models_dir: &Path) -> Vec<LocalModelFile> {
    let Ok(rd) = std::fs::read_dir(models_dir) else {
        return Vec::new();
    };
    let mut files = Vec::new();
    for entry in rd.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("gguf") {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            files.push(LocalModelFile {
                filename: entry.file_name().to_string_lossy().to_string(),
                path: path.display().to_string(),
                size_mb: meta.len() / (1024 * 1024),
            });
        }
    }
    files.sort_by(|a, b| a.filename.cmp(&b.filename));
    files
}

async fn download_single_file<F>(
    client: &Client,
    url: &str,
    dest: &Path,
    mut on_progress: F,
) -> Result<()>
where
    F: FnMut(u64, Option<u64>),
{
    if dest.exists() {
        let len = std::fs::metadata(dest)?.len();
        on_progress(len, Some(len));
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut req = client.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            req = req.bearer_auth(token.trim());
        }
    }

    let resp = req
        .send()
        .await
        .with_context(|| format!("gagal mengunduh {url}"))?;
    if !resp.status().is_success() {
        bail!("HTTP {} saat mengunduh {url}", resp.status());
    }

    let total = resp.content_length();
    let part_path = PathBuf::from(format!("{}.part", dest.display()));
    let mut file = tokio::fs::File::create(&part_path)
        .await
        .with_context(|| format!("gagal membuat file {}", part_path.display()))?;

    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.context("error saat membaca stream unduhan")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }
    file.flush().await?;
    drop(file);

    tokio::fs::rename(&part_path, dest)
        .await
        .with_context(|| format!("gagal memindahkan {} ke {}", part_path.display(), dest.display()))?;
    Ok(())
}

pub async fn pull_model_cli<F>(model_id: &str, models_dir: &Path, mut on_progress: F) -> Result<(PathBuf, Option<PathBuf>)>
where
    F: FnMut(&str, u64, Option<u64>),
{
    let model = find_catalog_model(model_id)
        .ok_or_else(|| anyhow!("Model '{model_id}' tidak ada di katalog. Jalankan `cmod-inference recommend`."))?;
    let client = Client::builder().build()?;

    let dest = models_dir.join(&model.filename);
    download_single_file(&client, &model.download_url, &dest, |dl, tot| {
        on_progress(&model.filename, dl, tot);
    })
    .await?;

    let mmproj_dest = if let (Some(mm_name), Some(mm_url)) = (&model.mmproj_filename, &model.mmproj_url) {
        let p = models_dir.join(mm_name);
        download_single_file(&client, mm_url, &p, |dl, tot| {
            on_progress(mm_name, dl, tot);
        })
        .await?;
        Some(p)
    } else {
        None
    };

    Ok((dest, mmproj_dest))
}

#[derive(Default)]
pub struct DownloadManager {
    jobs: Mutex<HashMap<String, DownloadProgress>>,
}

impl DownloadManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jobs: Mutex::new(HashMap::new()),
        })
    }

    pub fn get(&self, model_id: &str) -> Option<DownloadProgress> {
        self.jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(model_id)
            .cloned()
    }

    pub fn start_download(
        self: &Arc<Self>,
        model_id: String,
        models_dir: PathBuf,
    ) -> Result<DownloadProgress> {
        let model = find_catalog_model(&model_id)
            .ok_or_else(|| anyhow!("Model '{model_id}' tidak ditemukan di katalog"))?;

        if let Some(existing) = self.get(&model_id) {
            if existing.status == "downloading" {
                return Ok(existing);
            }
        }

        let initial = DownloadProgress {
            model_id: model_id.clone(),
            status: "downloading".to_string(),
            downloaded_bytes: 0,
            total_bytes: Some(model.disk_mb * 1024 * 1024),
            percent: Some(0.0),
            local_path: None,
            mmproj_path: None,
            error: None,
        };
        self.jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(model_id.clone(), initial.clone());

        let mgr = self.clone();
        tokio::spawn(async move {
            let client = Client::new();
            let dest = models_dir.join(&model.filename);
            let res = download_single_file(&client, &model.download_url, &dest, |dl, tot| {
                let pct = tot.filter(|&t| t > 0).map(|t| (dl as f32 / t as f32) * 100.0);
                if let Ok(mut g) = mgr.jobs.lock() {
                    if let Some(entry) = g.get_mut(&model_id) {
                        entry.downloaded_bytes = dl;
                        entry.total_bytes = tot;
                        entry.percent = pct;
                    }
                }
            })
            .await;

            let final_res = match res {
                Ok(()) => {
                    if let (Some(mm_name), Some(mm_url)) = (&model.mmproj_filename, &model.mmproj_url) {
                        let mm_dest = models_dir.join(mm_name);
                        download_single_file(&client, mm_url, &mm_dest, |_, _| {})
                            .await
                            .map(|()| (dest, Some(mm_dest)))
                    } else {
                        Ok((dest, None))
                    }
                }
                Err(e) => Err(e),
            };

            if let Ok(mut g) = mgr.jobs.lock() {
                if let Some(entry) = g.get_mut(&model_id) {
                    match final_res {
                        Ok((p, mm)) => {
                            entry.status = "completed".to_string();
                            entry.percent = Some(100.0);
                            entry.local_path = Some(p.display().to_string());
                            entry.mmproj_path = mm.map(|m| m.display().to_string());
                        }
                        Err(err) => {
                            entry.status = "failed".to_string();
                            entry.error = Some(err.to_string());
                        }
                    }
                }
            }
        });

        Ok(initial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::estimate_sizing;

    #[test]
    fn recommends_14b_for_m4_pro_24gb_and_7b_for_12gb() {
        let m4_hw = HardwareReport {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            cpu_brand: "Apple M4 Pro".to_string(),
            cpu_cores: 14,
            total_memory_mb: 24_576,
            available_memory_mb: 16_000,
            gpus: Vec::new(),
            available_accelerators: vec!["metal".to_string(), "cpu".to_string()],
            recommended_backend: "llama-metal".to_string(),
            recommended_runtime: "llama-server (Apple Metal)".to_string(),
            total_usable_vram_mb: 17_203,
            sizing: estimate_sizing(17_203, "llama-metal"),
        };
        let recs = recommend_for_hardware(&m4_hw, Path::new("/tmp/nonexistent_cmod_models"));
        assert_eq!(recs[0].status, "recommended");
        assert_eq!(recs[0].model.id, "qwen3-14b-doc");

        let gpu_12gb = HardwareReport {
            total_usable_vram_mb: 12_800,
            total_memory_mb: 16_384,
            ..m4_hw
        };
        let recs2 = recommend_for_hardware(&gpu_12gb, Path::new("/tmp/nonexistent_cmod_models"));
        assert!(recs2.len() >= 30);
        assert_eq!(recs2[0].status, "recommended");
        assert_eq!(recs2[0].model.id, "glm-4-9b-0414");
    }
}
