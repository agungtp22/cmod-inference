use std::process::Command;

use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuDevice {
    pub vendor: String,
    pub name: String,
    pub vram_mb: Option<u64>,
    pub driver: Option<String>,
    pub backend_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSizingHint {
    pub tier: String,
    pub dense_max_params_b: u32,
    pub moe_max_total_params_b: u32,
    pub recommended_quant: String,
    pub kv_cache_quant: String,
    pub recommended_context_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareReport {
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_memory_mb: u64,
    pub available_memory_mb: u64,
    pub gpus: Vec<GpuDevice>,
    pub available_accelerators: Vec<String>,
    pub recommended_backend: String,
    pub recommended_runtime: String,
    pub total_usable_vram_mb: u64,
    pub sizing: ModelSizingHint,
}

fn run_cmd(bin: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

pub fn detect_hardware() -> HardwareReport {
    let sys = System::new_all();
    let total_memory_mb = sys.total_memory() / 1024 / 1024;
    let available_memory_mb = sys.available_memory() / 1024 / 1024;
    let cpu_cores = sys.cpus().len().max(1);
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let mut gpus = Vec::new();
    let mut accelerators = Vec::new();

    // 1. NVIDIA CUDA
    if let Some(out) = run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
    ) {
        for line in out.lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if let Some(name) = parts.first().filter(|s| !s.is_empty()) {
                let vram_mb = parts.get(1).and_then(|v| v.parse::<u64>().ok());
                let driver = parts.get(2).map(|s| s.to_string());
                gpus.push(GpuDevice {
                    vendor: "nvidia".to_string(),
                    name: (*name).to_string(),
                    vram_mb,
                    driver,
                    backend_hint: "cuda".to_string(),
                });
            }
        }
        if !gpus.is_empty() {
            accelerators.push("cuda".to_string());
        }
    }

    // 2. AMD ROCm
    if let Some(out) = run_cmd("rocm-smi", &["--showproductname"]) {
        let mut found_rocm = false;
        for line in out.lines() {
            if line.contains("Card series") || line.contains("Card model") {
                if let Some((_, val)) = line.split_once(':') {
                    let name = val.trim();
                    if !name.is_empty() {
                        found_rocm = true;
                        gpus.push(GpuDevice {
                            vendor: "amd".to_string(),
                            name: name.to_string(),
                            vram_mb: None,
                            driver: Some("rocm".to_string()),
                            backend_hint: "rocm".to_string(),
                        });
                    }
                }
            }
        }
        if found_rocm {
            accelerators.push("rocm".to_string());
        }
    }

    // 3. Apple Silicon Metal (Unified Memory)
    if os == "macos" && arch == "aarch64" {
        let usable_unified = (total_memory_mb * 70) / 100;
        gpus.push(GpuDevice {
            vendor: "apple".to_string(),
            name: format!("Apple Silicon GPU ({cpu_brand})"),
            vram_mb: Some(usable_unified),
            driver: Some("metal".to_string()),
            backend_hint: "metal".to_string(),
        });
        accelerators.push("metal".to_string());
    }

    // 4. Vulkan probe (Intel Arc / AMD / Mixed GPU)
    if let Some(out) = run_cmd("vulkaninfo", &["--summary"]) {
        if out.contains("deviceName") {
            accelerators.push("vulkan".to_string());
            if gpus.is_empty() {
                for line in out.lines() {
                    if let Some((_, name)) = line.split_once("deviceName") {
                        let clean = name.trim_start_matches([' ', '=', ':']).trim();
                        if !clean.is_empty() {
                            gpus.push(GpuDevice {
                                vendor: infer_vendor(clean),
                                name: clean.to_string(),
                                vram_mb: None,
                                driver: Some("vulkan".to_string()),
                                backend_hint: "vulkan".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }

    accelerators.push("cpu".to_string());

    let (recommended_backend, recommended_runtime) = if accelerators.contains(&"cuda".to_string()) {
        if os == "linux" {
            ("vllm-cuda".to_string(), "vLLM (NVIDIA CUDA)".to_string())
        } else {
            ("llama-cuda".to_string(), "llama-server (NVIDIA CUDA)".to_string())
        }
    } else if accelerators.contains(&"rocm".to_string()) {
        ("vllm-rocm".to_string(), "vLLM (AMD ROCm)".to_string())
    } else if accelerators.contains(&"metal".to_string()) {
        ("llama-metal".to_string(), "llama-server (Apple Metal)".to_string())
    } else if accelerators.contains(&"vulkan".to_string()) {
        ("llama-vulkan".to_string(), "llama-server (Vulkan Multi-GPU)".to_string())
    } else {
        ("llama-cpu".to_string(), "llama-server (CPU Fallback)".to_string())
    };

    let total_usable_vram_mb: u64 = gpus
        .iter()
        .filter_map(|g| g.vram_mb)
        .sum::<u64>()
        .max(if gpus.is_empty() { available_memory_mb / 2 } else { 0 });

    let sizing = estimate_sizing(total_usable_vram_mb, &recommended_backend);

    HardwareReport {
        os,
        arch,
        cpu_brand,
        cpu_cores,
        total_memory_mb,
        available_memory_mb,
        gpus,
        available_accelerators: accelerators,
        recommended_backend,
        recommended_runtime,
        total_usable_vram_mb,
        sizing,
    }
}

fn infer_vendor(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("rtx") {
        "nvidia".to_string()
    } else if lower.contains("amd") || lower.contains("radeon") || lower.contains("instinct") {
        "amd".to_string()
    } else if lower.contains("intel") || lower.contains("arc") {
        "intel".to_string()
    } else if lower.contains("apple") {
        "apple".to_string()
    } else {
        "other".to_string()
    }
}

pub fn estimate_sizing(usable_vram_mb: u64, backend: &str) -> ModelSizingHint {
    let is_vllm = backend.starts_with("vllm");
    let kv = if is_vllm { "fp8" } else { "q8_0" }.to_string();

    if usable_vram_mb >= 80_000 {
        ModelSizingHint {
            tier: "high".to_string(),
            dense_max_params_b: 72,
            moe_max_total_params_b: 120,
            recommended_quant: if is_vllm { "FP8 / AWQ" } else { "Q5_K_M / Q8_0" }.to_string(),
            kv_cache_quant: kv,
            recommended_context_len: 65_536,
        }
    } else if usable_vram_mb >= 40_000 {
        ModelSizingHint {
            tier: "middle".to_string(),
            dense_max_params_b: 32,
            moe_max_total_params_b: 70,
            recommended_quant: if is_vllm { "AWQ / FP8" } else { "Q4_K_M / Q5_K_M" }.to_string(),
            kv_cache_quant: kv,
            recommended_context_len: 32_768,
        }
    } else if usable_vram_mb >= 14_000 {
        ModelSizingHint {
            tier: "low".to_string(),
            dense_max_params_b: 14,
            moe_max_total_params_b: 30,
            recommended_quant: if is_vllm { "AWQ INT4" } else { "Q4_K_M" }.to_string(),
            kv_cache_quant: kv,
            recommended_context_len: 32_768,
        }
    } else {
        ModelSizingHint {
            tier: "compact".to_string(),
            dense_max_params_b: 7,
            moe_max_total_params_b: 14,
            recommended_quant: "Q4_K_M".to_string(),
            kv_cache_quant: "q8_0".to_string(),
            recommended_context_len: 16_384,
        }
    }
}
