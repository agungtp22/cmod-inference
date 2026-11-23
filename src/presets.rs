use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPreset {
    pub id: String,
    pub label: String,
    pub purpose: String,
    pub recommended_models: Vec<String>,
    pub architecture: String,
    pub multimodal: bool,
    pub min_quant: String,
    pub kv_cache_quant: String,
    pub default_temperature: f64,
    pub non_thinking: bool,
    pub min_context_len: u32,
    pub recommended_context_len: u32,
}

pub fn builtin_presets() -> Vec<ModelPreset> {
    vec![
        ModelPreset {
            id: "cmod-doc-extract".to_string(),
            label: "Document Extraction & Mapping (Non-Thinking)".to_string(),
            purpose: "Ekstraksi JSON terstruktur & pemetaan katalog deterministik".to_string(),
            recommended_models: vec![
                "Qwen2.5-14B-Instruct".to_string(),
                "Qwen3-30B-A3B-Instruct".to_string(),
            ],
            architecture: "dense_or_moe".to_string(),
            multimodal: false,
            min_quant: "Q4_K_M / AWQ / FP8".to_string(),
            kv_cache_quant: "q8_0".to_string(),
            default_temperature: 0.0,
            non_thinking: true,
            min_context_len: 16_384,
            recommended_context_len: 32_768,
        },
        ModelPreset {
            id: "cmod-doc-chat".to_string(),
            label: "Cited Document Q&A (Long Context)".to_string(),
            purpose: "Tanya jawab lintas halaman bersitasi [Halaman N] & ringkasan dokumen".to_string(),
            recommended_models: vec![
                "Qwen2.5-14B-Instruct".to_string(),
                "Qwen2.5-32B-Instruct".to_string(),
                "Qwen3-30B-A3B".to_string(),
            ],
            architecture: "dense_or_moe".to_string(),
            multimodal: false,
            min_quant: "Q4_K_M / AWQ / FP8".to_string(),
            kv_cache_quant: "q8_0".to_string(),
            default_temperature: 0.1,
            non_thinking: false,
            min_context_len: 32_768,
            recommended_context_len: 65_536,
        },
        ModelPreset {
            id: "cmod-doc-vision".to_string(),
            label: "Multimodal Document VLM (Local On-Prem)".to_string(),
            purpose: "Halaman tabel rumit, formulir scan padat, atau lampiran visual lokal".to_string(),
            recommended_models: vec![
                "Qwen2.5-VL-7B-Instruct".to_string(),
                "Qwen2.5-VL-32B-Instruct".to_string(),
                "Gemma-3-12B-IT".to_string(),
            ],
            architecture: "vlm".to_string(),
            multimodal: true,
            min_quant: "Q4_K_M / FP8 (+ mmproj)".to_string(),
            kv_cache_quant: "q8_0".to_string(),
            default_temperature: 0.0,
            non_thinking: true,
            min_context_len: 16_384,
            recommended_context_len: 32_768,
        },
        ModelPreset {
            id: "cmod-doc-moe".to_string(),
            label: "High-Throughput MoE Server".to_string(),
            purpose: "Throughput tinggi dengan total parameter besar & active parameter kecil".to_string(),
            recommended_models: vec![
                "Qwen3-30B-A3B-Instruct".to_string(),
                "Qwen3-235B-A22B".to_string(),
            ],
            architecture: "moe".to_string(),
            multimodal: false,
            min_quant: "Q4_K_M / UD-Q4_K_XL / FP8".to_string(),
            kv_cache_quant: "q8_0".to_string(),
            default_temperature: 0.0,
            non_thinking: true,
            min_context_len: 32_768,
            recommended_context_len: 65_536,
        },
    ]
}

pub fn find_preset(id: &str) -> Option<ModelPreset> {
    builtin_presets().into_iter().find(|p| p.id == id)
}
