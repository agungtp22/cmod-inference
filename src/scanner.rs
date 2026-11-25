use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MAX_SANE_TENSORS: u64 = 100_000;
const MAX_SANE_KV: u64 = 100_000;
const MAX_SANE_STRING_LEN: u64 = 16 * 1024 * 1024; // 16 MiB
const MAX_SANE_ARRAY_LEN: u64 = 2_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ScanSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ScanVerdict {
    Pass,
    Warn,
    Block,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanFinding {
    pub severity: ScanSeverity,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelScanReport {
    pub path: String,
    pub file_size_bytes: u64,
    pub verdict: ScanVerdict,
    pub safe_to_load: bool,
    pub gguf_version: Option<u32>,
    pub tensor_count: Option<u64>,
    pub metadata_kv_count: Option<u64>,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub chat_template_len: Option<usize>,
    pub sha256: Option<String>,
    pub sha256_verified: Option<bool>,
    pub scan_duration_ms: u128,
    pub findings: Vec<ScanFinding>,
}

fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r).context("gagal membaca panjang string GGUF")?;
    if len > MAX_SANE_STRING_LEN {
        anyhow::bail!("panjang string GGUF tidak wajar ({len} byte > batas aman)");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)
        .context("EOF tak terduga saat membaca string GGUF")?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_gguf_value<R: Read + Seek>(r: &mut R, vtype: u32) -> Result<Option<String>> {
    match vtype {
        // UINT8, INT8, BOOL
        0 | 1 | 7 => {
            let _ = read_u8(r)?;
            Ok(None)
        }
        // UINT16, INT16
        2 | 3 => {
            r.seek(SeekFrom::Current(2))?;
            Ok(None)
        }
        // UINT32, INT32, FLOAT32
        4 | 5 | 6 => {
            r.seek(SeekFrom::Current(4))?;
            Ok(None)
        }
        // STRING
        8 => Ok(Some(read_gguf_string(r)?)),
        // ARRAY
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            if count > MAX_SANE_ARRAY_LEN {
                anyhow::bail!("panjang array metadata GGUF tidak wajar ({count} elemen)");
            }
            // Fast-skip fixed-width primitive arrays in O(1) seek
            let fixed_elem_size: Option<u64> = match elem_type {
                0 | 1 | 7 => Some(1),
                2 | 3 => Some(2),
                4 | 5 | 6 => Some(4),
                10 | 11 | 12 => Some(8),
                _ => None,
            };
            if let Some(sz) = fixed_elem_size {
                let total = count
                    .checked_mul(sz)
                    .ok_or_else(|| anyhow::anyhow!("overflow ukuran array metadata GGUF"))?;
                r.seek(SeekFrom::Current(total as i64))?;
            } else {
                for _ in 0..count {
                    let _ = skip_gguf_value(r, elem_type)?;
                }
            }
            Ok(None)
        }
        // UINT64, INT64, FLOAT64
        10 | 11 | 12 => {
            r.seek(SeekFrom::Current(8))?;
            Ok(None)
        }
        other => anyhow::bail!("tipe nilai metadata GGUF tidak dikenal: {other}"),
    }
}

fn detect_disguised_format(magic: &[u8]) -> Option<&'static str> {
    if magic.len() >= 4 && &magic[..4] == b"\x7fELF" {
        return Some("ELF Linux executable");
    }
    if magic.len() >= 4
        && (magic[..4] == [0xCF, 0xFA, 0xED, 0xFE]
            || magic[..4] == [0xFE, 0xED, 0xFA, 0xCF]
            || magic[..4] == [0xCA, 0xFE, 0xBA, 0xBE])
    {
        return Some("Mach-O macOS executable");
    }
    if magic.len() >= 2 && &magic[..2] == b"MZ" {
        return Some("Windows PE executable (.exe/.dll)");
    }
    if magic.len() >= 4 && &magic[..4] == b"PK\x03\x04" {
        return Some("ZIP / PyTorch archive (berpotensi mengandung Python pickle)");
    }
    if magic.len() >= 2 && magic[0] == 0x80 && (2..=5).contains(&magic[1]) {
        return Some("Python Pickle binary (risiko Remote Code Execution)");
    }
    None
}

fn audit_chat_template(template: &str, findings: &mut Vec<ScanFinding>) {
    let lower = template.to_ascii_lowercase();

    const SSTI_PATTERNS: &[(&str, &str)] = &[
        ("__class__", "Jinja SSTI gadget `__class__`"),
        ("__subclasses__", "Jinja SSTI gadget `__subclasses__`"),
        ("__mro__", "Jinja SSTI gadget `__mro__`"),
        ("__globals__", "Jinja SSTI gadget `__globals__`"),
        ("__builtins__", "Jinja SSTI gadget `__builtins__`"),
        ("__import__", "Python code-exec gadget `__import__`"),
        ("os.system", "Shell execution gadget `os.system`"),
        ("subprocess.", "Process execution gadget `subprocess`"),
        ("eval(", "Dynamic code execution `eval(`"),
        ("exec(", "Dynamic code execution `exec(`"),
    ];

    for (pat, desc) in SSTI_PATTERNS {
        if lower.contains(pat) {
            findings.push(ScanFinding {
                severity: ScanSeverity::Critical,
                code: "TEMPLATE_SSTI_PATTERN".to_string(),
                message: format!(
                    "Ditemukan pola berbahaya di tokenizer.chat_template: {desc}"
                ),
            });
        }
    }

    const SUSPICIOUS_PATTERNS: &[(&str, &str)] = &[
        ("http://", "URL eksternal `http://` di dalam chat_template"),
        ("https://", "URL eksternal `https://` di dalam chat_template"),
        ("curl ", "Perintah `curl` di dalam chat_template"),
        ("wget ", "Perintah `wget` di dalam chat_template"),
        ("bash -c", "Perintah `bash -c` di dalam chat_template"),
        ("/bin/sh", "Referensi `/bin/sh` di dalam chat_template"),
        (
            "ignore previous instructions",
            "Indikasi hidden prompt injection di chat_template",
        ),
    ];

    for (pat, desc) in SUSPICIOUS_PATTERNS {
        if lower.contains(pat) {
            findings.push(ScanFinding {
                severity: ScanSeverity::Warning,
                code: "TEMPLATE_SUSPICIOUS_CONTENT".to_string(),
                message: format!("Peringatan pada tokenizer.chat_template: {desc}"),
            });
        }
    }
}

pub fn compute_file_sha256(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("Gagal membuka file untuk SHA-256: {}", path.display()))?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Perform a memory-safe, pure-Rust security scan of a GGUF model file BEFORE `llama.cpp`
/// touches it:
/// 1. Verifies GGUF magic/version and blocks disguised executables or Python pickles.
/// 2. Validates header counts & all tensor offsets against file bounds.
/// 3. Audits `tokenizer.chat_template` and metadata for SSTI / injection patterns.
/// 4. Optionally computes & verifies full streaming SHA-256 integrity.
pub fn scan_gguf_model(
    path: &Path,
    compute_sha256: bool,
    expected_sha256: Option<&str>,
) -> Result<ModelScanReport> {
    let started = std::time::Instant::now();
    let meta = std::fs::metadata(path)
        .with_context(|| format!("File model tidak ditemukan: {}", path.display()))?;
    if !meta.is_file() {
        anyhow::bail!("Path bukan file reguler: {}", path.display());
    }
    let file_size = meta.len();
    let mut findings: Vec<ScanFinding> = Vec::new();

    let mut report = ModelScanReport {
        path: path.display().to_string(),
        file_size_bytes: file_size,
        verdict: ScanVerdict::Pass,
        safe_to_load: true,
        gguf_version: None,
        tensor_count: None,
        metadata_kv_count: None,
        architecture: None,
        model_name: None,
        author: None,
        license: None,
        chat_template_len: None,
        sha256: None,
        sha256_verified: None,
        scan_duration_ms: 0,
        findings: Vec::new(),
    };

    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);

    // 1. Magic bytes check
    let mut magic = [0u8; 4];
    if file_size < 24 || reader.read_exact(&mut magic).is_err() {
        findings.push(ScanFinding {
            severity: ScanSeverity::Critical,
            code: "INVALID_GGUF_HEADER".to_string(),
            message: "File terlalu kecil atau tidak memiliki header GGUF yang valid.".to_string(),
        });
    } else if &magic != GGUF_MAGIC {
        if let Some(disguised) = detect_disguised_format(&magic) {
            findings.push(ScanFinding {
                severity: ScanSeverity::Critical,
                code: "DISGUISED_EXECUTABLE_OR_PICKLE".to_string(),
                message: format!(
                    "BAHAYA: File bukan GGUF, terdeteksi sebagai {disguised}!"
                ),
            });
        } else {
            findings.push(ScanFinding {
                severity: ScanSeverity::Critical,
                code: "INVALID_GGUF_HEADER".to_string(),
                message: format!(
                    "Magic bytes bukan 'GGUF' (ditemukan: {:02x?}).",
                    magic
                ),
            });
        }
    } else {
        // 2. Version, tensor_count, metadata_kv_count
        let version = read_u32(&mut reader)?;
        report.gguf_version = Some(version);
        if version != 2 && version != 3 {
            findings.push(ScanFinding {
                severity: ScanSeverity::Critical,
                code: "UNSUPPORTED_GGUF_VERSION".to_string(),
                message: format!(
                    "Versi GGUF tidak didukung atau korup: v{version} (hanya v2/v3)."
                ),
            });
        } else {
            let tensor_count = read_u64(&mut reader)?;
            let kv_count = read_u64(&mut reader)?;
            report.tensor_count = Some(tensor_count);
            report.metadata_kv_count = Some(kv_count);

            if tensor_count > MAX_SANE_TENSORS || kv_count > MAX_SANE_KV {
                findings.push(ScanFinding {
                    severity: ScanSeverity::Critical,
                    code: "HEADER_COUNT_OVERFLOW".to_string(),
                    message: format!(
                        "Jumlah tensor ({tensor_count}) atau metadata KV ({kv_count}) melampaui batas aman."
                    ),
                });
            } else {
                // 3. Parse & audit metadata KVs
                let mut alignment: u64 = 32;
                let mut parse_failed = false;
                for _ in 0..kv_count {
                    let key = match read_gguf_string(&mut reader) {
                        Ok(k) => k,
                        Err(err) => {
                            findings.push(ScanFinding {
                                severity: ScanSeverity::Critical,
                                code: "CORRUPT_METADATA".to_string(),
                                message: format!("Metadata GGUF rusak: {err}"),
                            });
                            parse_failed = true;
                            break;
                        }
                    };
                    let vtype = match read_u32(&mut reader) {
                        Ok(t) => t,
                        Err(err) => {
                            findings.push(ScanFinding {
                                severity: ScanSeverity::Critical,
                                code: "CORRUPT_METADATA".to_string(),
                                message: format!("Tipe metadata rusak pada key '{key}': {err}"),
                            });
                            parse_failed = true;
                            break;
                        }
                    };
                    if key == "general.alignment" && vtype == 4 {
                        if let Ok(a) = read_u32(&mut reader) {
                            if a >= 1 && a <= 4096 {
                                alignment = a as u64;
                            }
                        }
                        continue;
                    }
                    match skip_gguf_value(&mut reader, vtype) {
                        Ok(Some(val)) => match key.as_str() {
                            "general.architecture" => report.architecture = Some(val),
                            "general.name" => report.model_name = Some(val),
                            "general.author" => report.author = Some(val),
                            "general.license" => report.license = Some(val),
                            "tokenizer.chat_template" => {
                                report.chat_template_len = Some(val.len());
                                audit_chat_template(&val, &mut findings);
                            }
                            _ => {}
                        },
                        Ok(None) => {}
                        Err(err) => {
                            findings.push(ScanFinding {
                                severity: ScanSeverity::Critical,
                                code: "CORRUPT_METADATA".to_string(),
                                message: format!("Nilai metadata rusak pada key '{key}': {err}"),
                            });
                            parse_failed = true;
                            break;
                        }
                    }
                }

                // 4. Validate tensor table & offsets against file_size
                if !parse_failed {
                    let mut max_rel_offset: u64 = 0;
                    for idx in 0..tensor_count {
                        let _tname = match read_gguf_string(&mut reader) {
                            Ok(s) => s,
                            Err(err) => {
                                findings.push(ScanFinding {
                                    severity: ScanSeverity::Critical,
                                    code: "CORRUPT_TENSOR_TABLE".to_string(),
                                    message: format!("Nama tensor #{idx} rusak: {err}"),
                                });
                                parse_failed = true;
                                break;
                            }
                        };
                        let n_dims = match read_u32(&mut reader) {
                            Ok(d) => d,
                            Err(err) => {
                                findings.push(ScanFinding {
                                    severity: ScanSeverity::Critical,
                                    code: "CORRUPT_TENSOR_TABLE".to_string(),
                                    message: format!("Dimensi tensor #{idx} rusak: {err}"),
                                });
                                parse_failed = true;
                                break;
                            }
                        };
                        if n_dims > 8 {
                            findings.push(ScanFinding {
                                severity: ScanSeverity::Critical,
                                code: "INVALID_TENSOR_DIMS".to_string(),
                                message: format!(
                                    "Tensor #{idx} memiliki n_dims={n_dims} (> 8, tidak wajar)."
                                ),
                            });
                            parse_failed = true;
                            break;
                        }
                        for _ in 0..n_dims {
                            let dim = match read_u64(&mut reader) {
                                Ok(v) => v,
                                Err(_) => {
                                    parse_failed = true;
                                    break;
                                }
                            };
                            if dim == 0 || dim > 100_000_000 {
                                findings.push(ScanFinding {
                                    severity: ScanSeverity::Critical,
                                    code: "INVALID_TENSOR_DIMS".to_string(),
                                    message: format!(
                                        "Tensor #{idx} memiliki ukuran dimensi tidak wajar ({dim})."
                                    ),
                                });
                                parse_failed = true;
                                break;
                            }
                        }
                        if parse_failed {
                            break;
                        }
                        let ggml_type = read_u32(&mut reader).unwrap_or(u32::MAX);
                        if ggml_type > 64 {
                            findings.push(ScanFinding {
                                severity: ScanSeverity::Critical,
                                code: "INVALID_TENSOR_TYPE".to_string(),
                                message: format!(
                                    "Tensor #{idx} memakai ggml_type tidak dikenal ({ggml_type})."
                                ),
                            });
                            parse_failed = true;
                            break;
                        }
                        let rel_offset = match read_u64(&mut reader) {
                            Ok(o) => o,
                            Err(_) => {
                                parse_failed = true;
                                break;
                            }
                        };
                        if rel_offset > max_rel_offset {
                            max_rel_offset = rel_offset;
                        }
                    }

                    if !parse_failed {
                        let header_end = reader.stream_position().unwrap_or(0);
                        let data_start = header_end.div_ceil(alignment) * alignment;
                        if data_start.saturating_add(max_rel_offset) > file_size {
                            findings.push(ScanFinding {
                                severity: ScanSeverity::Critical,
                                code: "TENSOR_OUT_OF_BOUNDS".to_string(),
                                message: format!(
                                    "Offset tensor melampaui ukuran file ({} > {} byte) — berpotensi truncated atau crafted exploit.",
                                    data_start.saturating_add(max_rel_offset),
                                    file_size
                                ),
                            });
                        }
                    }
                }
            }
        }
    }

    // 5. Optional cryptographic SHA-256 verification
    if compute_sha256 || expected_sha256.is_some() {
        let actual = compute_file_sha256(path)?;
        if let Some(expected) = expected_sha256 {
            let expected_clean = expected.trim().to_ascii_lowercase();
            let matched = actual == expected_clean;
            report.sha256_verified = Some(matched);
            if !matched {
                findings.push(ScanFinding {
                    severity: ScanSeverity::Critical,
                    code: "SHA256_MISMATCH".to_string(),
                    message: format!(
                        "Integritas SHA-256 TIDAK COCOK! Diharapkan: {expected_clean}, Aktual: {actual}"
                    ),
                });
            }
        }
        report.sha256 = Some(actual);
    }

    let has_critical = findings.iter().any(|f| f.severity == ScanSeverity::Critical);
    let has_warning = findings.iter().any(|f| f.severity == ScanSeverity::Warning);
    report.verdict = if has_critical {
        ScanVerdict::Block
    } else if has_warning {
        ScanVerdict::Warn
    } else {
        ScanVerdict::Pass
    };
    report.safe_to_load = !has_critical;
    report.findings = findings;
    report.scan_duration_ms = started.elapsed().as_millis();

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_gguf_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_synthetic_gguf(chat_template: Option<&str>, bad_tensor_offset: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // v3
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        let kv_count: u64 = if chat_template.is_some() { 2 } else { 1 };
        buf.extend_from_slice(&kv_count.to_le_bytes());

        // KV 1: general.architecture = "llama"
        write_gguf_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_gguf_str(&mut buf, "llama");

        if let Some(tmpl) = chat_template {
            write_gguf_str(&mut buf, "tokenizer.chat_template");
            buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
            write_gguf_str(&mut buf, tmpl);
        }

        // Tensor 0
        write_gguf_str(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        let offset = if bad_tensor_offset { 999_999u64 } else { 0u64 };
        buf.extend_from_slice(&offset.to_le_bytes());

        // Pad to 32-byte alignment + 64 bytes dummy tensor data
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&[0u8; 64]);
        buf
    }

    #[test]
    fn passes_clean_gguf_and_verifies_sha256() {
        let dir = std::env::temp_dir().join(format!("cmod_scan_ok_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("clean.gguf");
        std::fs::write(&file, build_synthetic_gguf(Some("{% for m in messages %}{{ m.content }}{% endfor %}"), false)).unwrap();

        let report = scan_gguf_model(&file, true, None).unwrap();
        assert_eq!(report.verdict, ScanVerdict::Pass);
        assert!(report.safe_to_load);
        assert_eq!(report.architecture.as_deref(), Some("llama"));
        let sha = report.sha256.clone().unwrap();

        let verified = scan_gguf_model(&file, false, Some(&sha)).unwrap();
        assert_eq!(verified.verdict, ScanVerdict::Pass);
        assert_eq!(verified.sha256_verified, Some(true));

        let mismatch = scan_gguf_model(&file, false, Some("0000deadbeef")).unwrap();
        assert_eq!(mismatch.verdict, ScanVerdict::Block);
        assert!(!mismatch.safe_to_load);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn blocks_disguised_pickle_and_ssti_template_and_oob_tensor() {
        let dir = std::env::temp_dir().join(format!("cmod_scan_bad_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        // 1. Python pickle disguised as .gguf
        let pickle = dir.join("evil_pickle.gguf");
        std::fs::write(&pickle, [0x80u8, 0x04, 0x95, 0x20, 0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let r1 = scan_gguf_model(&pickle, false, None).unwrap();
        assert_eq!(r1.verdict, ScanVerdict::Block);
        assert!(r1.findings.iter().any(|f| f.code == "DISGUISED_EXECUTABLE_OR_PICKLE"));

        // 2. Malicious SSTI chat_template
        let ssti = dir.join("ssti.gguf");
        std::fs::write(&ssti, build_synthetic_gguf(Some("{{ ''.__class__.__subclasses__() }}"), false)).unwrap();
        let r2 = scan_gguf_model(&ssti, false, None).unwrap();
        assert_eq!(r2.verdict, ScanVerdict::Block);
        assert!(r2.findings.iter().any(|f| f.code == "TEMPLATE_SSTI_PATTERN"));

        // 3. Out-of-bounds tensor offset
        let oob = dir.join("oob.gguf");
        std::fs::write(&oob, build_synthetic_gguf(None, true)).unwrap();
        let r3 = scan_gguf_model(&oob, false, None).unwrap();
        assert_eq!(r3.verdict, ScanVerdict::Block);
        assert!(r3.findings.iter().any(|f| f.code == "TENSOR_OUT_OF_BOUNDS"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
