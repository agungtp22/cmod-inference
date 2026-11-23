use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KvCacheStats {
    pub l1_vram_hits: u64,
    pub l2_ram_budget_mb: u64,
    pub l2_ram_used_bytes: u64,
    pub l2_entries: usize,
    pub l2_ram_hits: u64,
    pub l3_disk_dir: Option<String>,
    pub l3_disk_budget_mb: u64,
    pub l3_disk_files: usize,
    pub l3_disk_hits: u64,
    pub misses: u64,
    pub ram_evictions: u64,
    pub disk_evictions: u64,
}

/// Compute a deterministic model+config fingerprint so stale `.kvcache` snapshots
/// are never loaded after swapping GGUF files, changing KV quantization, or changing `n_ctx`.
pub fn compute_model_fingerprint(
    model_path: &str,
    kv_cache_quant: &str,
    max_model_len: u32,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model_path.as_bytes());
    hasher.update(b"|");
    hasher.update(kv_cache_quant.as_bytes());
    hasher.update(b"|");
    hasher.update(max_model_len.to_le_bytes());

    if let Ok(meta) = std::fs::metadata(model_path) {
        hasher.update(meta.len().to_le_bytes());
        if let Ok(mtime) = meta.modified() {
            if let Ok(dur) = mtime.duration_since(UNIX_EPOCH) {
                hasher.update(dur.as_secs().to_le_bytes());
            }
        }
    }

    let digest = hasher.finalize();
    format!("{:x}", digest)[..12].to_string()
}

struct LruEntry<T> {
    value: T,
    byte_len: usize,
}

pub struct RamLruCache<T> {
    max_bytes: usize,
    used_bytes: usize,
    map: HashMap<String, LruEntry<T>>,
    order: VecDeque<String>,
    evictions: u64,
}

impl<T: Clone> RamLruCache<T> {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            map: HashMap::new(),
            order: VecDeque::new(),
            evictions: 0,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<T> {
        let val = self.map.get(key).map(|e| e.value.clone())?;
        self.touch(key);
        Some(val)
    }

    pub fn put(&mut self, key: String, value: T, byte_len: usize) {
        if self.max_bytes == 0 || byte_len > self.max_bytes {
            return;
        }
        if let Some(old) = self.map.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.byte_len);
            self.order.retain(|k| k != &key);
        }
        while self.used_bytes + byte_len > self.max_bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(removed) = self.map.remove(&oldest) {
                self.used_bytes = self.used_bytes.saturating_sub(removed.byte_len);
                self.evictions += 1;
            }
        }
        self.used_bytes += byte_len;
        self.order.push_back(key.clone());
        self.map.insert(key, LruEntry { value, byte_len });
    }

    fn touch(&mut self, key: &str) {
        self.order.retain(|k| k != key);
        self.order.push_back(key.to_string());
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }
}

#[cfg(feature = "native")]
#[derive(Clone)]
pub struct NativeKvSnapshot {
    pub state: llama_cpp_2::context::session::SeqState,
    pub tokens: Vec<llama_cpp_2::token::LlamaToken>,
}

/// 3-Tier KV-Cache Coordinator:
/// - L1: Active GPU VRAM (tracked per persistent worker LlamaContext)
/// - L2: Host RAM LRU (`SeqState` in memory, millisecond restore)
/// - L3: Local NVMe Disk (`.kvcache` files with automatic LRU eviction)
pub struct TieredKvCache {
    l2_ram_budget_mb: u64,
    l3_disk_budget_mb: u64,
    l3_disk_dir: Option<PathBuf>,
    l1_vram_hits: AtomicU64,
    l2_ram_hits: AtomicU64,
    l3_disk_hits: AtomicU64,
    misses: AtomicU64,
    disk_evictions: AtomicU64,
    #[cfg(feature = "native")]
    ram_native: Mutex<RamLruCache<NativeKvSnapshot>>,
    #[cfg(not(feature = "native"))]
    ram_bytes: Mutex<RamLruCache<Vec<u8>>>,
}

impl TieredKvCache {
    pub fn new(ram_budget_mb: u64, disk_budget_mb: u64, disk_dir: Option<PathBuf>) -> Arc<Self> {
        if let Some(ref dir) = disk_dir {
            let _ = std::fs::create_dir_all(dir);
        }
        let ram_bytes_cap = (ram_budget_mb as usize).saturating_mul(1024 * 1024);
        Arc::new(Self {
            l2_ram_budget_mb: ram_budget_mb,
            l3_disk_budget_mb: disk_budget_mb,
            l3_disk_dir: disk_dir,
            l1_vram_hits: AtomicU64::new(0),
            l2_ram_hits: AtomicU64::new(0),
            l3_disk_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            disk_evictions: AtomicU64::new(0),
            #[cfg(feature = "native")]
            ram_native: Mutex::new(RamLruCache::new(ram_bytes_cap)),
            #[cfg(not(feature = "native"))]
            ram_bytes: Mutex::new(RamLruCache::new(ram_bytes_cap)),
        })
    }

    pub fn disk_path_for_key(&self, key: &str) -> Option<PathBuf> {
        let dir = self.l3_disk_dir.as_ref()?;
        let safe: String = key
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        if safe.is_empty() {
            return None;
        }
        Some(dir.join(format!("{safe}.kvcache")))
    }

    pub fn record_l1_hit(&self) {
        self.l1_vram_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_l2_hit(&self) {
        self.l2_ram_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_l3_hit(&self) {
        self.l3_disk_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Enforce L3 NVMe disk budget by deleting oldest `.kvcache` files first (LRU by mtime).
    pub fn enforce_disk_budget(&self) -> usize {
        let Some(ref dir) = self.l3_disk_dir else {
            return 0;
        };
        let budget_bytes = self.l3_disk_budget_mb.saturating_mul(1024 * 1024);
        if budget_bytes == 0 {
            return 0;
        }
        let evicted = enforce_dir_budget_bytes(dir, budget_bytes);
        if evicted > 0 {
            self.disk_evictions
                .fetch_add(evicted as u64, Ordering::Relaxed);
        }
        evicted
    }

    #[cfg(feature = "native")]
    pub fn get_ram(&self, key: &str) -> Option<NativeKvSnapshot> {
        let mut guard = self.ram_native.lock().unwrap_or_else(|e| e.into_inner());
        guard.get(key)
    }

    #[cfg(feature = "native")]
    pub fn put_ram(&self, key: String, snap: NativeKvSnapshot) {
        let byte_len = snap.state.byte_len();
        let mut guard = self.ram_native.lock().unwrap_or_else(|e| e.into_inner());
        guard.put(key, snap, byte_len);
    }

    pub fn stats(&self) -> KvCacheStats {
        let (l2_used, l2_entries, ram_evictions) = {
            #[cfg(feature = "native")]
            {
                let g = self.ram_native.lock().unwrap_or_else(|e| e.into_inner());
                (g.used_bytes() as u64, g.len(), g.evictions())
            }
            #[cfg(not(feature = "native"))]
            {
                let g = self.ram_bytes.lock().unwrap_or_else(|e| e.into_inner());
                (g.used_bytes() as u64, g.len(), g.evictions())
            }
        };

        let l3_files = self
            .l3_disk_dir
            .as_deref()
            .map(count_kv_files)
            .unwrap_or(0);

        KvCacheStats {
            l1_vram_hits: self.l1_vram_hits.load(Ordering::Relaxed),
            l2_ram_budget_mb: self.l2_ram_budget_mb,
            l2_ram_used_bytes: l2_used,
            l2_entries,
            l2_ram_hits: self.l2_ram_hits.load(Ordering::Relaxed),
            l3_disk_dir: self
                .l3_disk_dir
                .as_ref()
                .map(|p| p.display().to_string()),
            l3_disk_budget_mb: self.l3_disk_budget_mb,
            l3_disk_files: l3_files,
            l3_disk_hits: self.l3_disk_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            ram_evictions,
            disk_evictions: self.disk_evictions.load(Ordering::Relaxed),
        }
    }
}

fn enforce_dir_budget_bytes(dir: &Path, budget_bytes: u64) -> usize {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut files: Vec<(PathBuf, u64, u128)> = Vec::new();
    let mut total: u64 = 0;

    for entry in rd.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("kvcache") {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            let len = meta.len();
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            total = total.saturating_add(len);
            files.push((path, len, mtime_ns));
        }
    }

    if total <= budget_bytes {
        return 0;
    }

    files.sort_by_key(|(_, _, mtime)| *mtime);
    let mut evicted = 0usize;
    for (path, len, _) in files {
        if total <= budget_bytes {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
            evicted += 1;
        }
    }
    evicted
}

fn count_kv_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        == Some("kvcache")
                })
                .count()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_lru_evicts_oldest_and_respects_budget() {
        let mut lru = RamLruCache::<Vec<u8>>::new(100);
        lru.put("doc-a".to_string(), vec![1u8; 60], 60);
        lru.put("doc-b".to_string(), vec![2u8; 60], 60);
        assert!(lru.get("doc-a").is_none());
        assert_eq!(lru.get("doc-b").unwrap().len(), 60);
        assert_eq!(lru.evictions(), 1);
    }

    #[test]
    fn disk_lru_evicts_oldest_kvcache_files() {
        let tmp = std::env::temp_dir().join(format!("cmod_kv_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        let f1 = tmp.join("a.kvcache");
        let f2 = tmp.join("b.kvcache");
        std::fs::write(&f1, vec![0u8; 1024]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&f2, vec![0u8; 1024]).unwrap();

        let evicted = enforce_dir_budget_bytes(&tmp, 1200);
        assert_eq!(evicted, 1);
        assert!(!f1.exists());
        assert!(f2.exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
