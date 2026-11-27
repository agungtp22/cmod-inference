use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::SchedulerConfig;
use crate::ir::PriorityLane;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SchedulerStatus {
    pub interactive_slots: usize,
    pub batch_slots: usize,
    pub interactive_active: usize,
    pub interactive_queued: usize,
    pub batch_active: usize,
    pub batch_queued: usize,
    pub total_completed: u64,
    pub total_rejected: u64,
    pub tracked_prefixes: usize,
}

pub struct LanePermit {
    _permit: OwnedSemaphorePermit,
    lane: PriorityLane,
    assigned_slot: Option<u32>,
    scheduler: Arc<PriorityScheduler>,
}

impl Drop for LanePermit {
    fn drop(&mut self) {
        if let Some(slot) = self.assigned_slot {
            if let Some(counter) = self.scheduler.slot_in_flight.get(slot as usize) {
                counter.fetch_sub(1, Ordering::AcqRel);
            }
        }
        match self.lane {
            PriorityLane::Interactive => {
                self.scheduler
                    .interactive_active
                    .fetch_sub(1, Ordering::Relaxed);
            }
            PriorityLane::Batch => {
                self.scheduler.batch_active.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.scheduler
            .total_completed
            .fetch_add(1, Ordering::Relaxed);
    }
}

pub struct PriorityScheduler {
    interactive_sem: Arc<Semaphore>,
    batch_sem: Arc<Semaphore>,
    interactive_slots: usize,
    batch_slots: usize,
    queue_timeout: Duration,
    interactive_active: AtomicUsize,
    interactive_queued: AtomicUsize,
    batch_active: AtomicUsize,
    batch_queued: AtomicUsize,
    total_completed: AtomicU64,
    total_rejected: AtomicU64,
    prefix_slots: Mutex<HashMap<String, u32>>,
    slot_in_flight: Vec<AtomicUsize>,
    slot_last_prefix: Mutex<Vec<Option<String>>>,
    slot_rr: AtomicUsize,
    parallel_slots: u32,
}

impl PriorityScheduler {
    pub fn new(cfg: &SchedulerConfig, parallel_slots: u32) -> Arc<Self> {
        let slots = parallel_slots.max(1);
        let slot_in_flight = (0..slots).map(|_| AtomicUsize::new(0)).collect();
        let slot_last_prefix = vec![None; slots as usize];
        Arc::new(Self {
            interactive_sem: Arc::new(Semaphore::new(cfg.interactive_slots)),
            batch_sem: Arc::new(Semaphore::new(cfg.batch_slots)),
            interactive_slots: cfg.interactive_slots,
            batch_slots: cfg.batch_slots,
            queue_timeout: Duration::from_secs(cfg.queue_timeout_secs.max(1)),
            interactive_active: AtomicUsize::new(0),
            interactive_queued: AtomicUsize::new(0),
            batch_active: AtomicUsize::new(0),
            batch_queued: AtomicUsize::new(0),
            total_completed: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            prefix_slots: Mutex::new(HashMap::new()),
            slot_in_flight,
            slot_last_prefix: Mutex::new(slot_last_prefix),
            slot_rr: AtomicUsize::new(0),
            parallel_slots: slots,
        })
    }

    /// Acquire a concurrency slot for the requested priority lane.
    /// Interactive requests are isolated from batch congestion so UI chat never stalls.
    pub async fn acquire(
        self: &Arc<Self>,
        lane: PriorityLane,
    ) -> Result<LanePermit, String> {
        let (sem, queued_counter, active_counter) = match lane {
            PriorityLane::Interactive => (
                self.interactive_sem.clone(),
                &self.interactive_queued,
                &self.interactive_active,
            ),
            PriorityLane::Batch => (
                self.batch_sem.clone(),
                &self.batch_queued,
                &self.batch_active,
            ),
        };

        queued_counter.fetch_add(1, Ordering::Relaxed);
        let acquired = tokio::time::timeout(self.queue_timeout, sem.acquire_owned()).await;
        queued_counter.fetch_sub(1, Ordering::Relaxed);

        match acquired {
            Ok(Ok(permit)) => {
                active_counter.fetch_add(1, Ordering::Relaxed);
                Ok(LanePermit {
                    _permit: permit,
                    lane,
                    assigned_slot: None,
                    scheduler: self.clone(),
                })
            }
            Ok(Err(_)) => {
                self.total_rejected.fetch_add(1, Ordering::Relaxed);
                Err("Scheduler ditutup".to_string())
            }
            Err(_) => {
                self.total_rejected.fetch_add(1, Ordering::Relaxed);
                Err(format!(
                    "Antrean inferensi penuh (timeout {}s untuk jalur {:?})",
                    self.queue_timeout.as_secs(),
                    lane
                ))
            }
        }
    }

    /// Idle-aware prefix affinity slot acquisition:
    /// 1. Prefer an IDLE slot whose last prefix matches `prefix_hash` (zero-copy L1 VRAM hit).
    /// 2. Otherwise pick any IDLE slot (uninitialized first, then round-robin) so concurrent
    ///    requests with the same system prompt run truly in parallel across all VRAM slots
    ///    while warming instantly from L2 Host RAM KV-Cache.
    /// 3. If all slots are busy, pick the slot with the smallest in-flight queue (tie-breaking
    ///    on matching prefix).
    pub async fn acquire_with_slot(
        self: &Arc<Self>,
        lane: PriorityLane,
        prefix_hash: &str,
    ) -> Result<(LanePermit, u32), String> {
        let mut permit = self.acquire(lane).await?;
        let slot = self.claim_idle_aware_slot(prefix_hash);
        permit.assigned_slot = Some(slot);
        Ok((permit, slot))
    }

    fn claim_idle_aware_slot(&self, prefix_hash: &str) -> u32 {
        let n = self.parallel_slots as usize;
        let mut last_prefixes = self
            .slot_last_prefix
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // 1. Prefer an idle slot that already holds this exact prefix in L1 VRAM.
        let mut chosen = (0..n).find(|&i| {
            self.slot_in_flight[i].load(Ordering::Acquire) == 0
                && last_prefixes[i].as_deref() == Some(prefix_hash)
        });

        // 2. Otherwise prefer an idle slot that hasn't been used yet, then round-robin idle slots.
        if chosen.is_none() {
            chosen = (0..n).find(|&i| {
                self.slot_in_flight[i].load(Ordering::Acquire) == 0 && last_prefixes[i].is_none()
            });
        }
        if chosen.is_none() {
            let start = self.slot_rr.fetch_add(1, Ordering::Relaxed) % n;
            chosen = (0..n)
                .map(|off| (start + off) % n)
                .find(|&i| self.slot_in_flight[i].load(Ordering::Acquire) == 0);
        }

        // 3. All slots busy: choose least-loaded slot, preferring prefix match on ties.
        let idx = chosen.unwrap_or_else(|| {
            (0..n)
                .min_by_key(|&i| {
                    let load = self.slot_in_flight[i].load(Ordering::Acquire);
                    let miss_penalty = usize::from(last_prefixes[i].as_deref() != Some(prefix_hash));
                    (load, miss_penalty)
                })
                .unwrap_or(0)
        });

        self.slot_in_flight[idx].fetch_add(1, Ordering::AcqRel);
        last_prefixes[idx] = Some(prefix_hash.to_string());
        drop(last_prefixes);

        if let Ok(mut map) = self.prefix_slots.lock() {
            if map.len() > 2048 {
                map.clear();
            }
            map.insert(prefix_hash.to_string(), idx as u32);
        }
        idx as u32
    }

    /// Sticky slot assignment for KV/prefix cache reuse on llama-server backends.
    pub fn assign_slot(&self, prefix_hash: &str) -> u32 {
        let mut map = self.prefix_slots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&slot) = map.get(prefix_hash) {
            return slot;
        }
        if map.len() > 2048 {
            map.clear();
        }
        let next_slot = (map.len() as u32) % self.parallel_slots;
        map.insert(prefix_hash.to_string(), next_slot);
        next_slot
    }

    pub fn status(&self) -> SchedulerStatus {
        let tracked_prefixes = self
            .prefix_slots
            .lock()
            .map(|m| m.len())
            .unwrap_or(0);
        SchedulerStatus {
            interactive_slots: self.interactive_slots,
            batch_slots: self.batch_slots,
            interactive_active: self.interactive_active.load(Ordering::Relaxed),
            interactive_queued: self.interactive_queued.load(Ordering::Relaxed),
            batch_active: self.batch_active.load(Ordering::Relaxed),
            batch_queued: self.batch_queued.load(Ordering::Relaxed),
            total_completed: self.total_completed.load(Ordering::Relaxed),
            total_rejected: self.total_rejected.load(Ordering::Relaxed),
            tracked_prefixes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn idle_aware_prefix_affinity_spreads_concurrent_same_prefix_and_reuses_idle() {
        let cfg = SchedulerConfig {
            interactive_slots: 4,
            batch_slots: 4,
            queue_timeout_secs: 5,
        };
        let sched = PriorityScheduler::new(&cfg, 2);

        // Two concurrent requests with the EXACT same SYSTEM_PROMPT hash must run on
        // distinct slots (0 and 1) rather than serializing on slot 0.
        let (p1, slot1) = sched
            .acquire_with_slot(PriorityLane::Interactive, "same_sys_prompt")
            .await
            .expect("slot 1");
        let (p2, slot2) = sched
            .acquire_with_slot(PriorityLane::Interactive, "same_sys_prompt")
            .await
            .expect("slot 2");
        assert_ne!(slot1, slot2, "concurrent requests must spread across idle slots");

        // Release slot1; next request with the same prefix must immediately reuse slot1.
        drop(p1);
        let (p3, slot3) = sched
            .acquire_with_slot(PriorityLane::Interactive, "same_sys_prompt")
            .await
            .expect("slot 3");
        assert_eq!(slot3, slot1, "idle slot with matching prefix should be reused");

        drop(p2);
        drop(p3);
    }
}
