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
    scheduler: Arc<PriorityScheduler>,
}

impl Drop for LanePermit {
    fn drop(&mut self) {
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
    parallel_slots: u32,
}

impl PriorityScheduler {
    pub fn new(cfg: &SchedulerConfig, parallel_slots: u32) -> Arc<Self> {
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
            parallel_slots: parallel_slots.max(1),
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
