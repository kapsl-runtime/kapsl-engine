use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(30);
const DEFAULT_CLASS_SLOT_SIZES: [usize; 5] = [
    256 * 1024,       // 256 KiB
    1024 * 1024,      // 1 MiB
    4 * 1024 * 1024,  // 4 MiB
    16 * 1024 * 1024, // 16 MiB
    64 * 1024 * 1024, // 64 MiB
];

#[derive(Debug, Clone, Copy, Default)]
pub struct ShmAllocatorSnapshot {
    pub in_use_slots: usize,
    pub oldest_lease_ms: u64,
}

pub trait ShmPoolAllocator: Send + Sync {
    fn try_allocate(&self, required_size: usize) -> Option<usize>;
    fn release(&self, offset: usize) -> bool;
    fn snapshot(&self) -> ShmAllocatorSnapshot;
    fn layout_summary(&self) -> String;
    fn largest_slot_size(&self) -> usize;
}

/// A slot-based allocator for shared memory tensor payloads.
///
/// Each slot is leased for a bounded TTL so concurrent writers avoid
/// immediately clobbering in-flight responses.
pub struct SimpleShmAllocator {
    base_offset: usize,
    slot_size: usize,
    num_slots: usize,
    counter: AtomicUsize,
    lease_ttl: Duration,
    // Per-slot lease expiration; `None` means currently free.
    leases: Mutex<Vec<Option<Instant>>>,
}

impl SimpleShmAllocator {
    pub fn new(base_offset: usize, slot_size: usize, num_slots: usize) -> Self {
        Self::new_with_ttl(base_offset, slot_size, num_slots, DEFAULT_LEASE_TTL)
    }

    pub fn new_with_ttl(
        base_offset: usize,
        slot_size: usize,
        num_slots: usize,
        lease_ttl: Duration,
    ) -> Self {
        let slot_size = slot_size.max(1);
        let num_slots = num_slots.max(1);
        Self {
            base_offset,
            slot_size,
            num_slots,
            counter: AtomicUsize::new(0),
            lease_ttl,
            leases: Mutex::new(vec![None; num_slots]),
        }
    }

    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Best-effort legacy API.
    ///
    /// Prefer `try_allocate(required_size)` so callers can handle pool
    /// exhaustion without overwriting in-flight slots.
    pub fn allocate(&self) -> usize {
        if let Some(offset) = self.try_allocate(self.slot_size) {
            return offset;
        }
        // Fallback for legacy callers: preserve previous round-robin behavior.
        let slot = self.counter.fetch_add(1, Ordering::Relaxed) % self.num_slots;
        self.base_offset + slot * self.slot_size
    }

    /// Try to lease a slot that can hold `required_size` bytes.
    pub fn try_allocate(&self, required_size: usize) -> Option<usize> {
        if required_size == 0 || required_size > self.slot_size {
            return None;
        }

        let now = Instant::now();
        let start = self.counter.fetch_add(1, Ordering::Relaxed) % self.num_slots;
        let mut leases = self
            .leases
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        for i in 0..self.num_slots {
            let idx = (start + i) % self.num_slots;
            let reusable = leases[idx].map(|deadline| deadline <= now).unwrap_or(true);
            if reusable {
                leases[idx] = Some(now + self.lease_ttl);
                return Some(self.base_offset + idx * self.slot_size);
            }
        }

        None
    }

    /// Explicitly release a slot by offset.
    ///
    /// Returns `true` when the offset matches a slot start and release succeeds.
    pub fn release(&self, offset: usize) -> bool {
        if offset < self.base_offset {
            return false;
        }
        let rel = offset - self.base_offset;
        if !rel.is_multiple_of(self.slot_size) {
            return false;
        }
        let idx = rel / self.slot_size;
        if idx >= self.num_slots {
            return false;
        }

        let mut leases = self
            .leases
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        leases[idx] = None;
        true
    }

    /// Snapshot allocator occupancy and lease age.
    ///
    /// `in_use_slots` excludes expired leases.
    /// `oldest_lease_ms` is the age of the oldest non-expired lease.
    pub fn snapshot(&self) -> ShmAllocatorSnapshot {
        let now = Instant::now();
        let leases = self
            .leases
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let mut in_use_slots = 0usize;
        let mut oldest_age = Duration::from_millis(0);
        for lease in leases.iter().copied().flatten() {
            if lease <= now {
                continue;
            }
            in_use_slots += 1;
            let remaining = lease.saturating_duration_since(now);
            let age = self.lease_ttl.saturating_sub(remaining);
            if age > oldest_age {
                oldest_age = age;
            }
        }

        ShmAllocatorSnapshot {
            in_use_slots,
            oldest_lease_ms: oldest_age.as_millis().min(u64::MAX as u128) as u64,
        }
    }
}

impl ShmPoolAllocator for SimpleShmAllocator {
    fn try_allocate(&self, required_size: usize) -> Option<usize> {
        SimpleShmAllocator::try_allocate(self, required_size)
    }

    fn release(&self, offset: usize) -> bool {
        SimpleShmAllocator::release(self, offset)
    }

    fn snapshot(&self) -> ShmAllocatorSnapshot {
        SimpleShmAllocator::snapshot(self)
    }

    fn layout_summary(&self) -> String {
        format!(
            "single(base={} slot_size={} slots={})",
            self.base_offset, self.slot_size, self.num_slots
        )
    }

    fn largest_slot_size(&self) -> usize {
        self.slot_size
    }
}

struct ShmAllocatorClass {
    name: String,
    base_offset: usize,
    slot_size: usize,
    num_slots: usize,
    bytes: usize,
    allocator: SimpleShmAllocator,
}

#[derive(Debug, Clone)]
pub struct ShmAllocatorClassInfo {
    pub name: String,
    pub base_offset: usize,
    pub slot_size: usize,
    pub num_slots: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShmClassBudget {
    pub slot_size: usize,
    pub weight: u32,
}

/// Tiered slab allocator built from multiple fixed-size slot pools.
///
/// Allocation selects the smallest slot class that can hold `required_size`,
/// then falls back to larger classes if the preferred class is temporarily full.
pub struct TieredShmAllocator {
    classes: Vec<ShmAllocatorClass>,
}

impl TieredShmAllocator {
    pub fn new_with_default_classes(
        base_offset: usize,
        pool_bytes: usize,
        lease_ttl: Duration,
    ) -> Self {
        Self::new_with_class_sizes(
            base_offset,
            pool_bytes,
            &DEFAULT_CLASS_SLOT_SIZES,
            lease_ttl,
        )
    }

    pub fn new(base_offset: usize, pool_bytes: usize) -> Self {
        Self::new_with_default_classes(base_offset, pool_bytes, DEFAULT_LEASE_TTL)
    }

    pub fn new_with_class_sizes(
        base_offset: usize,
        pool_bytes: usize,
        class_slot_sizes: &[usize],
        lease_ttl: Duration,
    ) -> Self {
        let budgets: Vec<ShmClassBudget> = class_slot_sizes
            .iter()
            .copied()
            .filter(|v| *v > 0)
            .map(|slot_size| ShmClassBudget {
                slot_size,
                weight: 1,
            })
            .collect();
        Self::new_with_class_budgets(base_offset, pool_bytes, &budgets, lease_ttl)
    }

    pub fn new_with_class_budgets(
        base_offset: usize,
        pool_bytes: usize,
        class_budgets: &[ShmClassBudget],
        lease_ttl: Duration,
    ) -> Self {
        let mut merged: Vec<ShmClassBudget> = class_budgets
            .iter()
            .copied()
            .filter(|b| b.slot_size > 0 && b.weight > 0)
            .collect();
        merged.sort_unstable_by_key(|b| b.slot_size);
        merged.dedup_by(|a, b| {
            if a.slot_size == b.slot_size {
                a.weight = a.weight.saturating_add(b.weight);
                true
            } else {
                false
            }
        });

        if merged.is_empty() {
            merged.push(ShmClassBudget {
                slot_size: pool_bytes.max(1),
                weight: 1,
            });
        }

        let mut remaining = pool_bytes;
        let mut slots_per_class = vec![0usize; merged.len()];

        // Ensure every class gets at least one slot when pool capacity allows it.
        for (idx, budget) in merged.iter().enumerate() {
            if remaining >= budget.slot_size {
                slots_per_class[idx] = 1;
                remaining = remaining.saturating_sub(budget.slot_size);
            }
        }

        // Weighted round-robin slot assignment to distribute remaining bytes.
        let total_weight: i64 = merged.iter().map(|b| b.weight as i64).sum::<i64>().max(1);
        let mut credits = vec![0i64; merged.len()];
        while remaining > 0 {
            let mut best_idx: Option<usize> = None;
            let mut best_credit = i64::MIN;
            for (idx, budget) in merged.iter().enumerate() {
                if remaining < budget.slot_size {
                    continue;
                }
                credits[idx] += budget.weight as i64;
                if credits[idx] > best_credit {
                    best_credit = credits[idx];
                    best_idx = Some(idx);
                }
            }
            let Some(idx) = best_idx else {
                break;
            };
            slots_per_class[idx] = slots_per_class[idx].saturating_add(1);
            remaining = remaining.saturating_sub(merged[idx].slot_size);
            credits[idx] -= total_weight;
        }

        let mut classes = Vec::new();
        let mut cursor = base_offset;
        for (idx, budget) in merged.iter().copied().enumerate() {
            let num_slots = slots_per_class[idx];
            if num_slots == 0 {
                continue;
            }
            let bytes = num_slots.saturating_mul(budget.slot_size);
            let allocator =
                SimpleShmAllocator::new_with_ttl(cursor, budget.slot_size, num_slots, lease_ttl);
            classes.push(ShmAllocatorClass {
                name: format!("class{}", idx),
                base_offset: cursor,
                slot_size: budget.slot_size,
                num_slots,
                bytes,
                allocator,
            });

            cursor = cursor.saturating_add(bytes);
            remaining = remaining.saturating_sub(bytes);
        }

        if classes.is_empty() {
            let slot_size = pool_bytes.max(1);
            classes.push(ShmAllocatorClass {
                name: "class0".to_string(),
                base_offset,
                slot_size,
                num_slots: 1,
                bytes: slot_size,
                allocator: SimpleShmAllocator::new_with_ttl(base_offset, slot_size, 1, lease_ttl),
            });
        }

        Self { classes }
    }

    pub fn classes(&self) -> Vec<ShmAllocatorClassInfo> {
        self.classes
            .iter()
            .map(|c| ShmAllocatorClassInfo {
                name: c.name.clone(),
                base_offset: c.base_offset,
                slot_size: c.slot_size,
                num_slots: c.num_slots,
                bytes: c.bytes,
            })
            .collect()
    }
}

impl ShmPoolAllocator for TieredShmAllocator {
    fn try_allocate(&self, required_size: usize) -> Option<usize> {
        if required_size == 0 {
            return None;
        }

        let start_idx = self
            .classes
            .iter()
            .position(|class| required_size <= class.slot_size)?;

        for class in self.classes.iter().skip(start_idx) {
            if let Some(offset) = class.allocator.try_allocate(required_size) {
                return Some(offset);
            }
        }

        None
    }

    fn release(&self, offset: usize) -> bool {
        for class in &self.classes {
            let end = class.base_offset.saturating_add(class.bytes);
            if offset >= class.base_offset && offset < end {
                return class.allocator.release(offset);
            }
        }
        false
    }

    fn snapshot(&self) -> ShmAllocatorSnapshot {
        let mut in_use_slots = 0usize;
        let mut oldest_lease_ms = 0u64;
        for class in &self.classes {
            let snap = class.allocator.snapshot();
            in_use_slots = in_use_slots.saturating_add(snap.in_use_slots);
            oldest_lease_ms = oldest_lease_ms.max(snap.oldest_lease_ms);
        }
        ShmAllocatorSnapshot {
            in_use_slots,
            oldest_lease_ms,
        }
    }

    fn layout_summary(&self) -> String {
        self.classes
            .iter()
            .map(|class| {
                format!(
                    "{}(base={} slot_size={} slots={})",
                    class.name, class.base_offset, class.slot_size, class.num_slots
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn largest_slot_size(&self) -> usize {
        self.classes.iter().map(|c| c.slot_size).max().unwrap_or(0)
    }
}

#[cfg(test)]
#[path = "allocator_tests.rs"]
mod allocator_tests;
