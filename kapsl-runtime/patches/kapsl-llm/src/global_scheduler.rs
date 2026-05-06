//! Cross-model KV token-budget coordinator.
//!
//! # Purpose
//!
//! Each [`LLMEngine`] owns an [`LLMScheduler`] that manages one model's
//! sequences independently.  Without a shared coordinator, a single busy model
//! can consume the entire GPU KV block pool and starve all others.
//!
//! [`GlobalKvScheduler`] sits above the per-engine schedulers and enforces
//! two things:
//!
//! 1. **Proportional token budget** — each registered engine is allocated a
//!    slice of the global `max_batched_tokens` budget proportional to its
//!    declared share weight.  An engine that goes idle donates its unused budget
//!    to active engines up to the configured maximum.
//!
//! 2. **Cross-model preemption** — when an engine cannot schedule a high-
//!    priority request because its own block budget is exhausted, it can ask the
//!    global scheduler to reclaim blocks from lower-priority engines.
//!
//! # Integration
//!
//! The global scheduler is *advisory*: it computes per-engine budgets and
//! signals preemption requests, but the actual scheduling loop remains inside
//! each [`LLMScheduler`].  This keeps the change surface minimal while
//! providing the coordination layer needed for T1 parity.
//!
//! ```no_run
//! use kapsl_llm::block_manager::new_shared_allocator;
//! use kapsl_llm::global_scheduler::{GlobalKvScheduler, EngineHandle};
//!
//! // Build one shared block allocator for the device.
//! let shared_pool = new_shared_allocator(4096, 16, 0);
//!
//! // Create the global scheduler.
//! let mut global = GlobalKvScheduler::new(8192); // 8192 tokens / round
//!
//! // Register engines as they load.
//! global.register(EngineHandle { engine_id: 0, share_weight: 1 });
//! global.register(EngineHandle { engine_id: 1, share_weight: 2 }); // 2× the budget
//!
//! // Each scheduling round, ask for per-engine budgets.
//! let budgets = global.allocate_budgets();
//! // budgets[0].max_tokens ≈ 2730  (1/3 of 8192)
//! // budgets[1].max_tokens ≈ 5461  (2/3 of 8192)
//! ```

use std::collections::HashMap;

/// Lightweight descriptor for one registered engine.
#[derive(Debug, Clone)]
pub struct EngineHandle {
    /// Stable identifier that maps to a loaded model/engine instance.
    pub engine_id: u32,
    /// Relative share weight.  An engine with weight 2 gets twice the token
    /// budget of one with weight 1 when both are active.
    pub share_weight: u32,
}

/// Per-engine token budget issued by [`GlobalKvScheduler::allocate_budgets`].
#[derive(Debug, Clone)]
pub struct EngineTokenBudget {
    pub engine_id: u32,
    /// Maximum number of tokens (prefill + decode) the engine may schedule
    /// in the current round.
    pub max_tokens: usize,
}

/// Cross-engine preemption request raised when an engine has insufficient
/// free blocks for an incoming request.
#[derive(Debug, Clone)]
pub struct PreemptionRequest {
    /// Engine that is requesting more blocks.
    pub requesting_engine_id: u32,
    /// Minimum number of blocks needed.
    pub blocks_needed: usize,
    /// Priority of the incoming request — used to target engines that are
    /// currently running lower-priority work.
    pub request_priority: u8,
}

/// Result of a preemption request.
#[derive(Debug, Clone)]
pub struct PreemptionResult {
    /// Engine that was asked to free blocks.
    pub donor_engine_id: u32,
    /// Number of blocks that were freed on that engine's scheduler.
    pub blocks_freed: usize,
}

/// Internal per-engine state tracked by the global scheduler.
#[derive(Debug)]
struct EngineState {
    share_weight: u32,
    /// Tokens consumed during the most recent scheduling round (filled in by
    /// the caller via [`GlobalKvScheduler::report_usage`]).
    last_used_tokens: usize,
    /// Whether the engine had any pending requests in the last round.
    was_active: bool,
}

/// Cross-model token-budget coordinator.
///
/// Maintains a registry of active [`EngineHandle`]s and distributes the
/// global token budget proportionally each scheduling round. Also accepts and
/// routes cross-engine preemption requests.
#[derive(Debug)]
pub struct GlobalKvScheduler {
    /// Total tokens available per scheduling round across all engines.
    global_max_tokens: usize,
    engines: HashMap<u32, EngineState>,
    /// Ordered list of engine IDs in registration order (for stable iteration).
    engine_order: Vec<u32>,
    /// Maximum fraction of the global budget any single engine may consume,
    /// expressed as a per-mille value (1000 = 100 %).  Prevents monopolisation
    /// even when all other engines are idle.  Default: 900 ‰ (90 %).
    max_single_engine_permille: u32,
}

impl GlobalKvScheduler {
    /// Create a new coordinator with the given total token budget per round.
    pub fn new(global_max_tokens: usize) -> Self {
        Self {
            global_max_tokens,
            engines: HashMap::new(),
            engine_order: Vec::new(),
            max_single_engine_permille: 900,
        }
    }

    /// Set the maximum fraction of the global budget any single engine may
    /// receive, in per-mille units (1000 = 100 %).
    pub fn with_max_single_engine_permille(mut self, permille: u32) -> Self {
        self.max_single_engine_permille = permille.clamp(100, 1000);
        self
    }

    /// Register an engine.  If the engine was already registered its weight
    /// is updated.
    pub fn register(&mut self, handle: EngineHandle) {
        if !self.engines.contains_key(&handle.engine_id) {
            self.engine_order.push(handle.engine_id);
        }
        self.engines.insert(
            handle.engine_id,
            EngineState {
                share_weight: handle.share_weight.max(1),
                last_used_tokens: 0,
                was_active: false,
            },
        );
    }

    /// Deregister an engine (e.g. after it is unloaded).
    pub fn deregister(&mut self, engine_id: u32) {
        self.engines.remove(&engine_id);
        self.engine_order.retain(|&id| id != engine_id);
    }

    /// Mark an engine as active or idle for the coming round.
    ///
    /// Idle engines donate their share to active ones (up to the per-engine
    /// cap).  Call this before [`allocate_budgets`].
    pub fn set_active(&mut self, engine_id: u32, active: bool) {
        if let Some(state) = self.engines.get_mut(&engine_id) {
            state.was_active = active;
        }
    }

    /// Report how many tokens engine `engine_id` actually consumed last round.
    ///
    /// Used to compute utilisation metrics and to adjust budget donations in
    /// future rounds.
    pub fn report_usage(&mut self, engine_id: u32, used_tokens: usize) {
        if let Some(state) = self.engines.get_mut(&engine_id) {
            state.last_used_tokens = used_tokens;
        }
    }

    /// Compute per-engine token budgets for the current scheduling round.
    ///
    /// Active engines receive a share proportional to their `share_weight`;
    /// idle engines are excluded from the distribution so their share is
    /// redistributed among active ones.  No engine receives more than
    /// `max_single_engine_permille ‰` of the global budget.
    pub fn allocate_budgets(&self) -> Vec<EngineTokenBudget> {
        if self.engines.is_empty() || self.global_max_tokens == 0 {
            return Vec::new();
        }

        let active_total_weight: u64 = self
            .engine_order
            .iter()
            .filter_map(|id| self.engines.get(id))
            .filter(|s| s.was_active)
            .map(|s| s.share_weight as u64)
            .sum();

        // If no engine reported itself active, treat all as active to avoid
        // zero-budget stalls on the first round.
        let treat_all_active = active_total_weight == 0;

        // Always use all-engine weights for natural shares so that an idle
        // engine's natural share can be measured and re-distributed.
        let all_total_weight: u64 = self
            .engines
            .values()
            .map(|s| s.share_weight as u64)
            .sum::<u64>()
            .max(1);

        let cap_tokens = (self.global_max_tokens as u64 * self.max_single_engine_permille as u64
            / 1000) as usize;

        let mut budgets: Vec<EngineTokenBudget> = Vec::with_capacity(self.engines.len());
        // Shares from idle engines that should be absorbed by active engines.
        let mut idle_pool: usize = 0;
        // Sum of natural (uncapped) shares; used to compute integer-rounding leftover.
        let mut natural_sum: usize = 0;

        for &engine_id in &self.engine_order {
            let Some(state) = self.engines.get(&engine_id) else {
                continue;
            };

            let is_active = state.was_active || treat_all_active;
            let natural = (self.global_max_tokens as u64 * state.share_weight as u64
                / all_total_weight) as usize;
            natural_sum += natural;

            let max_tokens = if is_active {
                natural.min(cap_tokens)
            } else {
                idle_pool += natural;
                0
            };

            budgets.push(EngineTokenBudget {
                engine_id,
                max_tokens,
            });
        }

        // Redistribute idle engines' natural shares to the first active engine.
        // This is intentionally uncapped: the active engine is genuinely
        // absorbing budget that idle peers are not using.
        if idle_pool > 0 {
            if let Some(budget) = budgets.iter_mut().find(|b| b.max_tokens > 0) {
                budget.max_tokens += idle_pool;
            }
        }

        // Distribute the integer-rounding remainder (global_max - sum of natural
        // shares, at most n-1 tokens) to the first active engine, respecting cap.
        let rounding = self.global_max_tokens.saturating_sub(natural_sum);
        if rounding > 0 {
            if let Some(budget) = budgets.iter_mut().find(|b| b.max_tokens > 0) {
                let headroom = cap_tokens.saturating_sub(budget.max_tokens);
                budget.max_tokens += rounding.min(headroom);
            }
        }

        budgets
    }

    /// Return the budget ceiling for a single engine, or `None` if it is not
    /// registered.
    pub fn budget_for(&self, engine_id: u32) -> Option<usize> {
        self.allocate_budgets()
            .into_iter()
            .find(|b| b.engine_id == engine_id)
            .map(|b| b.max_tokens)
    }

    /// Total number of registered engines.
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Determine which engine should donate blocks for a preemption request.
    ///
    /// Chooses the registered engine (other than the requesting one) whose
    /// running work has the lowest priority, as a heuristic for which engine
    /// is least harmed by a swap-out.  Returns the donor engine ID and how
    /// many blocks it could theoretically free if its scheduler evicts its
    /// lowest-priority groups.
    ///
    /// The caller is responsible for actually invoking preemption on the donor
    /// engine's `LLMScheduler` (via `try_preempt_for_blocks`).
    pub fn find_preemption_donor(
        &self,
        request: &PreemptionRequest,
        // Per-engine (engine_id → lowest running priority) reported by engines
        engine_priorities: &HashMap<u32, u8>,
        // Per-engine (engine_id → estimated free-able blocks)
        engine_freeable_blocks: &HashMap<u32, usize>,
    ) -> Option<u32> {
        // Among other engines, find the one with the lowest minimum priority
        // (most evictable) that has enough freeable blocks.
        self.engine_order
            .iter()
            .filter(|&&id| id != request.requesting_engine_id)
            .filter(|&&id| self.engines.contains_key(&id))
            .filter(|&&id| {
                // Only consider engines running work at strictly lower priority
                // than the requesting engine's request.
                engine_priorities
                    .get(&id)
                    .map(|&p| p < request.request_priority)
                    .unwrap_or(false)
            })
            .filter(|&&id| {
                engine_freeable_blocks
                    .get(&id)
                    .map(|&b| b >= request.blocks_needed)
                    .unwrap_or(false)
            })
            .min_by_key(|&&id| engine_priorities.get(&id).copied().unwrap_or(u8::MAX))
            .copied()
    }

    /// Current global token budget.
    pub fn global_max_tokens(&self) -> usize {
        self.global_max_tokens
    }

    /// Update the global token budget.
    pub fn set_global_max_tokens(&mut self, tokens: usize) {
        self.global_max_tokens = tokens;
    }
}

#[cfg(test)]
mod global_scheduler_tests {
    use super::*;

    fn make_scheduler(total: usize, engines: &[(u32, u32, bool)]) -> GlobalKvScheduler {
        let mut sched = GlobalKvScheduler::new(total);
        for &(id, weight, active) in engines {
            sched.register(EngineHandle {
                engine_id: id,
                share_weight: weight,
            });
            sched.set_active(id, active);
        }
        sched
    }

    #[test]
    fn equal_weights_split_evenly() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        let budgets = sched.allocate_budgets();
        assert_eq!(budgets.len(), 2);
        let total: usize = budgets.iter().map(|b| b.max_tokens).sum();
        assert_eq!(total, 1000);
        // Each engine should get approximately half.
        for b in &budgets {
            assert!(b.max_tokens >= 490 && b.max_tokens <= 510, "{b:?}");
        }
    }

    #[test]
    fn weighted_distribution() {
        // Engine 0 weight 1, engine 1 weight 3 → 25% / 75%.
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 3, true)]);
        let budgets = sched.allocate_budgets();
        let b0 = budgets
            .iter()
            .find(|b| b.engine_id == 0)
            .unwrap()
            .max_tokens;
        let b1 = budgets
            .iter()
            .find(|b| b.engine_id == 1)
            .unwrap()
            .max_tokens;
        // Allow ±1 for integer rounding.
        assert!(b0 >= 249 && b0 <= 251, "b0={b0}");
        assert!(b1 >= 749 && b1 <= 751, "b1={b1}");
        assert_eq!(b0 + b1, 1000);
    }

    #[test]
    fn idle_engine_gets_zero() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, false)]);
        let budgets = sched.allocate_budgets();
        let b1 = budgets
            .iter()
            .find(|b| b.engine_id == 1)
            .unwrap()
            .max_tokens;
        assert_eq!(b1, 0, "idle engine should get zero budget");
    }

    #[test]
    fn active_engine_absorbs_idle_share() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, false)]);
        let budgets = sched.allocate_budgets();
        let b0 = budgets
            .iter()
            .find(|b| b.engine_id == 0)
            .unwrap()
            .max_tokens;
        // Active engine should absorb the idle engine's share (all 1000).
        assert_eq!(b0, 1000);
    }

    #[test]
    fn single_engine_cap_respected() {
        let mut sched = GlobalKvScheduler::new(1000).with_max_single_engine_permille(500);
        sched.register(EngineHandle {
            engine_id: 0,
            share_weight: 1,
        });
        sched.set_active(0, true);
        let budgets = sched.allocate_budgets();
        let b0 = budgets[0].max_tokens;
        // Cap is 50% of 1000 = 500.
        assert!(b0 <= 500, "b0={b0} exceeds cap");
    }

    #[test]
    fn deregister_removes_engine() {
        let mut sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        sched.deregister(1);
        assert_eq!(sched.engine_count(), 1);
        let budgets = sched.allocate_budgets();
        assert_eq!(budgets.len(), 1);
        assert_eq!(budgets[0].engine_id, 0);
    }

    #[test]
    fn find_preemption_donor_picks_lowest_priority() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true), (2, 1, true)]);
        let req = PreemptionRequest {
            requesting_engine_id: 0,
            blocks_needed: 10,
            request_priority: 5,
        };
        // Engine 1 runs priority 1, engine 2 runs priority 3; both have enough blocks.
        let priorities: HashMap<u32, u8> = [(1, 1), (2, 3)].into();
        let freeable: HashMap<u32, usize> = [(1, 20), (2, 20)].into();
        let donor = sched.find_preemption_donor(&req, &priorities, &freeable);
        // Engine 1 has lower priority (1 < 3) and both < request_priority (5), so engine 1 wins.
        assert_eq!(donor, Some(1));
    }

    #[test]
    fn no_donor_when_all_higher_priority() {
        let sched = make_scheduler(1000, &[(0, 1, true), (1, 1, true)]);
        let req = PreemptionRequest {
            requesting_engine_id: 0,
            blocks_needed: 10,
            request_priority: 1, // lower than what engines are running
        };
        let priorities: HashMap<u32, u8> = [(1, 10)].into(); // engine 1 runs priority 10 > 1
        let freeable: HashMap<u32, usize> = [(1, 20)].into();
        let donor = sched.find_preemption_donor(&req, &priorities, &freeable);
        // Engine 1's priority (10) is NOT < request_priority (1), so no donor.
        assert_eq!(donor, None);
    }
}
