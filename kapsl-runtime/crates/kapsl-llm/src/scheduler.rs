use crate::block_manager::BlockManager;
use crate::sequence::{FinishReason, Sequence, SequenceGroup, SequenceGroupOutput, SequenceStatus};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct SchedulerConfig {
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub max_paddings: usize,
}

pub struct SchedulerOutputs {
    pub scheduled_seq_groups: Vec<Arc<Mutex<SequenceGroup>>>,
}

pub struct LLMScheduler {
    config: SchedulerConfig,
    waiting_queue: VecDeque<Arc<Mutex<SequenceGroup>>>,
    running_queue: VecDeque<Arc<Mutex<SequenceGroup>>>,
    swapped_queue: VecDeque<Arc<Mutex<SequenceGroup>>>,
    block_manager: BlockManager,
}

impl LLMScheduler {
    pub fn new(config: SchedulerConfig, block_manager: BlockManager) -> Self {
        Self {
            config,
            waiting_queue: VecDeque::new(),
            running_queue: VecDeque::new(),
            swapped_queue: VecDeque::new(),
            block_manager,
        }
    }

    pub fn add_sequence_group(&mut self, seq_group: SequenceGroup) {
        self.waiting_queue
            .push_back(Arc::new(Mutex::new(seq_group)));
    }

    pub fn active_sequence_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        for group_arc in self
            .waiting_queue
            .iter()
            .chain(self.running_queue.iter())
            .chain(self.swapped_queue.iter())
        {
            if let Ok(group) = group_arc.lock() {
                ids.extend(group.sequences.keys().copied());
            }
        }
        ids
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        // Calculate current running tokens (decode): 1 token per running sequence.
        let mut num_batched_tokens = 0usize;

        for group in &self.running_queue {
            let group = group.lock().unwrap();
            num_batched_tokens += group.cached_running_count();
        }

        let block_size = self.block_manager.block_size();

        // 1. Swap in swapped-out groups (best-effort FIFO).
        while let Some(group_arc) = self.swapped_queue.front().cloned() {
            let cancelled = {
                let group = group_arc.lock().unwrap();
                group
                    .cancellation
                    .as_ref()
                    .is_some_and(|t| t.is_cancelled())
            };
            if cancelled {
                Self::cancel_group(&group_arc);
                let _ = self.swapped_queue.pop_front();
                continue;
            }

            let seqs: Vec<(u64, Arc<Mutex<Sequence>>)> = {
                let group = group_arc.lock().unwrap();
                if group.is_finished() {
                    // Nothing to do; drop it.
                    drop(group);
                    let _ = self.swapped_queue.pop_front();
                    continue;
                }

                group
                    .sequences
                    .iter()
                    .map(|(seq_id, seq_arc)| (*seq_id, seq_arc.clone()))
                    .collect()
            };

            let mut requirements: Vec<(u64, Arc<Mutex<Sequence>>, usize)> =
                Vec::with_capacity(seqs.len());
            let mut total_additional_blocks = 0usize;

            for (seq_id, seq_arc) in &seqs {
                let (is_finished, seq_len) = {
                    let seq = seq_arc.lock().unwrap();
                    (seq.is_finished(), seq.get_len())
                };
                if is_finished {
                    continue;
                }

                let required_blocks = seq_len.div_ceil(block_size);
                let existing_blocks = self
                    .block_manager
                    .get_block_table(*seq_id)
                    .map(|table| table.len())
                    .unwrap_or(0);
                let additional_blocks = required_blocks.saturating_sub(existing_blocks);
                total_additional_blocks += additional_blocks;
                requirements.push((*seq_id, seq_arc.clone(), additional_blocks));
            }

            let group_decode_tokens = requirements.len();
            if group_decode_tokens == 0 {
                let _ = self.swapped_queue.pop_front();
                continue;
            }

            if num_batched_tokens + group_decode_tokens > self.config.max_num_batched_tokens {
                break;
            }

            if !self.block_manager.can_allocate(total_additional_blocks) {
                break;
            }

            let mut status_updates: Vec<(SequenceStatus, SequenceStatus)> =
                Vec::with_capacity(requirements.len());
            for (seq_id, seq_arc, additional_blocks) in &requirements {
                for _ in 0..*additional_blocks {
                    let _ = self.block_manager.allocate(*seq_id);
                }

                let (old_status, new_status) = {
                    let mut seq = seq_arc.lock().unwrap();
                    let old_status = seq.status;
                    if !seq.is_finished() && seq.status != SequenceStatus::Running {
                        seq.status = SequenceStatus::Running;
                    }
                    (old_status, seq.status)
                };
                if old_status != new_status {
                    status_updates.push((old_status, new_status));
                }
            }
            if !status_updates.is_empty() {
                let mut group = group_arc.lock().unwrap();
                for (old_status, new_status) in status_updates {
                    group.update_seq_status(old_status, new_status);
                }
            }

            let group_arc = self.swapped_queue.pop_front().unwrap();
            self.running_queue.push_back(group_arc);
            num_batched_tokens += group_decode_tokens;
        }

        // 2. Schedule waiting groups (Prefill).
        // Try to add new groups from waiting queue
        while let Some(group_arc) = self.waiting_queue.front().cloned() {
            let cancelled = {
                let group = group_arc.lock().unwrap();
                group
                    .cancellation
                    .as_ref()
                    .is_some_and(|t| t.is_cancelled())
            };
            if cancelled {
                Self::cancel_group(&group_arc);
                let _ = self.waiting_queue.pop_front();
                continue;
            }

            let seqs: Vec<(u64, Arc<Mutex<Sequence>>)> = {
                let group = group_arc.lock().unwrap();
                if group.is_finished() {
                    // Unexpected in the waiting queue, but safe to drop.
                    drop(group);
                    let _ = self.waiting_queue.pop_front();
                    continue;
                }

                group
                    .sequences
                    .iter()
                    .map(|(seq_id, seq_arc)| (*seq_id, seq_arc.clone()))
                    .collect()
            };

            let mut num_pending_tokens = 0usize;
            let mut requirements: Vec<(u64, Arc<Mutex<Sequence>>, usize)> =
                Vec::with_capacity(seqs.len());
            let mut total_additional_blocks = 0usize;

            for (seq_id, seq_arc) in &seqs {
                let (is_finished, seq_len, kv_cached_len) = {
                    let seq = seq_arc.lock().unwrap();
                    (seq.is_finished(), seq.get_len(), seq.kv_cached_len)
                };
                if is_finished {
                    continue;
                }

                num_pending_tokens += seq_len.saturating_sub(kv_cached_len);

                let required_blocks = seq_len.div_ceil(block_size);
                let existing_blocks = self
                    .block_manager
                    .get_block_table(*seq_id)
                    .map(|table| table.len())
                    .unwrap_or(0);
                let additional_blocks = required_blocks.saturating_sub(existing_blocks);
                total_additional_blocks += additional_blocks;
                requirements.push((*seq_id, seq_arc.clone(), additional_blocks));
            }

            if num_pending_tokens == 0 {
                // Nothing to prefill; move to running to let decode proceed.
                let group_arc = self.waiting_queue.pop_front().unwrap();
                self.running_queue.push_back(group_arc);
                continue;
            }

            if num_batched_tokens + num_pending_tokens > self.config.max_num_batched_tokens {
                break; // Token limit reached
            }

            if !self.block_manager.can_allocate(total_additional_blocks) {
                break; // Not enough memory
            }

            let mut status_updates: Vec<(SequenceStatus, SequenceStatus)> =
                Vec::with_capacity(requirements.len());
            for (seq_id, seq_arc, additional_blocks) in &requirements {
                for _ in 0..*additional_blocks {
                    let _ = self.block_manager.allocate(*seq_id);
                }

                let (old_status, new_status) = {
                    let mut seq = seq_arc.lock().unwrap();
                    let old_status = seq.status;
                    if !seq.is_finished() && seq.status != SequenceStatus::Running {
                        seq.status = SequenceStatus::Running;
                    }
                    (old_status, seq.status)
                };
                if old_status != new_status {
                    status_updates.push((old_status, new_status));
                }
            }
            if !status_updates.is_empty() {
                let mut group = group_arc.lock().unwrap();
                for (old_status, new_status) in status_updates {
                    group.update_seq_status(old_status, new_status);
                }
            }

            let group_arc = self.waiting_queue.pop_front().unwrap();
            self.running_queue.push_back(group_arc);
            num_batched_tokens += num_pending_tokens;
        }

        let scheduled_seq_groups = self.running_queue.iter().cloned().collect();

        SchedulerOutputs {
            scheduled_seq_groups,
        }
    }

    fn cancel_group(group_arc: &Arc<Mutex<SequenceGroup>>) {
        let (request_id, response_tx, seqs, already_finished) = {
            let group = group_arc.lock().unwrap();
            (
                group.request_id.clone(),
                group.response_tx.clone(),
                group.sequences.values().cloned().collect::<Vec<_>>(),
                group.is_finished(),
            )
        };
        if already_finished {
            return;
        }

        let mut status_updates: Vec<(SequenceStatus, SequenceStatus)> = Vec::new();
        for seq_arc in &seqs {
            let (old_status, new_status) = {
                let mut seq = seq_arc.lock().unwrap();
                let old_status = seq.status;
                if !seq.is_finished() {
                    seq.status = SequenceStatus::Finished(FinishReason::Cancelled);
                }
                (old_status, seq.status)
            };
            if old_status != new_status {
                status_updates.push((old_status, new_status));
            }
        }

        if !status_updates.is_empty() {
            let mut group = group_arc.lock().unwrap();
            for (old_status, new_status) in status_updates {
                group.update_seq_status(old_status, new_status);
            }
        }

        let _ = response_tx.try_send(SequenceGroupOutput {
            request_id,
            text: String::new(),
            finish_reason: Some(FinishReason::Cancelled),
        });
    }

    pub fn free_finished_sequences(&mut self) -> Vec<u64> {
        let mut finished_ids = Vec::new();
        self.running_queue.retain(|group_arc| {
            let group = group_arc.lock().unwrap();
            if group.is_finished() {
                if group.session_id.is_none() {
                    // Free blocks for non-session sequences.
                    for seq_id in group.sequences.keys() {
                        self.block_manager.free(*seq_id);
                        finished_ids.push(*seq_id);
                    }
                }
                false
            } else {
                true
            }
        });
        finished_ids
    }
}

#[path = "scheduler_tests.rs"]
mod scheduler_tests;
