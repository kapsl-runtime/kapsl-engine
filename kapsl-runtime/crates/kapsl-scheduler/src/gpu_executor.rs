use crate::request::Request;
use kapsl_engine_api::{EngineError, EngineHandle, InferenceRequest};
use log::info;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::time::timeout;

struct InFlightGuard {
    counter: Arc<AtomicUsize>,
    n: usize,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.n, Ordering::Relaxed);
    }
}

struct WorkQueueInner {
    queue: Mutex<VecDeque<Request>>,
    capacity: usize,
    queue_len: AtomicUsize,
    not_empty: Notify,
    not_full: Notify,
}

#[derive(Clone)]
pub(crate) struct WorkQueue {
    inner: Arc<WorkQueueInner>,
}

impl WorkQueue {
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Arc::new(WorkQueueInner {
                queue: Mutex::new(VecDeque::with_capacity(capacity)),
                capacity,
                queue_len: AtomicUsize::new(0),
                not_empty: Notify::new(),
                not_full: Notify::new(),
            }),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.queue_len.load(Ordering::Relaxed)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub(crate) fn try_push_drop_newest(&self, request: Request) -> Result<(), Request> {
        let mut queue = self.inner.queue.lock().unwrap();
        if queue.len() >= self.inner.capacity {
            return Err(request);
        }
        queue.push_back(request);
        self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
        drop(queue);
        self.inner.not_empty.notify_one();
        Ok(())
    }

    pub(crate) fn push_drop_oldest(&self, request: Request) -> Option<Request> {
        let mut queue = self.inner.queue.lock().unwrap();
        let is_full = queue.len() >= self.inner.capacity;
        let dropped = if is_full { queue.pop_front() } else { None };
        queue.push_back(request);
        if !is_full {
            self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
        }
        drop(queue);
        self.inner.not_empty.notify_one();
        dropped
    }

    pub(crate) async fn push_block(&self, request: Request) {
        let mut pending = Some(request);
        loop {
            let queued = {
                let mut queue = self.inner.queue.lock().unwrap();
                if queue.len() < self.inner.capacity {
                    queue.push_back(pending.take().expect("pending request must exist"));
                    self.inner.queue_len.fetch_add(1, Ordering::Relaxed);
                    true
                } else {
                    false
                }
            };

            if queued {
                self.inner.not_empty.notify_one();
                return;
            }

            self.inner.not_full.notified().await;
        }
    }

    pub(crate) fn pop_nowait(&self) -> Option<Request> {
        let mut queue = self.inner.queue.lock().unwrap();
        let popped = queue.pop_front();
        if popped.is_some() {
            self.inner.queue_len.fetch_sub(1, Ordering::Relaxed);
        }
        drop(queue);
        if popped.is_some() {
            self.inner.not_full.notify_one();
        }
        popped
    }

    pub(crate) async fn pop_timeout(&self, timeout_duration: Duration) -> Option<Request> {
        let deadline = Instant::now() + timeout_duration;
        loop {
            if let Some(request) = self.pop_nowait() {
                return Some(request);
            }

            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let remaining = deadline.saturating_duration_since(now);

            if timeout(remaining, self.inner.not_empty.notified())
                .await
                .is_err()
            {
                return None;
            }
        }
    }

    pub(crate) async fn wait_for_item(&self) {
        loop {
            if !self.inner.queue.lock().unwrap().is_empty() {
                return;
            }
            self.inner.not_empty.notified().await;
        }
    }
}

/// GPU Executor that processes requests from priority queues
pub struct GpuExecutor {
    high_priority_queue: WorkQueue,
    low_priority_queue: WorkQueue,
    engine: EngineHandle,
    max_micro_batch: usize,
    queue_delay: Duration,
    in_flight: Arc<AtomicUsize>,
}

impl GpuExecutor {
    pub(crate) fn new(
        high_priority_queue: WorkQueue,
        low_priority_queue: WorkQueue,
        engine: EngineHandle,
        max_micro_batch: usize,
        queue_delay_ms: u64,
        in_flight: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            high_priority_queue,
            low_priority_queue,
            engine,
            max_micro_batch: max_micro_batch.max(1),
            queue_delay: Duration::from_millis(queue_delay_ms),
            in_flight,
        }
    }

    fn dispatch_single(engine: EngineHandle, req: Request, in_flight: Arc<AtomicUsize>) {
        if req
            .input
            .cancellation
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            let _ = req
                .response_tx
                .send(Err(EngineError::cancelled("Request cancelled")));
            return;
        }

        in_flight.fetch_add(1, Ordering::Relaxed);
        tokio::task::spawn_blocking(move || {
            let _guard = InFlightGuard {
                counter: in_flight,
                n: 1,
            };
            let result = engine.infer(&req.input);
            let _ = req.response_tx.send(result);
        });
    }

    fn dispatch_batch(engine: EngineHandle, requests: Vec<Request>, in_flight: Arc<AtomicUsize>) {
        if requests.is_empty() {
            return;
        }

        let mut active = Vec::with_capacity(requests.len());
        for request in requests {
            if request
                .input
                .cancellation
                .as_ref()
                .is_some_and(|token| token.is_cancelled())
            {
                let _ = request
                    .response_tx
                    .send(Err(EngineError::cancelled("Request cancelled")));
                continue;
            }
            active.push(request);
        }

        if active.is_empty() {
            return;
        }

        let n = active.len();
        in_flight.fetch_add(n, Ordering::Relaxed);
        tokio::task::spawn_blocking(move || {
            let _guard = InFlightGuard {
                counter: in_flight,
                n,
            };
            if active.len() == 1 {
                let req = active
                    .into_iter()
                    .next()
                    .expect("single request batch must have one item");
                let result = engine.infer(&req.input);
                let _ = req.response_tx.send(result);
                return;
            }

            let infer_requests: Vec<InferenceRequest> =
                active.iter().map(|request| request.input.clone()).collect();

            match engine.infer_batch(&infer_requests) {
                Ok(outputs) => {
                    if outputs.len() != active.len() {
                        let reason = format!(
                            "Batched inference result length mismatch: expected {}, got {}",
                            active.len(),
                            outputs.len()
                        );
                        for request in active {
                            let _ = request.response_tx.send(Err(EngineError::InferenceError {
                                reason: reason.clone(),
                                source: None,
                            }));
                        }
                        return;
                    }

                    for (request, output) in active.into_iter().zip(outputs.into_iter()) {
                        let _ = request.response_tx.send(Ok(output));
                    }
                }
                Err(error) => {
                    let reason = format!("Batched inference failed: {}", error);
                    for request in active {
                        let _ = request.response_tx.send(Err(EngineError::InferenceError {
                            reason: reason.clone(),
                            source: None,
                        }));
                    }
                }
            }
        });
    }

    pub async fn run(self) {
        info!("GPU Executor started");
        loop {
            if let Some(req) = self.high_priority_queue.pop_nowait() {
                let engine = self.engine.clone();
                Self::dispatch_single(engine, req, self.in_flight.clone());
                continue;
            }

            if let Some(req) = self.low_priority_queue.pop_nowait() {
                let engine = self.engine.clone();
                if self.max_micro_batch <= 1 || self.queue_delay.is_zero() {
                    Self::dispatch_single(engine, req, self.in_flight.clone());
                    continue;
                }

                let mut batch = Vec::with_capacity(self.max_micro_batch);
                batch.push(req);
                let deadline = Instant::now() + self.queue_delay;

                while batch.len() < self.max_micro_batch {
                    if self.high_priority_queue.len() > 0 {
                        break;
                    }

                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }

                    let remaining = deadline.saturating_duration_since(now);
                    match self.low_priority_queue.pop_timeout(remaining).await {
                        Some(next_req) => batch.push(next_req),
                        None => break,
                    }
                }

                Self::dispatch_batch(engine, batch, self.in_flight.clone());
                continue;
            }

            tokio::select! {
                _ = self.high_priority_queue.wait_for_item() => {}
                _ = self.low_priority_queue.wait_for_item() => {}
            }
        }
    }
}
