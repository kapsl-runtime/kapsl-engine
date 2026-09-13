//! Opt-in, bounded request timing for manual diagnosis. No request data is recorded.
//!
//! Records stay in memory until explicit lifecycle teardown, so diagnostic log
//! I/O does not occur inside measured inference. This is not a qualification mode.

use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const LIMIT: usize = 8192;
const MAX_PHASES: usize = 16;

pub(crate) struct RequestProfile {
    component: &'static str,
    model_id: u32,
    replica_id: u32,
    origin: Instant,
    origin_unix_ns: u128,
    limit: usize,
    started: AtomicUsize,
    records: Mutex<Vec<Record>>,
}

#[derive(Serialize)]
struct Record {
    operation: &'static str,
    request_id: u64,
    sequence: usize,
    started_ns: u64,
    wall_ns: u64,
    phases: Vec<Phase>,
}

#[derive(Serialize)]
struct Phase {
    name: &'static str,
    wall_ns: u64,
}

pub(crate) struct RequestTiming<'a> {
    active: Option<ActiveTiming<'a>>,
}

struct ActiveTiming<'a> {
    profile: &'a RequestProfile,
    started: Instant,
    previous: Instant,
    record: Record,
}

fn ns(value: std::time::Duration) -> u64 {
    value.as_nanos().min(u128::from(u64::MAX)) as u64
}

impl RequestProfile {
    pub(crate) fn new(component: &'static str, model_id: u32, replica_id: u32) -> Self {
        Self::with_limit(
            component,
            model_id,
            replica_id,
            if std::env::var("KAPSL_REQUEST_PROFILING").as_deref() == Ok("1") {
                LIMIT
            } else {
                0
            },
        )
    }

    fn with_limit(component: &'static str, model_id: u32, replica_id: u32, limit: usize) -> Self {
        Self {
            component,
            model_id,
            replica_id,
            origin: Instant::now(),
            origin_unix_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            limit: limit.min(LIMIT),
            started: AtomicUsize::new(0),
            records: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn start(&self, operation: &'static str, request_id: u64) -> RequestTiming<'_> {
        if self.limit == 0 {
            return RequestTiming { active: None };
        }
        // Reserve before collecting phases: concurrent calls cannot exceed the
        // memory bound, including records whose spans have not finished yet.
        let sequence = self.started.fetch_add(1, Ordering::Relaxed);
        if sequence >= self.limit {
            return RequestTiming { active: None };
        }
        let started = Instant::now();
        RequestTiming {
            active: Some(ActiveTiming {
                profile: self,
                started,
                previous: started,
                record: Record {
                    operation,
                    request_id,
                    sequence,
                    started_ns: ns(started.duration_since(self.origin)),
                    wall_ns: 0,
                    phases: Vec::with_capacity(MAX_PHASES),
                },
            }),
        }
    }

    /// Call only after inference has drained. Keep one lifetime-wide sample
    /// budget across reloads; flushing cannot start unbounded new collection.
    pub(crate) fn flush(&self, mut emit: impl FnMut(&str)) {
        let records = std::mem::take(&mut *self.records.lock().unwrap_or_else(|p| p.into_inner()));
        for chunk in records.chunks(128) {
            let report = serde_json::json!({
                "schema_version": 1,
                "component": self.component,
                "process_id": std::process::id(),
                "model_id": self.model_id,
                "replica_id": self.replica_id,
                "origin_unix_ns": self.origin_unix_ns.to_string(),
                "sample_limit": self.limit,
                "started_samples": self.started.load(Ordering::Relaxed),
                "records": chunk,
            });
            emit(&format!("KAPSL_REQUEST_PROFILE {report}"));
        }
    }
}

impl RequestTiming<'_> {
    pub(crate) fn request_id(&mut self, id: u64) {
        if let Some(active) = &mut self.active {
            active.record.request_id = id;
        }
    }

    pub(crate) fn mark(&mut self, name: &'static str) {
        if let Some(active) = &mut self.active {
            let now = Instant::now();
            if active.record.phases.len() < MAX_PHASES {
                active.record.phases.push(Phase {
                    name,
                    wall_ns: ns(now.duration_since(active.previous)),
                });
            }
            active.previous = now;
        }
    }
}

impl Drop for RequestTiming<'_> {
    fn drop(&mut self) {
        self.mark("return_cleanup");
        if let Some(mut active) = self.active.take() {
            active.record.wall_ns = ns(active.started.elapsed());
            active
                .profile
                .records
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(active.record);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_profiles_collect_nothing_and_reservations_bound_concurrent_spans() {
        let disabled = RequestProfile::with_limit("test", 1, 2, 0);
        drop(disabled.start("infer", 9));
        disabled.flush(|_| panic!("disabled profiling emitted output"));
        let profile = RequestProfile::with_limit("test", 1, 2, 2);
        let mut first = profile.start("infer", 0);
        let second = profile.start("infer", 10);
        drop(profile.start("infer", 11));
        first.request_id(9);
        first.mark("execute");
        drop(second);
        drop(first);
        let mut reports = Vec::new();
        profile.flush(|line| reports.push(line.to_owned()));
        let report: serde_json::Value =
            serde_json::from_str(reports[0].strip_prefix("KAPSL_REQUEST_PROFILE ").unwrap())
                .unwrap();
        assert_eq!(report["model_id"], 1);
        assert_eq!(report["replica_id"], 2);
        let records = report["records"].as_array().unwrap();
        assert_eq!(records.len(), 2);
        assert!(records.iter().any(|record| record["request_id"] == 9));
        drop(profile.start("infer", 12));
        profile.flush(|_| panic!("flush reset the lifetime collection limit"));
    }
}
