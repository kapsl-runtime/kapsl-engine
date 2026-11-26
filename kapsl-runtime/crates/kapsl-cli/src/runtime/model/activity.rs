//! Per-load inference activity, read by the opt-in idle unloader.

use super::*;

/// High bit of [`ModelActivity::state`]: the idle unloader has claimed this
/// load, so no new request may start on it.
const IDLE_CLAIMED: usize = 1 << (usize::BITS - 1);

/// Inference activity for one installed load of a model.
///
/// A fresh value is created every time a model is installed, so a request
/// guard left over from an earlier load can never keep a later load busy.
/// Idle time starts at installation.
pub(crate) struct ModelActivity {
    origin: Instant,
    /// Number of requests in flight, plus [`IDLE_CLAIMED`] while the idle
    /// unloader holds this load. Keeping both in one word makes "no request
    /// is in flight" and "no new request may start" a single atomic step.
    state: AtomicUsize,
    /// Milliseconds after `origin` at which a request last started or ended.
    last_active_ms: AtomicU64,
}

impl ModelActivity {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            origin: Instant::now(),
            state: AtomicUsize::new(0),
            last_active_ms: AtomicU64::new(0),
        })
    }

    /// Count one request as in flight until the returned guard is dropped.
    /// Returns `None` once the idle unloader has claimed this load.
    pub(crate) fn try_begin_request(self: &Arc<Self>) -> Option<ModelActivityGuard> {
        self.state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & IDLE_CLAIMED == 0).then_some(state + 1)
            })
            .ok()?;
        self.touch(Instant::now());
        Some(ModelActivityGuard {
            activity: Arc::clone(self),
        })
    }

    /// Requests currently in flight on this load.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.state.load(Ordering::Acquire) & !IDLE_CLAIMED
    }

    /// How long this load has had no request in flight, or `None` while a
    /// request is running or the idle unloader holds it.
    pub(crate) fn idle_for(&self, now: Instant) -> Option<Duration> {
        (self.state.load(Ordering::Acquire) == 0).then(|| self.since_last_activity(now))
    }

    /// Claim this load for the idle unloader. Succeeds only when no request is
    /// in flight and the last one ended at least `timeout` before `now`. While
    /// the claim is held, [`Self::try_begin_request`] refuses new requests.
    pub(crate) fn try_claim_idle(&self, timeout: Duration, now: Instant) -> bool {
        if self
            .state
            .compare_exchange(0, IDLE_CLAIMED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        // A request may have started and finished since the caller last
        // looked; its guard updated the timestamp before releasing its count.
        if self.since_last_activity(now) < timeout {
            self.release_idle_claim();
            return false;
        }
        true
    }

    /// Give a claimed load back to request traffic, for example after the
    /// unloader failed to stop it.
    pub(crate) fn release_idle_claim(&self) {
        self.state.fetch_and(!IDLE_CLAIMED, Ordering::AcqRel);
    }

    fn since_last_activity(&self, now: Instant) -> Duration {
        let last = self.origin + Duration::from_millis(self.last_active_ms.load(Ordering::Acquire));
        now.saturating_duration_since(last)
    }

    fn touch(&self, now: Instant) {
        let elapsed_ms = now.saturating_duration_since(self.origin).as_millis();
        self.last_active_ms.fetch_max(
            u64::try_from(elapsed_ms).unwrap_or(u64::MAX),
            Ordering::AcqRel,
        );
    }
}

/// Keeps one request counted as in flight. Streaming responses hold it until
/// the stream is dropped, so an open stream keeps its model loaded.
pub(crate) struct ModelActivityGuard {
    activity: Arc<ModelActivity>,
}

impl Drop for ModelActivityGuard {
    fn drop(&mut self) {
        // Record the end time before releasing the count, so an idle check
        // that sees zero requests also sees this request's end time.
        self.activity.touch(Instant::now());
        self.activity.state.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(60);

    #[test]
    fn a_new_load_is_idle_from_installation() {
        let activity = ModelActivity::new();
        let later = activity.origin + Duration::from_secs(90);

        assert_eq!(activity.idle_for(later), Some(Duration::from_secs(90)));
        assert!(activity.try_claim_idle(TIMEOUT, later));
    }

    #[test]
    fn an_in_flight_request_blocks_the_idle_claim() {
        let activity = ModelActivity::new();
        let far_future = activity.origin + TIMEOUT * 10;
        let guard = activity.try_begin_request().expect("request admitted");

        assert_eq!(activity.in_flight(), 1);
        assert_eq!(activity.idle_for(far_future), None);
        assert!(!activity.try_claim_idle(TIMEOUT, far_future));

        drop(guard);
        assert_eq!(activity.in_flight(), 0);
        assert!(activity.try_claim_idle(TIMEOUT, far_future));
    }

    #[test]
    fn idle_time_restarts_when_a_request_ends() {
        let activity = ModelActivity::new();
        drop(activity.try_begin_request().expect("request admitted"));
        let ended = Instant::now();

        assert!(!activity.try_claim_idle(TIMEOUT, ended + TIMEOUT / 2));
        // The failed claim must not leave the load closed to traffic.
        assert!(activity.try_begin_request().is_some());
        assert!(activity.try_claim_idle(TIMEOUT, ended + TIMEOUT * 2));
    }

    #[test]
    fn a_claimed_load_refuses_requests_until_released() {
        let activity = ModelActivity::new();
        let later = activity.origin + TIMEOUT * 2;
        assert!(activity.try_claim_idle(TIMEOUT, later));

        assert!(activity.try_begin_request().is_none());
        assert_eq!(activity.idle_for(later), None);
        assert!(!activity.try_claim_idle(TIMEOUT, later));

        activity.release_idle_claim();
        let guard = activity.try_begin_request().expect("request admitted");
        assert_eq!(activity.in_flight(), 1);
        drop(guard);
    }
}
