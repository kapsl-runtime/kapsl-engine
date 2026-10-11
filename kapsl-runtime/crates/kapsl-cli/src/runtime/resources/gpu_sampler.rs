//! Non-blocking GPU telemetry source for the runtime monitor.
//!
//! `nvidia-smi` can take 1-2 s on managed GPU platforms (Cloud Run L4 measured
//! 1.4-1.8 s), so it must never run inline on a tokio worker. Samples come from
//! NVML when it is available and fall back to a bounded `nvidia-smi` call on a
//! blocking thread, with a per-call timeout and a failure backoff.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Env var overriding the timeout applied to GPU telemetry probes.
pub(crate) const NVIDIA_SMI_TIMEOUT_ENV: &str = "KAPSL_NVIDIA_SMI_TIMEOUT_MS";
const DEFAULT_NVIDIA_SMI_TIMEOUT: Duration = Duration::from_millis(5000);
/// How long to stop sampling after a failed or timed-out probe.
pub(crate) const GPU_SAMPLE_BACKOFF: Duration = Duration::from_secs(30);

/// `(average utilization 0.0-1.0, used VRAM bytes, total VRAM bytes)`.
pub(crate) type GpuSample = (f64, usize, usize);
/// A blocking sample source. Runs on a blocking thread, never on the runtime.
pub(crate) type GpuSampleSource = Arc<dyn Fn() -> Option<GpuSample> + Send + Sync>;

pub(crate) fn parse_timeout_ms(raw: Option<&str>) -> Duration {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_NVIDIA_SMI_TIMEOUT)
}

pub(crate) fn nvidia_smi_timeout() -> Duration {
    parse_timeout_ms(std::env::var(NVIDIA_SMI_TIMEOUT_ENV).ok().as_deref())
}

/// Run `program` and return its stdout, or `None` on spawn failure, non-zero
/// exit, or when `timeout` elapses (the child is killed so a wedged driver
/// call does not leave processes behind).
pub(crate) fn run_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut bytes = Vec::new();
        let _ = stdout.read_to_end(&mut bytes);
        buf.push_str(&String::from_utf8_lossy(&bytes));
        buf
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return None;
            }
        }
    };
    let out = reader.join().ok()?;
    status.success().then_some(out)
}

/// Run blocking `work` on the blocking pool, giving up after `timeout`. The
/// blocking thread cannot be cancelled, so callers keep a backoff to bound how
/// many abandoned threads can accumulate.
pub(crate) async fn run_blocking_with_timeout<T, F>(timeout: Duration, work: F) -> Option<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    match tokio::time::timeout(timeout, tokio::task::spawn_blocking(work)).await {
        Ok(Ok(value)) => Some(value),
        _ => None,
    }
}

fn shared_nvml() -> Option<&'static nvml_wrapper::Nvml> {
    static NVML: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();
    NVML.get_or_init(|| nvml_wrapper::Nvml::init().ok())
        .as_ref()
}

/// Format NVML's encoded CUDA driver version (`1000 * major + 10 * minor`,
/// e.g. 13000 -> "13.0") the way `nvidia-smi` prints it.
pub(crate) fn format_cuda_driver_version(encoded: i32) -> Option<String> {
    (encoded > 0).then(|| format!("{}.{}", encoded / 1000, (encoded % 1000) / 10))
}

/// Highest CUDA version the installed driver supports, via NVML.
pub(crate) fn nvml_cuda_version() -> Option<String> {
    format_cuda_driver_version(shared_nvml()?.sys_cuda_driver_version().ok()?)
}

/// Fill CUDA versions that device detection left empty. `lookup` (NVML in
/// production) is only invoked when at least one entry is missing, and a
/// failed lookup leaves every entry untouched.
pub(crate) fn fill_cuda_versions<'a>(
    versions: impl IntoIterator<Item = &'a mut Option<String>>,
    lookup: impl FnOnce() -> Option<String>,
) {
    let mut missing: Vec<_> = versions.into_iter().filter(|v| v.is_none()).collect();
    if missing.is_empty() {
        return;
    }
    if let Some(version) = lookup() {
        for slot in &mut missing {
            **slot = Some(version.clone());
        }
    }
}

/// Per-device `(utilization percent, used bytes, total bytes)` rows read
/// through NVML, in device-index order. `None` when NVML cannot be loaded or
/// no device could be read, so the caller falls back to `nvidia-smi`.
pub(crate) fn nvml_device_rows() -> Option<Vec<Option<(f64, usize, usize)>>> {
    let nvml = shared_nvml()?;
    let count = nvml.device_count().ok()?;
    let rows: Vec<_> = (0..count)
        .map(|index| {
            let device = nvml.device_by_index(index).ok()?;
            let util = device.utilization_rates().ok()?.gpu as f64;
            let mem = device.memory_info().ok()?;
            Some((util, mem.used as usize, mem.total as usize))
        })
        .collect();
    rows.iter().any(Option::is_some).then_some(rows)
}

/// Samples GPU telemetry off the async runtime, with a timeout and a backoff
/// after failure.
pub(crate) struct GpuSampler {
    source: GpuSampleSource,
    timeout: Duration,
    backoff: Duration,
    retry_after: Option<Instant>,
}

impl GpuSampler {
    pub(crate) fn new(source: GpuSampleSource, timeout: Duration, backoff: Duration) -> Self {
        Self {
            source,
            timeout,
            backoff,
            retry_after: None,
        }
    }

    /// True while a previous failure's backoff window is still open at `now`.
    pub(crate) fn backing_off(&self, now: Instant) -> bool {
        self.retry_after
            .is_some_and(|retry_after| now < retry_after)
    }

    /// One sample, or `None` while backing off or when the source fails or
    /// exceeds the timeout (which opens a new backoff window).
    pub(crate) async fn sample(&mut self, now: Instant) -> Option<GpuSample> {
        if self.backing_off(now) {
            return None;
        }
        let source = self.source.clone();
        let sample = run_blocking_with_timeout(self.timeout, move || source())
            .await
            .flatten();
        self.retry_after = match sample {
            Some(_) => None,
            None => Some(now + self.backoff),
        };
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const SAMPLE: GpuSample = (0.5, 1024, 4096);

    fn counting_source(
        calls: Arc<AtomicUsize>,
        delay: Duration,
        result: Option<GpuSample>,
    ) -> GpuSampleSource {
        Arc::new(move || {
            calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(delay);
            result
        })
    }

    #[tokio::test(flavor = "current_thread")]
    async fn slow_source_does_not_block_the_async_runtime() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut sampler = GpuSampler::new(
            counting_source(calls, Duration::from_millis(300), Some(SAMPLE)),
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticker = {
            let ticks = ticks.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    ticks.fetch_add(1, Ordering::SeqCst);
                }
            })
        };
        let sampled = sampler.sample(Instant::now()).await;
        ticker.abort();
        assert_eq!(sampled, Some(SAMPLE));
        // A blocking call on this single-threaded runtime would starve the ticker.
        assert!(ticks.load(Ordering::SeqCst) >= 5, "runtime was blocked");
    }

    #[tokio::test]
    async fn timeout_is_honored_and_triggers_backoff() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut sampler = GpuSampler::new(
            counting_source(calls, Duration::from_secs(3), Some(SAMPLE)),
            Duration::from_millis(50),
            Duration::from_secs(30),
        );
        let start = Instant::now();
        assert_eq!(sampler.sample(start).await, None);
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "timeout not honored"
        );
        assert!(sampler.backing_off(start + Duration::from_secs(1)));
    }

    #[tokio::test]
    async fn failure_backs_off_without_calling_source_then_retries() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut sampler = GpuSampler::new(
            counting_source(calls.clone(), Duration::ZERO, None),
            Duration::from_secs(5),
            Duration::from_secs(30),
        );
        let t0 = Instant::now();
        assert_eq!(sampler.sample(t0).await, None);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Inside the 30 s window the source is not touched.
        assert!(sampler.backing_off(t0 + Duration::from_secs(29)));
        assert_eq!(sampler.sample(t0 + Duration::from_secs(29)).await, None);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // After the window it is retried.
        assert!(!sampler.backing_off(t0 + Duration::from_secs(31)));
        assert_eq!(sampler.sample(t0 + Duration::from_secs(31)).await, None);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn success_clears_backoff() {
        let ok = Arc::new(AtomicUsize::new(0));
        let flip = ok.clone();
        let source: GpuSampleSource =
            Arc::new(move || (flip.load(Ordering::SeqCst) == 1).then_some(SAMPLE));
        let mut sampler = GpuSampler::new(source, Duration::from_secs(5), Duration::from_secs(30));
        let t0 = Instant::now();
        assert_eq!(sampler.sample(t0).await, None);
        ok.store(1, Ordering::SeqCst);
        let t1 = t0 + Duration::from_secs(31);
        assert_eq!(sampler.sample(t1).await, Some(SAMPLE));
        assert!(!sampler.backing_off(t1));
    }

    #[test]
    fn command_runner_kills_a_hung_child_at_the_deadline() {
        let start = Instant::now();
        let out = run_command_with_timeout("sleep", &["5"], Duration::from_millis(100));
        assert!(out.is_none());
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn command_runner_returns_stdout_on_success() {
        let out = run_command_with_timeout("echo", &["hello"], Duration::from_secs(5));
        assert_eq!(out.as_deref().map(str::trim), Some("hello"));
    }

    #[test]
    fn command_runner_none_on_failure_or_missing_binary() {
        assert!(run_command_with_timeout("false", &[], Duration::from_secs(5)).is_none());
        assert!(
            run_command_with_timeout("definitely-not-a-binary", &[], Duration::from_secs(5))
                .is_none()
        );
    }

    #[test]
    fn timeout_env_parsing_defaults_on_missing_zero_or_garbage() {
        assert_eq!(parse_timeout_ms(None), Duration::from_millis(5000));
        assert_eq!(parse_timeout_ms(Some("0")), Duration::from_millis(5000));
        assert_eq!(parse_timeout_ms(Some("abc")), Duration::from_millis(5000));
        assert_eq!(
            parse_timeout_ms(Some(" 1200 ")),
            Duration::from_millis(1200)
        );
    }

    #[test]
    fn cuda_driver_version_formats_as_major_minor() {
        assert_eq!(format_cuda_driver_version(13000).as_deref(), Some("13.0"));
        assert_eq!(format_cuda_driver_version(12060).as_deref(), Some("12.6"));
        assert_eq!(format_cuda_driver_version(12040).as_deref(), Some("12.4"));
        assert_eq!(format_cuda_driver_version(11080).as_deref(), Some("11.8"));
        assert_eq!(format_cuda_driver_version(0), None);
        assert_eq!(format_cuda_driver_version(-1), None);
    }

    #[test]
    fn fill_cuda_versions_sets_only_missing_entries() {
        let mut versions = [None, Some("12.1".to_string()), None];
        fill_cuda_versions(versions.iter_mut(), || Some("13.0".to_string()));
        assert_eq!(
            versions,
            [
                Some("13.0".to_string()),
                Some("12.1".to_string()),
                Some("13.0".to_string())
            ]
        );
    }

    #[test]
    fn fill_cuda_versions_leaves_none_when_nvml_unavailable() {
        let mut versions = [None, None];
        fill_cuda_versions(versions.iter_mut(), || None);
        assert_eq!(versions, [None, None]);
    }

    #[test]
    fn fill_cuda_versions_does_not_query_nvml_when_nothing_is_missing() {
        let mut versions = [Some("12.1".to_string())];
        fill_cuda_versions(versions.iter_mut(), || panic!("NVML must not be queried"));
        assert_eq!(versions, [Some("12.1".to_string())]);
    }
}
