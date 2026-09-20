//! Diagnostic-only wall timings. Read time includes the OS/page cache, not
//! necessarily physical disk I/O. Per-worker intervals overlap and must not be
//! added to the validation critical path.

use super::*;
use serde::Serialize;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;
use std::time::Instant;

static NEXT_VALIDATION: AtomicU64 = AtomicU64::new(1);

#[derive(Default, Serialize)]
pub(super) struct HashMeasurements {
    pub(super) bytes_read: u64,
    pub(super) open_ns: u64,
    pub(super) read_ns: u64,
    pub(super) hash_ns: u64,
}

pub(super) fn measure<const ENABLED: bool, T>(elapsed: &mut u64, work: impl FnOnce() -> T) -> T {
    if ENABLED {
        let start = Instant::now();
        let result = work();
        *elapsed = elapsed.saturating_add(ns(start));
        result
    } else {
        work()
    }
}

fn ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

#[derive(Serialize)]
struct FileRecord {
    path: String,
    worker: String,
    started_ns: u64,
    finished_ns: u64,
    outcome: &'static str,
    #[serde(flatten)]
    measurements: HashMeasurements,
}

#[derive(Serialize)]
pub(super) struct Report {
    schema_version: u32,
    process_id: u32,
    validation_id: u64,
    root: String,
    algorithm: &'static str,
    expected_files: usize,
    available_workers: usize,
    elapsed_ns: u64,
    outcome: &'static str,
    files: Vec<FileRecord>,
}

pub(super) fn verify(
    root: &Path,
    files: &BTreeMap<String, String>,
    available_workers: usize,
    algorithm: Algorithm,
) -> (io::Result<bool>, Report) {
    let validation_id = NEXT_VALIDATION.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let records = Mutex::new(Vec::with_capacity(files.len()));
    // Preserve the manifest spelling, including Windows separators. Path
    // component normalization is not a reversible lookup into the signed map.
    let manifest_paths: BTreeMap<_, _> = files
        .iter()
        .map(|(relative, digest)| (root.join(relative).into_os_string(), (relative, digest)))
        .collect();
    let result = verify_using(
        root,
        files,
        available_workers,
        algorithm,
        |path, algorithm| {
            let started_ns = ns(started);
            let (result, measurements) = hash_file_measured::<true>(path, algorithm);
            let finished_ns = ns(started);
            let (relative, expected) = manifest_paths
                .get(path.as_os_str())
                .expect("verifier only hashes paths from this manifest");
            let outcome = match &result {
                Ok(digest) if digest.eq_ignore_ascii_case(expected) => "matched",
                Ok(_) => "mismatch",
                Err(_) => "io_error",
            };
            records
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(FileRecord {
                    path: (*relative).clone(),
                    worker: format!("{:?}", std::thread::current().id()),
                    started_ns,
                    finished_ns,
                    outcome,
                    measurements,
                });
            result
        },
    );
    let elapsed_ns = ns(started);
    let mut records = records.into_inner().unwrap_or_else(|p| p.into_inner());
    records.sort_by_key(|record| record.started_ns);
    let report = Report {
        schema_version: 1,
        process_id: std::process::id(),
        validation_id,
        root: root.display().to_string(),
        algorithm: match algorithm {
            Algorithm::Sha256 => "sha256",
            Algorithm::Blake3 => "blake3",
        },
        expected_files: files.len(),
        available_workers,
        elapsed_ns,
        outcome: match &result {
            Ok(true) => "valid",
            Ok(false) => "invalid",
            Err(_) => "io_error",
        },
        files: records,
    };
    (result, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiling_preserves_digests_and_accounts_for_all_bytes() {
        for algorithm in [Algorithm::Sha256, Algorithm::Blake3] {
            let (root, files) = super::super::tests::fixture(algorithm);
            let (result, report) = verify(root.path(), &files, 4, algorithm);
            assert!(result.unwrap());
            assert_eq!(report.outcome, "valid");
            assert_eq!(report.files.len(), files.len());
            let mut workers = std::collections::HashSet::new();
            for record in &report.files {
                let path = root.path().join(&record.path);
                assert_eq!(
                    record.measurements.bytes_read,
                    path.metadata().unwrap().len()
                );
                assert_eq!(record.outcome, "matched");
                assert!(record.started_ns <= record.finished_ns);
                assert!(record.finished_ns <= report.elapsed_ns);
                let wall = record.finished_ns - record.started_ns;
                assert!(
                    record.measurements.open_ns
                        + record.measurements.read_ns
                        + record.measurements.hash_ns
                        <= wall
                );
                workers.insert(&record.worker);
                assert_eq!(hash_file(&path, algorithm).unwrap(), files[&record.path]);
            }
            assert!(workers.len() <= MAX_CHECKSUM_WORKERS);
            let encoded = serde_json::to_value(&report).unwrap();
            assert_eq!(encoded["expected_files"], files.len());
        }
    }

    #[test]
    fn profiled_validation_rejects_tampering_and_missing_files() {
        let (root, files) = super::super::tests::fixture(Algorithm::Blake3);
        assert!(verify(root.path(), &files, 4, Algorithm::Blake3).0.unwrap());
        std::fs::write(root.path().join("file-2"), [1; 17]).unwrap();
        let (result, report) = verify(root.path(), &files, 4, Algorithm::Blake3);
        assert!(!result.unwrap());
        assert_eq!(report.outcome, "invalid");
        assert!(report
            .files
            .iter()
            .any(|r| r.path == "file-2" && r.outcome == "mismatch"));
        std::fs::remove_file(root.path().join("file-2")).unwrap();
        let (result, report) = verify(root.path(), &files, 4, Algorithm::Blake3);
        assert!(!result.unwrap());
        assert_eq!(report.outcome, "invalid");
        // Metadata preflight rejects before any hashes. A short report does
        // not imply skipped files were accepted.
        assert!(report.files.is_empty());
        assert_eq!(report.expected_files, files.len());
    }

    #[test]
    fn nested_manifest_paths_retain_the_signed_spelling() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("nested")).unwrap();
        std::fs::write(root.path().join("nested/data"), b"test").unwrap();
        let spellings = vec!["nested/data", "nested//data"];
        #[cfg(windows)]
        let spellings = {
            let mut spellings = spellings;
            spellings.push(r"nested\data");
            spellings
        };
        for relative in spellings {
            let files =
                BTreeMap::from([(relative.into(), blake3::hash(b"test").to_hex().to_string())]);
            let (result, report) = verify(root.path(), &files, 1, Algorithm::Blake3);
            assert!(result.unwrap());
            assert_eq!(report.files[0].path, relative);
            assert_eq!(report.files[0].outcome, "matched");
        }
    }

    #[test]
    fn concurrent_validation_reports_do_not_share_records() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("data"), b"test").unwrap();
        let files = BTreeMap::from([("data".into(), blake3::hash(b"test").to_hex().to_string())]);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| verify(root.path(), &files, 4, Algorithm::Blake3)))
                .collect();
            let mut ids = std::collections::HashSet::new();
            for handle in handles {
                let (result, report) = handle.join().unwrap();
                assert!(result.unwrap());
                assert!(ids.insert(report.validation_id));
                assert_eq!(report.files.len(), 1);
                assert_eq!(report.files[0].measurements.bytes_read, 4);
            }
        });
    }

    #[test]
    fn measured_read_errors_remain_errors_and_disabled_timing_is_empty() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("missing");
        for algorithm in [Algorithm::Sha256, Algorithm::Blake3] {
            let (result, record) = hash_file_measured::<true>(&missing, algorithm);
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
            assert_eq!(record.bytes_read, 0);
            assert_eq!(record.hash_ns, 0);
            let path = root.path().join("present");
            std::fs::write(&path, b"test").unwrap();
            let (result, record) = hash_file_measured::<false>(&path, algorithm);
            assert!(result.is_ok());
            assert_eq!(
                record.open_ns + record.read_ns + record.hash_ns + record.bytes_read,
                0
            );
        }
    }
}
