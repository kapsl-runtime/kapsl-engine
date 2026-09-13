//! Bounded verification of every file covered by an installed pack's manifest.
//!
//! Verification is repeated for each load. File metadata only chooses how to
//! schedule reads; it is never accepted as evidence that content is unchanged.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_CHECKSUM_WORKERS: usize = 4;
const MIN_PARALLEL_BYTES: u64 = 32 * 1024 * 1024;

/// Diagnostic wall time, including scheduling/preemption, not CPU or disk time.
#[derive(Debug, Default)]
pub(super) struct FileTiming {
    pub bytes: u64,
    pub read: Duration,
    pub hash: Duration,
    pub elapsed: Duration,
}

pub(super) fn sha256_file(path: &Path) -> io::Result<String> {
    Ok(hash_file::<false>(path)?.0)
}

fn hash_file<const PROFILE: bool>(path: &Path) -> io::Result<(String, FileTiming)> {
    let started = PROFILE.then(Instant::now);
    let mut timing = FileTiming::default();
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read_started = PROFILE.then(Instant::now);
        let read = reader.read(&mut buffer)?;
        if let Some(started) = read_started {
            timing.read += started.elapsed();
            timing.bytes += read as u64;
        }
        if read == 0 {
            break;
        }
        let hash_started = PROFILE.then(Instant::now);
        hasher.update(&buffer[..read]);
        if let Some(started) = hash_started {
            timing.hash += started.elapsed();
        }
    }
    let hash_started = PROFILE.then(Instant::now);
    let digest = hasher.finalize();
    if let Some(started) = hash_started {
        timing.hash += started.elapsed();
    }
    if let Some(started) = started {
        timing.elapsed = started.elapsed();
    }
    Ok((format!("{digest:x}"), timing))
}

pub(super) fn verify_installed_files(
    root: &Path,
    files: &BTreeMap<String, String>,
) -> io::Result<bool> {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    verify_with_workers(root, files, available)
}

struct FileCheck<'a> {
    path: PathBuf,
    digest: &'a str,
    bytes: u64,
}

fn verify_with_workers(
    root: &Path,
    files: &BTreeMap<String, String>,
    available_workers: usize,
) -> io::Result<bool> {
    verify_observed::<false>(root, files, available_workers, &|_, _, _, _| {})
}

/// Uses the production scheduler and reader. The observer runs once per file;
/// callers must keep diagnostics separate from qualification measurements.
pub(super) fn verify_installed_files_profiled(
    root: &Path,
    files: &BTreeMap<String, String>,
    observer: &(impl Fn(usize, &Path, &FileTiming, bool) + Sync),
) -> io::Result<bool> {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    verify_observed::<true>(root, files, available, observer)
}

fn verify_observed<const PROFILE: bool>(
    root: &Path,
    files: &BTreeMap<String, String>,
    available_workers: usize,
    observer: &(impl Fn(usize, &Path, &FileTiming, bool) + Sync),
) -> io::Result<bool> {
    let mut checks = Vec::with_capacity(files.len());
    let mut total_bytes = 0_u64;
    for (relative, digest) in files {
        let path = root.join(relative);
        let metadata = match path.metadata() {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => return Ok(false),
        };
        total_bytes = total_bytes.saturating_add(metadata.len());
        checks.push(FileCheck {
            path,
            digest,
            bytes: metadata.len(),
        });
    }

    let workers = available_workers
        .clamp(1, MAX_CHECKSUM_WORKERS)
        .min(checks.len());
    if workers <= 1 || total_bytes < MIN_PARALLEL_BYTES {
        for check in checks {
            let (actual, timing) = hash_file::<PROFILE>(&check.path)?;
            let valid = actual.eq_ignore_ascii_case(check.digest);
            observer(0, &check.path, &timing, valid);
            if !valid {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    // Start the largest libraries first so that one large final file does not
    // leave the other workers idle. At most four 1 MiB read buffers are live.
    checks.sort_by_key(|check| std::cmp::Reverse(check.bytes));
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let worker = |worker_id| -> io::Result<bool> {
        while !failed.load(Ordering::Relaxed) {
            let Some(check) = checks.get(next.fetch_add(1, Ordering::Relaxed)) else {
                break;
            };
            match hash_file::<PROFILE>(&check.path) {
                Ok((actual, timing)) => {
                    let valid = actual.eq_ignore_ascii_case(check.digest);
                    observer(worker_id, &check.path, &timing, valid);
                    if !valid {
                        failed.store(true, Ordering::Relaxed);
                        return Ok(false);
                    }
                }
                Err(error) => {
                    failed.store(true, Ordering::Relaxed);
                    return Err(error);
                }
            }
        }
        Ok(true)
    };

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers - 1);
        for worker_id in 1..workers {
            let worker = &worker;
            match std::thread::Builder::new()
                .name("kapsl-pack-checksum".into())
                .spawn_scoped(scope, move || worker(worker_id))
            {
                Ok(handle) => handles.push(handle),
                // The caller also drains the work queue. A thread limit only
                // reduces parallelism; it cannot skip any required checksum.
                Err(_) => break,
            }
        }
        let mut result = worker(0);
        for handle in handles {
            let joined = handle
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("pack checksum worker panicked")));
            result = match (result, joined) {
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(left), Ok(right)) => Ok(left && right),
            };
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    fn fixture() -> (tempfile::TempDir, BTreeMap<String, String>) {
        let root = tempfile::tempdir().unwrap();
        let mut files = BTreeMap::new();
        for (index, bytes) in [20 * 1024 * 1024, 12 * 1024 * 1024, 17, 0]
            .into_iter()
            .enumerate()
        {
            let name = format!("file-{index}");
            let path = root.path().join(&name);
            File::create(&path).unwrap().set_len(bytes).unwrap();
            files.insert(name, sha256_file(&path).unwrap());
        }
        (root, files)
    }

    #[test]
    fn verifies_large_small_and_empty_files_with_bounded_parallelism() {
        let (root, mut files) = fixture();
        // Hexadecimal digest case remains compatible with signed manifests.
        for digest in files.values_mut() {
            *digest = digest.to_ascii_uppercase();
        }
        assert!(verify_with_workers(root.path(), &files, 1).unwrap());
        assert!(verify_with_workers(root.path(), &files, usize::MAX).unwrap());
    }

    #[test]
    fn rejects_each_tampered_file_after_successful_verification() {
        let (root, files) = fixture();
        assert!(verify_with_workers(root.path(), &files, 4).unwrap());
        for name in files.keys() {
            let path = root.path().join(name);
            let mut file = File::options().read(true).write(true).open(&path).unwrap();
            let metadata = file.metadata().unwrap();
            let original_len = metadata.len();
            let modified = metadata.modified().unwrap();
            let offset = original_len.saturating_sub(1);
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&[1]).unwrap();
            file.set_times(std::fs::FileTimes::new().set_modified(modified))
                .unwrap();
            // Restoring mtime (and retaining length for nonempty files) must
            // not hide a change, including after the large libraries.
            assert!(
                !verify_with_workers(root.path(), &files, 4).unwrap(),
                "{name}"
            );
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&[0]).unwrap();
            file.set_len(original_len).unwrap();
            assert!(
                verify_with_workers(root.path(), &files, 4).unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn rejects_missing_or_non_file_entries() {
        let (root, mut files) = fixture();
        files.insert("missing".into(), "00".repeat(32));
        assert!(!verify_with_workers(root.path(), &files, 4).unwrap());
        std::fs::create_dir(root.path().join("missing")).unwrap();
        assert!(!verify_with_workers(root.path(), &files, 4).unwrap());
    }
    #[test]
    fn diagnostic_reads_all_bytes_and_rejects_changed_digest_and_replacement() {
        let (root, mut files) = fixture();
        let observed = std::sync::Mutex::new(BTreeMap::new());
        assert!(
            verify_installed_files_profiled(root.path(), &files, &|_, path, timing, valid| {
                assert!(valid);
                assert!(timing.elapsed >= timing.read + timing.hash);
                observed.lock().unwrap().insert(
                    path.file_name().unwrap().to_str().unwrap().to_string(),
                    timing.bytes,
                );
            })
            .unwrap()
        );
        let observed = observed.into_inner().unwrap();
        assert_eq!(observed.len(), files.len());
        for name in files.keys() {
            assert_eq!(
                observed[name],
                root.path().join(name).metadata().unwrap().len()
            );
        }
        files.insert("file-2".into(), "00".repeat(32));
        assert!(!verify_installed_files_profiled(root.path(), &files, &|_, _, _, _| {}).unwrap());
        files.insert(
            "file-2".into(),
            sha256_file(&root.path().join("file-2")).unwrap(),
        );
        let replacement = root.path().join("replacement");
        std::fs::write(&replacement, [1; 17]).unwrap();
        std::fs::remove_file(root.path().join("file-2")).unwrap();
        std::fs::rename(replacement, root.path().join("file-2")).unwrap();
        assert!(!verify_installed_files_profiled(root.path(), &files, &|_, _, _, _| {}).unwrap());
    }

    #[test]
    fn concurrent_validations_each_cover_the_entire_manifest() {
        let (root, files) = fixture();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(|| {
                        let count = AtomicUsize::new(0);
                        assert!(verify_observed::<true>(
                            root.path(),
                            &files,
                            4,
                            &|_, _, _, valid| {
                                assert!(valid);
                                count.fetch_add(1, Ordering::Relaxed);
                            }
                        )
                        .unwrap());
                        assert_eq!(count.load(Ordering::Relaxed), files.len());
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn sha256_known_vectors() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("vector");
        for (bytes, expected) in [
            (
                b"".as_slice(),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc".as_slice(),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
        ] {
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(sha256_file(&path).unwrap(), expected);
            assert_eq!(hash_file::<true>(&path).unwrap().0, expected);
        }
    }
    #[test]
    fn worker_panic_is_an_error_and_other_workers_are_joined() {
        let (root, files) = fixture();
        let barrier = std::sync::Barrier::new(2);
        let visits = AtomicUsize::new(0);
        let result = verify_observed::<true>(root.path(), &files, 2, &|worker, _, _, _| {
            if visits.fetch_add(1, Ordering::Relaxed) < 2 {
                barrier.wait();
            }
            if worker == 1 {
                panic!("injected worker failure");
            }
        });
        assert_eq!(
            result.unwrap_err().to_string(),
            "pack checksum worker panicked"
        );
        assert!(verify_with_workers(root.path(), &files, 2).unwrap());
    }

    #[test]
    fn file_read_errors_are_not_successful_checksums() {
        let root = tempfile::tempdir().unwrap();
        // Opening or reading a directory fails across the supported hosts.
        assert!(hash_file::<true>(root.path()).is_err());
        assert!(sha256_file(root.path()).is_err());
    }
}
