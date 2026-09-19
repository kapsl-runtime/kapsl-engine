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

const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_CHECKSUM_WORKERS: usize = 4;
const MIN_PARALLEL_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) enum Algorithm {
    Sha256,
    Blake3,
}

pub(super) fn sha256_file(path: &Path) -> io::Result<String> {
    hash_file(path, Algorithm::Sha256)
}

fn hash_file(path: &Path, algorithm: Algorithm) -> io::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut sha256 = Sha256::new();
    let mut blake3 = blake3::Hasher::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        match algorithm {
            Algorithm::Sha256 => sha256.update(&buffer[..read]),
            Algorithm::Blake3 => {
                blake3.update(&buffer[..read]);
            }
        }
    }
    Ok(match algorithm {
        Algorithm::Sha256 => format!("{:x}", sha256.finalize()),
        Algorithm::Blake3 => blake3.finalize().to_hex().to_string(),
    })
}

pub(super) fn verify_installed_files(
    root: &Path,
    files: &BTreeMap<String, String>,
    algorithm: Algorithm,
) -> io::Result<bool> {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    verify_with_workers(root, files, available, algorithm)
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
    algorithm: Algorithm,
) -> io::Result<bool> {
    verify_using(root, files, available_workers, algorithm, hash_file)
}

fn verify_using(
    root: &Path,
    files: &BTreeMap<String, String>,
    available_workers: usize,
    algorithm: Algorithm,
    hash: impl Fn(&Path, Algorithm) -> io::Result<String> + Sync,
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
            if !hash(&check.path, algorithm)?.eq_ignore_ascii_case(check.digest) {
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
    let worker = || -> io::Result<bool> {
        while !failed.load(Ordering::Relaxed) {
            let Some(check) = checks.get(next.fetch_add(1, Ordering::Relaxed)) else {
                break;
            };
            match hash(&check.path, algorithm) {
                Ok(actual) if actual.eq_ignore_ascii_case(check.digest) => {}
                Ok(_) => {
                    failed.store(true, Ordering::Relaxed);
                    return Ok(false);
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
        for _ in 1..workers {
            match std::thread::Builder::new()
                .name("kapsl-pack-checksum".into())
                .spawn_scoped(scope, worker)
            {
                Ok(handle) => handles.push(handle),
                // The caller also drains the work queue. A thread limit only
                // reduces parallelism; it cannot skip any required checksum.
                Err(_) => break,
            }
        }
        let mut result = worker();
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

    fn fixture(algorithm: Algorithm) -> (tempfile::TempDir, BTreeMap<String, String>) {
        let root = tempfile::tempdir().unwrap();
        let mut files = BTreeMap::new();
        for (index, bytes) in [20 * 1024 * 1024, 12 * 1024 * 1024, 17, 0]
            .into_iter()
            .enumerate()
        {
            let name = format!("file-{index}");
            let path = root.path().join(&name);
            File::create(&path).unwrap().set_len(bytes).unwrap();
            files.insert(name, hash_file(&path, algorithm).unwrap());
        }
        (root, files)
    }

    #[test]
    fn verifies_large_small_and_empty_files_with_bounded_parallelism() {
        for algorithm in [Algorithm::Sha256, Algorithm::Blake3] {
            let (root, mut files) = fixture(algorithm);
            // Hexadecimal digest case remains compatible with signed manifests.
            for digest in files.values_mut() {
                *digest = digest.to_ascii_uppercase();
            }
            assert!(verify_with_workers(root.path(), &files, 1, algorithm).unwrap());
            assert!(verify_with_workers(root.path(), &files, usize::MAX, algorithm).unwrap());
        }
    }

    #[test]
    fn rejects_each_tampered_file_after_successful_verification() {
        for algorithm in [Algorithm::Sha256, Algorithm::Blake3] {
            let (root, files) = fixture(algorithm);
            assert!(verify_with_workers(root.path(), &files, 4, algorithm).unwrap());
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
                    !verify_with_workers(root.path(), &files, 4, algorithm).unwrap(),
                    "{name}"
                );
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.write_all(&[0]).unwrap();
                file.set_len(original_len).unwrap();
                assert!(
                    verify_with_workers(root.path(), &files, 4, algorithm).unwrap(),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn rejects_missing_or_non_file_entries() {
        for algorithm in [Algorithm::Sha256, Algorithm::Blake3] {
            let (root, mut files) = fixture(algorithm);
            files.insert("missing".into(), "00".repeat(32));
            assert!(!verify_with_workers(root.path(), &files, 4, algorithm).unwrap());
            std::fs::create_dir(root.path().join("missing")).unwrap();
            assert!(!verify_with_workers(root.path(), &files, 4, algorithm).unwrap());
        }
    }
    #[test]
    fn concurrent_blake3_checks_reject_same_size_replacement() {
        let (root, files) = fixture(Algorithm::Blake3);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    assert!(verify_with_workers(root.path(), &files, 4, Algorithm::Blake3).unwrap())
                });
            }
        });
        let path = root.path().join("file-2");
        let modified = path.metadata().unwrap().modified().unwrap();
        let replacement = root.path().join("replacement");
        std::fs::write(&replacement, [1_u8; 17]).unwrap();
        File::options()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        // Explicit remove also permits the test on Windows.
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(replacement, &path).unwrap();
        assert!(!verify_with_workers(root.path(), &files, 4, Algorithm::Blake3).unwrap());
    }

    #[test]
    fn read_errors_and_worker_panics_fail_closed() {
        let (root, files) = fixture(Algorithm::Blake3);
        let error = verify_using(root.path(), &files, 4, Algorithm::Blake3, |_, _| {
            Err(io::Error::other("injected read error"))
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected read error"));
        // Wait at most two seconds for a worker to reach the injected panic.
        // The caller succeeds, so the error must come from joining the worker.
        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = std::sync::Mutex::new(receiver);
        let caller_waited = AtomicBool::new(false);
        let error = verify_using(
            root.path(),
            &files,
            2,
            Algorithm::Blake3,
            |path, algorithm| {
                if std::thread::current().name() == Some("kapsl-pack-checksum") {
                    sender.send(()).unwrap();
                    panic!("injected worker panic");
                }
                if !caller_waited.swap(true, Ordering::SeqCst) {
                    receiver
                        .lock()
                        .unwrap()
                        .recv_timeout(std::time::Duration::from_secs(2))
                        .expect("checksum worker must start for panic injection");
                }
                hash_file(path, algorithm)
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("pack checksum worker panicked"));
    }
}
