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

pub(super) fn sha256_file(path: &Path) -> io::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
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
            if !sha256_file(&check.path)?.eq_ignore_ascii_case(check.digest) {
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
            match sha256_file(&check.path) {
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
                .spawn_scoped(scope, &worker)
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
}
