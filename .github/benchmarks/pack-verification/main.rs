//! Manual CPU-only timing of the production installed-file verifier.

#[path = "../../../kapsl-runtime/crates/kapsl-cli/src/backend/manager/checksums.rs"]
mod checksums;

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

fn verify_serial(root: &Path, files: &BTreeMap<String, String>) -> std::io::Result<bool> {
    // The previous installed_pack_is_valid loop, using the same SHA-256 code
    // as the candidate. Keep filesystem and digest validation in both routes.
    for (relative, expected) in files {
        let path = root.join(relative);
        if !path.is_file() || checksums::sha256_file(&path)? != expected.to_ascii_lowercase() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err(
            "usage: kapsl-pack-verification-bench PACK_ROOT CHECKSUMS_JSON [ABBA_BLOCKS]".into(),
        );
    }
    let root = Path::new(&args[0]);
    let files: BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if files.is_empty() {
        return Err("the checksum map must contain at least one file".into());
    }
    let blocks = args.get(2).map_or(Ok(10_usize), |value| value.parse())?;
    if !(1..=100).contains(&blocks) {
        return Err("ABBA_BLOCKS must be between 1 and 100".into());
    }
    let bytes = files.keys().try_fold(0_u64, |total, name| {
        Ok::<_, std::io::Error>(total.saturating_add(root.join(name).metadata()?.len()))
    })?;
    if !verify_serial(root, &files)? || !checksums::verify_installed_files(root, &files)? {
        return Err("input files failed checksum verification before measurement".into());
    }

    let mut serial_ms = Vec::with_capacity(blocks * 2);
    let mut parallel_ms = Vec::with_capacity(blocks * 2);
    let mut samples = Vec::with_capacity(blocks * 4);
    for block in 0..blocks {
        for parallel in [false, true, true, false] {
            let started = Instant::now();
            let valid = if parallel {
                checksums::verify_installed_files(root, &files)?
            } else {
                verify_serial(root, &files)?
            };
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            if !valid {
                return Err("input files changed or failed verification during measurement".into());
            }
            if parallel {
                parallel_ms.push(elapsed_ms);
            } else {
                serial_ms.push(elapsed_ms);
            }
            samples.push(serde_json::json!({
                "block": block + 1,
                "variant": if parallel { "bounded_parallel" } else { "serial" },
                "elapsed_ms": elapsed_ms,
            }));
        }
    }
    let serial_median_ms = median(&mut serial_ms);
    let parallel_median_ms = median(&mut parallel_ms);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "os": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "available_parallelism": std::thread::available_parallelism()?.get(),
            "files": files.len(),
            "bytes": bytes,
            "abba_blocks": blocks,
            "serial_median_ms": serial_median_ms,
            "parallel_median_ms": parallel_median_ms,
            "parallel_over_serial": parallel_median_ms / serial_median_ms,
            "samples": samples,
        }))?
    );
    Ok(())
}
