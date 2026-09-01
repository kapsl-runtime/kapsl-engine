use crate::app::Ansi;
use std::fs;
use std::path::Path;
use std::time::Duration;

fn format_human_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit_index = 0usize;
    while value >= 1024.0 && unit_index + 1 < units.len() {
        value /= 1024.0;
        unit_index += 1;
    }
    if unit_index == 0 {
        format!("{}{}", bytes, units[unit_index])
    } else if (value - value.round()).abs() < 0.05 {
        format!("{:.0}{}", value, units[unit_index])
    } else {
        format!("{:.1}{}", value, units[unit_index])
    }
}

pub(super) fn print_build_summary(package_path: &str, metadata_path: Option<&str>) {
    let ansi = Ansi::new();
    let display_name = Path::new(package_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(package_path);
    match fs::metadata(package_path) {
        Ok(metadata) => eprintln!(
            "  {}  {} {}",
            ansi.green("✓"),
            display_name,
            ansi.dim(&format!("({})", format_human_bytes(metadata.len())))
        ),
        Err(_) => eprintln!("  {}  {}", ansi.green("✓"), display_name),
    }
    if let Some(metadata_path) = metadata_path {
        eprintln!("  {}  created {}", ansi.green("✓"), metadata_path);
    }
}

fn format_elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 1.0 {
        format!("{}ms", duration.as_millis())
    } else if seconds < 60.0 {
        format!("{seconds:.2}s")
    } else {
        let minutes = (seconds / 60.0).floor() as u64;
        let remaining_seconds = seconds - (minutes as f64 * 60.0);
        format!("{minutes}m {remaining_seconds:.1}s")
    }
}

pub(super) fn print_transfer_summary(
    action: &str,
    bytes: u64,
    elapsed: Duration,
    path_or_target: &str,
) {
    let ansi = Ansi::new();
    let elapsed_seconds = elapsed.as_secs_f64().max(0.001);
    let bytes_per_second = (bytes as f64 / elapsed_seconds).round() as u64;
    eprintln!(
        "  {}  {} {}  {}  {}",
        ansi.green("✓"),
        action,
        format_human_bytes(bytes),
        ansi.dim("via http"),
        ansi.dim(&format!(
            "in {} ({}/s)",
            format_elapsed(elapsed),
            format_human_bytes(bytes_per_second)
        )),
    );
    eprintln!("     {}", ansi.teal(path_or_target));
}
