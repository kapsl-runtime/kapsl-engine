//! Startup banner and ready-state presentation.

use super::Ansi;

pub(crate) fn print_startup_banner() {
    let ansi = Ansi::new();
    let version = env!("CARGO_PKG_VERSION");
    eprintln!();
    eprintln!(
        "  {}  {}",
        ansi.teal("▌ Kapsl Runtime"),
        ansi.dim(&format!("v{}", version))
    );
    eprintln!("  {}", ansi.dim("─────────────────────────────────────"));
}

pub(crate) fn print_startup_ready(
    elapsed_ms: u128,
    serving_endpoint: &str,
    http_ip: &str,
    http_port: u16,
) {
    let ansi = Ansi::new();
    let url_base = format!("http://{}:{}", http_ip, http_port);

    eprintln!();
    eprintln!(
        "  {} {}  {}",
        ansi.green("✓"),
        ansi.bold("Ready"),
        ansi.dim(&format!("(started in {}ms)", elapsed_ms))
    );
    eprintln!();

    let rows: &[(&str, String)] = &[
        ("Inference", serving_endpoint.to_string()),
        ("API", format!("{}/api", url_base)),
        ("Dashboard", url_base.clone()),
        ("Metrics", format!("{}/metrics", url_base)),
    ];

    let label_width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    for (label, url) in rows {
        eprintln!(
            "  {}  {:label_width$}  {}",
            ansi.teal("→"),
            ansi.dim(label),
            ansi.teal(url),
        );
    }
    eprintln!();
}
