//! Startup banner and ready-state presentation.

use super::Ansi;

pub(crate) fn print_startup_banner() {
    if crate::observability::uses_json_format() {
        tracing::info!(
            version = env!("CARGO_PKG_VERSION"),
            "Kapsl runtime starting"
        );
        return;
    }
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
    if crate::observability::uses_json_format() {
        tracing::info!(
            elapsed_ms = elapsed_ms as u64,
            serving_endpoint,
            http_ip,
            http_port,
            "Kapsl runtime ready"
        );
        return;
    }
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
        eprintln!("{}", endpoint_line(&ansi, label, url, label_width));
    }
    eprintln!();
}

/// One `→  Label  url` row. The label is padded before it is styled: padding
/// a styled string counts its escape codes as width, which misaligned the
/// column whenever color was on.
fn endpoint_line(ansi: &Ansi, label: &str, url: &str, label_width: usize) -> String {
    format!(
        "  {}  {}  {}",
        ansi.teal("→"),
        ansi.dim(&format!("{:label_width$}", label)),
        ansi.teal(url),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                for c in chars.by_ref() {
                    if c == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn endpoint_urls_line_up_with_color_on() {
        let ansi = Ansi::with_color(true);
        let api = strip_ansi(&endpoint_line(&ansi, "API", "http://127.0.0.1:9095/api", 9));
        let inference = strip_ansi(&endpoint_line(&ansi, "Inference", "/tmp/kapsl.sock", 9));
        assert_eq!(api.find("http"), inference.find("/tmp"));
        assert_eq!(api, "  →  API        http://127.0.0.1:9095/api");
    }
}
