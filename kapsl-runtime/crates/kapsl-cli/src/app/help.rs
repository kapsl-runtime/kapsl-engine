//! Presentation for `--help`: clap's colour styles and the epilogue shown
//! after the generated option list.

use super::*;

pub(crate) fn kapsl_help_styles() -> clap::builder::Styles {
    use clap::builder::styling::{AnsiColor, Effects};
    clap::builder::Styles::styled()
        .header(AnsiColor::Cyan.on_default() | Effects::BOLD)
        .usage(AnsiColor::Cyan.on_default() | Effects::BOLD)
        .literal(AnsiColor::BrightCyan.on_default() | Effects::BOLD)
        .placeholder(AnsiColor::Cyan.on_default())
        .error(AnsiColor::Red.on_default() | Effects::BOLD)
        .invalid(AnsiColor::Yellow.on_default() | Effects::BOLD)
        .valid(AnsiColor::Green.on_default())
}

/// Worked examples shown under `Examples:` in `--help`, as
/// `(explanation, commands)` groups rendered in order.
const HELP_EXAMPLES: &[(&str, &[&str])] = &[
    (
        "# Start the runtime with one model",
        &["kapsl run models/gpt2/gpt2.aimod"],
    ),
    (
        "# Prepare and run a no-network deployment bundle",
        &[
            "kapsl bundle model.aimod --output model.kapsl-bundle",
            "kapsl run model.kapsl-bundle",
        ],
    ),
    (
        "# Load an extra model into an already-running runtime (no restart)",
        &["kapsl add-model --model models/llama/llama.aimod"],
    ),
    (
        "# List models loaded in the running runtime",
        &["kapsl list"],
    ),
    (
        "# Remove a loaded model (the package file is kept)",
        &["kapsl remove-model 2"],
    ),
    (
        "# Package a model directory or single file",
        &[
            "kapsl build ./models/gpt-llm",
            "kapsl build ./model.onnx --output ./model.aimod",
        ],
    ),
    (
        "# Push / pull packages to/from a remote registry",
        &[
            "kapsl push acme/gpt2:prod ./model.aimod",
            "kapsl pull acme/gpt2:prod --destination-dir ./models",
        ],
    ),
    (
        "# Authenticate (opens browser; use --device-code for SSH/headless)",
        &["kapsl login", "kapsl login --device-code"],
    ),
    (
        "# Add optional Windows GPU acceleration",
        &[
            "kapsl provider install cuda12",
            "kapsl provider install tensorrt10",
        ],
    ),
];

/// Environment variables documented under `Environment variables:` in `--help`.
const HELP_ENV_VARS: &[(&str, &str)] = &[
    ("KAPSL_API_TOKEN_READER", "Read-only API token"),
    ("KAPSL_API_TOKEN_WRITER", "Writer API token"),
    ("KAPSL_API_TOKEN_ADMIN", "Admin API token"),
    ("KAPSL_REMOTE_URL", "Default remote registry URL"),
    ("KAPSL_REMOTE_TOKEN", "Bearer token for push/pull"),
    ("KAPSL_OFFLINE", "Disable lazy-backend network access"),
    ("KAPSL_BACKEND_CACHE_DIR", "Lazy backend cache root"),
    (
        "KAPSL_TCP_AUTH_TOKEN",
        "Required native-inference token for non-loopback TCP",
    ),
    (
        "KAPSL_SHM_SIZE_MB",
        "Shared-memory pool size (MiB) for shm/hybrid transport",
    ),
];

/// Column width the environment-variable names are padded to.
const HELP_ENV_NAME_WIDTH: usize = 28;

pub(crate) fn cli_after_help() -> String {
    use std::fmt::Write as _;
    let a = Ansi::new();
    let header = |s: &str| a.bold(&a.teal(s)).into_owned();
    let cmd = |s: &str| a.bold(s).into_owned();
    let comment = |s: &str| a.dim(s).into_owned();

    let mut out = String::new();

    let _ = writeln!(out, "{}", header("Examples:"));
    for (explanation, commands) in HELP_EXAMPLES {
        let _ = writeln!(out, "  {}", comment(explanation));
        for command in *commands {
            let _ = writeln!(out, "  {}", cmd(command));
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "{}", header("Environment variables:"));
    for (name, desc) in HELP_ENV_VARS {
        let padded = format!("{:<width$}", name, width = HELP_ENV_NAME_WIDTH);
        let _ = writeln!(out, "  {}{}", a.teal(&padded), a.dim(desc));
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "{}", header("Compatibility:"));
    let _ = writeln!(out, "  {}", cmd("kapsl --model models/gpt2/gpt2.aimod"));
    let _ = write!(
        out,
        "    {}",
        comment("(equivalent to `kapsl run --model models/gpt2/gpt2.aimod`)")
    );
    out
}
