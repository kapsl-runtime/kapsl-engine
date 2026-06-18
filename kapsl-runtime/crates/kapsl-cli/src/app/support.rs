use super::*;

pub(crate) fn dyn_error_from_message(message: impl Into<String>) -> DynError {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

pub(crate) fn parse_runtime_args_and_matches(
    argv: &[String],
) -> Result<(Args, ArgMatches), DynError> {
    let cmd = <Args as clap::Args>::augment_args(clap::Command::new("kapsl"));
    let matches = cmd
        .try_get_matches_from(argv)
        .map_err(|e| dyn_error_from_message(e.to_string()))?;
    let args = Args::from_arg_matches(&matches)?;
    Ok((args, matches))
}

// ─── ANSI helpers (no external deps) ────────────────────────────────────────

pub(crate) fn cli_color_enabled() -> bool {
    // Respect NO_COLOR / TERM=dumb; fall back to stderr being a tty.
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("TERM").as_deref() == Ok("dumb") {
        return false;
    }
    // Use atty-free check: if stderr fd 2 is a character device we assume tty.
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stderr().as_raw_fd();
        libc_isatty(fd)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub(crate) fn cli_stdin_is_tty() -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        libc_isatty(std::io::stdin().as_raw_fd())
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(unix)]
pub(crate) fn libc_isatty(fd: i32) -> bool {
    extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    unsafe { isatty(fd) != 0 }
}

pub(crate) struct Ansi {
    enabled: bool,
}

impl Ansi {
    pub(crate) fn new() -> Self {
        Self {
            enabled: cli_color_enabled(),
        }
    }

    pub(crate) fn teal<'a>(&self, s: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[38;5;43m{}\x1b[0m", s).into()
        } else {
            s.into()
        }
    }

    pub(crate) fn green<'a>(&self, s: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[32m{}\x1b[0m", s).into()
        } else {
            s.into()
        }
    }

    pub(crate) fn red<'a>(&self, s: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[31m{}\x1b[0m", s).into()
        } else {
            s.into()
        }
    }

    pub(crate) fn dim<'a>(&self, s: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[2m{}\x1b[0m", s).into()
        } else {
            s.into()
        }
    }

    pub(crate) fn bold<'a>(&self, s: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[1m{}\x1b[0m", s).into()
        } else {
            s.into()
        }
    }
}

pub(crate) fn print_startup_banner() {
    let a = Ansi::new();
    let version = env!("CARGO_PKG_VERSION");
    eprintln!();
    eprintln!(
        "  {}  {}",
        a.teal("▌ Kapsl Runtime"),
        a.dim(&format!("v{}", version))
    );
    eprintln!("  {}", a.dim("─────────────────────────────────────"));
}

pub(crate) fn print_startup_ready(
    elapsed_ms: u128,
    serving_endpoint: &str,
    http_ip: &str,
    http_port: u16,
) {
    let a = Ansi::new();
    let url_base = format!("http://{}:{}", http_ip, http_port);

    eprintln!();
    eprintln!(
        "  {} {}  {}",
        a.green("✓"),
        a.bold("Ready"),
        a.dim(&format!("(started in {}ms)", elapsed_ms))
    );
    eprintln!();

    let rows: &[(&str, String)] = &[
        ("Inference", serving_endpoint.to_string()),
        ("API", format!("{}/api", url_base)),
        ("Dashboard", url_base.clone()),
        ("Metrics", format!("{}/metrics", url_base)),
    ];

    let label_w = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
    for (label, url) in rows {
        eprintln!(
            "  {}  {:label_w$}  {}",
            a.teal("→"),
            a.dim(label),
            a.teal(&url),
            label_w = label_w,
        );
    }
    eprintln!();
}

// ─── Spinner ─────────────────────────────────────────────────────────────────

pub(crate) const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub(crate) fn run_with_loading<T, E, F>(label: &str, action: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    let a = Ansi::new();
    let running = Arc::new(AtomicBool::new(true));
    let spinner_running = Arc::clone(&running);
    let spinner_label = label.to_string();
    let colors_on = a.enabled;

    let spinner_handle = std::thread::spawn(move || {
        let mut idx = 0usize;
        while spinner_running.load(Ordering::Relaxed) {
            let frame = SPINNER_FRAMES[idx % SPINNER_FRAMES.len()];
            if colors_on {
                eprint!(
                    "\r  \x1b[38;5;43m{}\x1b[0m  \x1b[2m{}\x1b[0m   ",
                    frame, spinner_label
                );
            } else {
                eprint!("\r  {}  {}   ", frame, spinner_label);
            }
            let _ = std::io::stderr().flush();
            std::thread::sleep(Duration::from_millis(80));
            idx = idx.wrapping_add(1);
        }
    });

    let result = action();

    running.store(false, Ordering::Relaxed);
    let _ = spinner_handle.join();

    if result.is_ok() {
        eprintln!("\r  {}  {}   ", a.green("✓"), label);
    } else {
        eprintln!("\r  {}  {}   ", a.red("✗"), label);
    }

    result
}

pub(crate) async fn run_with_loading_async<T, E, Fut>(label: &str, future: Fut) -> Result<T, E>
where
    Fut: Future<Output = Result<T, E>>,
{
    let a = Ansi::new();
    let running = Arc::new(AtomicBool::new(true));
    let spinner_running = Arc::clone(&running);
    let spinner_label = label.to_string();
    let colors_on = a.enabled;

    let spinner_handle = std::thread::spawn(move || {
        let mut idx = 0usize;
        while spinner_running.load(Ordering::Relaxed) {
            let frame = SPINNER_FRAMES[idx % SPINNER_FRAMES.len()];
            if colors_on {
                eprint!(
                    "\r  \x1b[38;5;43m{}\x1b[0m  \x1b[2m{}\x1b[0m   ",
                    frame, spinner_label
                );
            } else {
                eprint!("\r  {}  {}   ", frame, spinner_label);
            }
            let _ = std::io::stderr().flush();
            std::thread::sleep(Duration::from_millis(80));
            idx = idx.wrapping_add(1);
        }
    });

    let result = future.await;

    running.store(false, Ordering::Relaxed);
    let _ = spinner_handle.join();

    if result.is_ok() {
        eprintln!("\r  {}  {}   ", a.green("✓"), label);
    } else {
        eprintln!("\r  {}  {}   ", a.red("✗"), label);
    }

    result
}

pub(crate) fn runtime_argv_from_invocation(raw_argv: &[String]) -> Vec<String> {
    if matches!(raw_argv.get(1).map(|s| s.as_str()), Some("run")) {
        let mut runtime_argv = Vec::with_capacity(raw_argv.len().saturating_sub(1));
        runtime_argv.push(raw_argv[0].clone());
        runtime_argv.extend(raw_argv.iter().skip(2).cloned());
        runtime_argv
    } else {
        raw_argv.to_vec()
    }
}
