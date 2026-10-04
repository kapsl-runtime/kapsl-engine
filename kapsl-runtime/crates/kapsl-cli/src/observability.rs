//! Process-wide logging for runtime, transports, and dependency diagnostics.
//!
//! Libraries emit events; only the executable installs the subscriber. The
//! subscriber also bridges `log` records, so existing SDK logs use this sink.

use std::io;
use std::time::Duration;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub(crate) fn uses_json_format() -> bool {
    std::env::var("KAPSL_LOG_FORMAT").is_ok_and(|format| format == "json")
}

pub(crate) fn init() -> Result<(), crate::DynError> {
    let format = std::env::var("KAPSL_LOG_FORMAT").unwrap_or_else(|_| "text".to_string());
    let json = match format.as_str() {
        "text" => false,
        "json" => true,
        _ => return Err("KAPSL_LOG_FORMAT must be 'text' or 'json'".into()),
    };
    // Do not interpret field-value filter expressions as regular expressions.
    let filter = EnvFilter::builder()
        .with_regex(false)
        .with_default_directive(tracing::Level::WARN.into())
        .from_env()?;
    let registry = tracing_subscriber::registry().with(filter);
    let layer = fmt::layer().with_ansi(false).with_writer(io::stderr);
    if json {
        registry.with(layer.json()).try_init()?;
    } else {
        registry.with(layer).try_init()?;
    }
    Ok(())
}

/// Log completed ingress requests without headers, paths, query strings, or
/// payloads. Paths can contain user-controlled identifiers and secrets.
pub(crate) fn record_request(protocol: &'static str, method: &str, status: u16, elapsed: Duration) {
    tracing::info!(
        target: "kapsl::access",
        protocol,
        method,
        status,
        elapsed_us = elapsed.as_micros() as u64,
        "request completed"
    );
}

/// Archive raw managed-process output while making it available to the central
/// sink at `kapsl::backend_output=debug`. Each pipe is drained independently in
/// bounded chunks, including output without newlines.
pub(crate) fn spawn_logged_child(
    command: &mut std::process::Command,
    log_path: &std::path::Path,
    backend: &'static str,
) -> io::Result<std::process::Child> {
    use std::fs::OpenOptions;
    use std::process::Stdio;
    use std::sync::{Arc, Mutex};

    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let archive = Arc::new(Mutex::new(options.open(log_path)?));
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    for (stream, reader) in [
        ("stdout", Box::new(stdout) as Box<dyn io::Read + Send>),
        ("stderr", Box::new(stderr) as Box<dyn io::Read + Send>),
    ] {
        let archive = archive.clone();
        if let Err(error) = std::thread::Builder::new()
            .name(format!("kapsl-{backend}-{stream}"))
            .spawn(move || drain_child_output(reader, archive, backend, stream))
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    }
    Ok(child)
}

fn drain_child_output(
    mut reader: impl io::Read,
    archive: std::sync::Arc<std::sync::Mutex<std::fs::File>>,
    backend: &'static str,
    stream: &'static str,
) {
    use io::Write;
    let mut buffer = [0_u8; 8192];
    let mut archive_failed = false;
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                tracing::warn!(target: "kapsl::backend_output", backend, stream, %error, "failed to read backend output");
                break;
            }
        };
        if !archive_failed {
            if let Err(error) = archive
                .lock()
                .expect("backend log lock poisoned")
                .write_all(&buffer[..count])
            {
                archive_failed = true;
                tracing::warn!(target: "kapsl::backend_output", backend, stream, %error, "failed to archive backend output");
            }
        }
        // Raw backend diagnostics are opt-in: they can contain model inputs.
        tracing::debug!(
            target: "kapsl::backend_output",
            backend,
            stream,
            output = %String::from_utf8_lossy(&buffer[..count]),
            "backend output"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_output_archives_long_unterminated_and_binary_data() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let archive = std::sync::Arc::new(std::sync::Mutex::new(file.reopen().unwrap()));
        let mut input = vec![b'x'; 100_000];
        input.extend_from_slice(&[0, 255, b'\n']);
        drain_child_output(input.as_slice(), archive, "test", "stdout");
        assert_eq!(std::fs::read(file.path()).unwrap(), input);
    }

    #[cfg(unix)]
    #[test]
    fn managed_child_drains_both_pipes_beyond_pipe_capacity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("backend.log");
        let mut command = std::process::Command::new("sh");
        command.args([
            "-c",
            "head -c 131072 /dev/zero; head -c 131072 /dev/zero >&2",
        ]);
        let mut child = spawn_logged_child(&mut command, &path, "test").unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("backend output blocked the child");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // Pipe readers may finish archiving just after the child exits.
        while std::fs::metadata(&path).unwrap().len() < 262144 {
            assert!(
                std::time::Instant::now() < deadline,
                "backend output was lost"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(std::fs::read(path).unwrap(), vec![0_u8; 262144]);
    }
}
