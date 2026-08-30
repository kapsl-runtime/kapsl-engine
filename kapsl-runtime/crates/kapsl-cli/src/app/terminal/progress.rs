//! Loading spinner shared by synchronous and asynchronous operations.

use std::future::Future;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::Ansi;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

struct Spinner {
    ansi: Ansi,
    label: String,
    running: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Spinner {
    fn start(label: &str) -> Self {
        let ansi = Ansi::new();
        let running = Arc::new(AtomicBool::new(true));
        let spinner_running = Arc::clone(&running);
        let spinner_label = label.to_string();
        let colors_on = ansi.is_enabled();

        let handle = std::thread::spawn(move || {
            let mut index = 0usize;
            while spinner_running.load(Ordering::Relaxed) {
                let frame = SPINNER_FRAMES[index % SPINNER_FRAMES.len()];
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
                index = index.wrapping_add(1);
            }
        });

        Self {
            ansi,
            label: label.to_string(),
            running,
            handle: Some(handle),
        }
    }

    fn finish(mut self, ok: bool) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let mark = if ok {
            self.ansi.green("✓")
        } else {
            self.ansi.red("✗")
        };
        eprintln!("\r  {}  {}   ", mark, self.label);
    }
}

pub(crate) fn run_with_loading<T, E, F>(label: &str, action: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    let spinner = Spinner::start(label);
    let result = action();
    spinner.finish(result.is_ok());
    result
}

pub(crate) async fn run_with_loading_async<T, E, Fut>(label: &str, future: Fut) -> Result<T, E>
where
    Fut: Future<Output = Result<T, E>>,
{
    let spinner = Spinner::start(label);
    let result = future.await;
    spinner.finish(result.is_ok());
    result
}
