//! ANSI styling and terminal-capability detection.

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

pub(crate) struct Ansi {
    enabled: bool,
}

impl Ansi {
    pub(crate) fn new() -> Self {
        Self {
            enabled: cli_color_enabled(),
        }
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn teal<'a>(&self, value: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[38;5;43m{}\x1b[0m", value).into()
        } else {
            value.into()
        }
    }

    pub(crate) fn green<'a>(&self, value: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[32m{}\x1b[0m", value).into()
        } else {
            value.into()
        }
    }

    pub(crate) fn red<'a>(&self, value: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[31m{}\x1b[0m", value).into()
        } else {
            value.into()
        }
    }

    pub(crate) fn dim<'a>(&self, value: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[2m{}\x1b[0m", value).into()
        } else {
            value.into()
        }
    }

    pub(crate) fn bold<'a>(&self, value: &'a str) -> std::borrow::Cow<'a, str> {
        if self.enabled {
            format!("\x1b[1m{}\x1b[0m", value).into()
        } else {
            value.into()
        }
    }
}

fn cli_color_enabled() -> bool {
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("TERM").as_deref() == Ok("dumb") {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        libc_isatty(std::io::stderr().as_raw_fd())
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(unix)]
fn libc_isatty(fd: i32) -> bool {
    unsafe extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    unsafe { isatty(fd) != 0 }
}
