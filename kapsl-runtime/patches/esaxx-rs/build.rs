#[cfg(feature = "cpp")]
#[cfg(not(target_os = "macos"))]
fn main() {
    let mut build = cc::Build::new();
    build.cpp(true);

    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("msvc") {
        // Align with the Rust/ORT dynamic CRT linkage on Windows.
        build.static_crt(false);
    }
    if !target.contains("msvc") {
        build.flag("-std=c++11");
    }

    build.file("src/esaxx.cpp").include("src").compile("esaxx");
}

#[cfg(feature = "cpp")]
#[cfg(target_os = "macos")]
fn main() {
    cc::Build::new()
        .cpp(true)
        .flag("-std=c++11")
        .flag("-stdlib=libc++")
        .file("src/esaxx.cpp")
        .include("src")
        .compile("esaxx");
}

#[cfg(not(feature = "cpp"))]
fn main() {}
