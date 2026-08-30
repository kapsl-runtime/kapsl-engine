fn main() {
    println!("cargo:rerun-if-env-changed=KAPSL_VERSION");
    println!("cargo:rerun-if-changed=native/linux/glibc_compat.c");
    println!("cargo:rerun-if-changed=native/windows/posix_memalign_compat.c");
    // rust-embed consumes files outside this crate directory. Register the
    // dashboard explicitly for ordinary incremental builds. Timestamp-
    // preserving deployment syncs should use scripts/build-with-embedded-ui.sh,
    // which invalidates the Rust embedding module before invoking Cargo.
    println!("cargo:rerun-if-changed=../../ui");

    // On Linux, provide glibc 2.38+ compat symbols so ort-sys prebuilts link
    // on older cluster glibc (< 2.38 lacks __isoc23_strtoll et al.).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        cc::Build::new()
            .file("native/linux/glibc_compat.c")
            .compile("compat_glibc");

        // ONNX Runtime loads its provider bridge with a bare-name dlopen(), so
        // the loader searches this binary's own DT_RUNPATH for it. ORT is
        // linked into this executable, which makes the executable the calling
        // object, and $ORIGIN then resolves to wherever the binary was
        // installed. Without it the sidecar libraries that ship beside the
        // binary are invisible unless the directory happens to be a system
        // library path, and every accelerator silently degrades to CPU.
        // Windows needs no equivalent: its search order already includes the
        // directory the executable was loaded from.
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN");
    }

    // On Windows, provide a posix_memalign shim for llama-cpp-sys-2
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        cc::Build::new()
            .file("native/windows/posix_memalign_compat.c")
            .compile("posix_memalign_compat");
    }
}
