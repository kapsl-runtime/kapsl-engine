fn main() {
    // On Linux, provide glibc 2.38+ compat symbols so ort-sys prebuilts link
    // on older cluster glibc (< 2.38 lacks __isoc23_strtoll et al.).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        cc::Build::new()
            .file("compat_glibc.c")
            .compile("compat_glibc");
    }
    
    // On Windows, provide a posix_memalign shim for llama-cpp-sys-2
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        cc::Build::new()
            .file("posix_memalign_compat.c")
            .compile("posix_memalign_compat");
    }
}
