fn main() {
    println!("cargo:rustc-check-cfg=cfg(kapsl_llama_external_pool_sdk)");
    println!("cargo:rerun-if-env-changed=KAPSL_LLAMA_EXTERNAL_POOL_SDK");
    let enabled = std::env::var("KAPSL_LLAMA_EXTERNAL_POOL_SDK")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
    let contract_enabled = std::env::var_os("CARGO_FEATURE_EXTERNAL_POOL_CONTRACT").is_some();
    if enabled && !contract_enabled {
        panic!("KAPSL_LLAMA_EXTERNAL_POOL_SDK requires the external-pool-contract Cargo feature");
    }
    if contract_enabled {
        println!("cargo:rustc-cfg=kapsl_llama_external_pool_sdk");
    }
}
