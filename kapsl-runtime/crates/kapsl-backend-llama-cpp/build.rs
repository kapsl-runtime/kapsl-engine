fn main() {
    println!("cargo:rustc-check-cfg=cfg(kapsl_llama_external_pool_sdk)");
    let contract_enabled = std::env::var_os("CARGO_FEATURE_EXTERNAL_POOL_CONTRACT").is_some();
    if contract_enabled {
        println!("cargo:rustc-cfg=kapsl_llama_external_pool_sdk");
    }
}
