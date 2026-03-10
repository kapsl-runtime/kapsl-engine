Generated from the kapsl monorepo.

This repo owns the full runtime product line:
- kapsl-runtime
- runtime extensions
- benchmarks and inference test harnesses

Current state:
- The runtime application and benchmark crates remain here.
- Shared runtime libraries are now consumed from `https://github.com/kapsl-runtime/kapsl-sdk.git`.
- The local `patches/` directory remains to provide the `esaxx-rs` crates.io override used by the runtime stack.
