//! Commands that manage models through a running runtime's HTTP API.

use std::path::PathBuf;

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Add-Model Options")]
pub(crate) struct AddModelCommandArgs {
    /// Path to a .aimod package to load. Repeat to add multiple models in one call.
    #[arg(short, long, required = true, value_name = "PATH")]
    pub(crate) model: Vec<PathBuf>,

    /// HTTP port of the running runtime's API server
    #[arg(long, default_value_t = 9095, value_name = "PORT")]
    pub(crate) http_port: u16,

    /// Hostname or IP of the running runtime's API server
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    pub(crate) http_host: String,

    /// Full base URL of the running runtime (overrides --http-host / --http-port)
    #[arg(long, value_name = "URL")]
    pub(crate) http_url: Option<String>,

    /// Bearer token if the runtime has API authentication enabled
    #[arg(long, value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// Parallelism topology for the new model(s) — same values as `kapsl run --topology`
    #[arg(long, default_value = "data-parallel", value_name = "TOPOLOGY")]
    pub(crate) topology: String,

    /// Tensor-parallel device count for the new model(s)
    #[arg(long, default_value_t = 1, value_name = "N")]
    pub(crate) tp_degree: usize,

    /// HTTP request timeout (ms) for the load call — large models may take longer to respond
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "List Options")]
pub(crate) struct ListCommandArgs {
    /// HTTP port of the running runtime's API server
    #[arg(long, default_value_t = 9095, value_name = "PORT")]
    pub(crate) http_port: u16,

    /// Hostname or IP of the running runtime's API server
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    pub(crate) http_host: String,

    /// Full base URL of the running runtime (overrides --http-host / --http-port)
    #[arg(long, value_name = "URL")]
    pub(crate) http_url: Option<String>,

    /// Bearer token if the runtime has API authentication enabled
    #[arg(long, value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// Print the complete API response as JSON instead of a table
    #[arg(long)]
    pub(crate) json: bool,

    /// HTTP request timeout (ms)
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Remove-Model Options")]
pub(crate) struct RemoveModelCommandArgs {
    /// ID of the model to remove (shown by `kapsl list`)
    #[arg(value_name = "MODEL_ID")]
    pub(crate) model_id: u32,

    /// HTTP port of the running runtime's API server
    #[arg(long, default_value_t = 9095, value_name = "PORT")]
    pub(crate) http_port: u16,

    /// Hostname or IP of the running runtime's API server
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    pub(crate) http_host: String,

    /// Full base URL of the running runtime (overrides --http-host / --http-port)
    #[arg(long, value_name = "URL")]
    pub(crate) http_url: Option<String>,

    /// Admin bearer token if the runtime has API authentication enabled
    #[arg(long, value_name = "TOKEN")]
    pub(crate) auth_token: Option<String>,

    /// HTTP request timeout (ms)
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}
