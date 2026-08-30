//! Extension-management command options.

use clap::Subcommand;

#[derive(clap::Args, Debug)]
pub(crate) struct ExtensionCommandArgs {
    #[command(subcommand)]
    pub(crate) command: ExtensionSubcommand,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ExtensionSubcommand {
    /// Download and install an extension from Kapsl Hub
    Install(ExtensionInstallCommandArgs),
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Extension Install Options")]
pub(crate) struct ExtensionInstallCommandArgs {
    /// Marketplace extension ID (for example connector.s3)
    #[arg(value_name = "EXTENSION_ID")]
    pub(crate) extension_id: String,

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

    /// Override the Kapsl Hub marketplace endpoint
    #[arg(long, value_name = "URL")]
    pub(crate) marketplace_url: Option<String>,

    /// HTTP request timeout (ms)
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    pub(crate) timeout_ms: u64,
}
