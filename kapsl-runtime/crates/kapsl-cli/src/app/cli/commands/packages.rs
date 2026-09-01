//! Package build, bundle, transfer, and login command options.

use std::path::PathBuf;

use clap::{ArgGroup, ValueEnum};

use crate::backend::ServingBackendPolicy;

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Build Options")]
pub(crate) struct BuildCommandArgs {
    /// Build context: a model directory (containing kapsl.yaml) or a bare model file (.onnx, .gguf).
    /// When omitted, the current directory is used as the context.
    #[arg(value_name = "CONTEXT")]
    pub(crate) context: Option<PathBuf>,

    /// Explicit path to the source model file — use when the file lives outside the context directory
    #[arg(long, value_name = "PATH")]
    pub(crate) model: Option<PathBuf>,

    /// Output path for the generated .aimod package (defaults to <project_name>.aimod in the context)
    #[arg(long, value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Override the project name embedded in the package (defaults to the directory or file name)
    #[arg(long)]
    pub(crate) project_name: Option<String>,

    /// Deprecated: legacy combined framework tag (e.g. onnx, gguf, llm). Prefer
    /// --format / --model-type / --task.
    #[arg(long)]
    pub(crate) framework: Option<String>,

    /// Model file format / loader: onnx, gguf, safetensors
    #[arg(long)]
    pub(crate) format: Option<String>,

    /// Model capability class: causal-lm, embedding, seq-classifier, seq2seq, opaque
    #[arg(long = "model-type")]
    pub(crate) model_type: Option<String>,

    /// Serving operation: generate, embed, classify, rerank, forward
    #[arg(long)]
    pub(crate) task: Option<String>,

    /// Deployment backend policy: auto, llama_cpp, or vllm.
    /// Stored as metadata.serving.backend; omitted for legacy behavior.
    #[arg(long, value_enum, value_name = "BACKEND")]
    pub(crate) serving_backend: Option<ServingBackendPolicy>,

    /// Override the version string embedded in the package
    #[arg(long)]
    pub(crate) version: Option<String>,

    /// Arbitrary JSON object merged into the package manifest metadata
    #[arg(long, value_name = "JSON")]
    pub(crate) metadata_json: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Bundle Options")]
pub(crate) struct BundleCommandArgs {
    /// One or more .aimod packages to include
    #[arg(required = true, value_name = "MODEL")]
    pub(crate) model: Vec<PathBuf>,

    /// Output .kapsl-bundle path
    #[arg(long, required = true, value_name = "PATH")]
    pub(crate) output: PathBuf,

    /// Target host, for example linux-x86_64-cuda or linux-x86_64-cpu
    #[arg(long, value_name = "TARGET")]
    pub(crate) target: Option<String>,

    /// Resolve signed backend archives from this local directory instead of downloading them
    #[arg(long, value_name = "DIR")]
    pub(crate) backend_artifacts_dir: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Push Options")]
pub(crate) struct PushCommandArgs {
    /// Destination in the registry. Format: <repo>/<model>:<label>  (e.g. acme/gpt2:prod)
    #[arg(value_name = "TARGET")]
    pub(crate) target: String,

    /// Path to the .aimod package to upload (defaults to the only .aimod in the current directory)
    #[arg(value_name = "KAPSL")]
    pub(crate) kapsl: Option<PathBuf>,

    /// Explicit package path (alternative to the positional argument)
    #[arg(long, alias = "kapsl-path", value_name = "PATH")]
    pub(crate) model: Option<PathBuf>,

    /// Remote registry URL — overrides KAPSL_REMOTE_URL for this call
    #[arg(long, value_name = "URL")]
    pub(crate) remote_url: Option<String>,

    /// Bearer token for the remote registry (env: KAPSL_REMOTE_TOKEN)
    #[arg(long, value_name = "TOKEN")]
    pub(crate) remote_token: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Pull Options")]
#[command(
    group(
        ArgGroup::new("pull_target")
            .required(true)
            .args(["target", "model"])
    )
)]
pub(crate) struct PullCommandArgs {
    /// Package to download. Format: <repo>/<model>:<label>  (e.g. acme/gpt2:prod)
    #[arg(value_name = "TARGET")]
    pub(crate) target: Option<String>,

    /// Package to download (alternative to the positional argument)
    #[arg(long, alias = "target-ref", value_name = "TARGET")]
    pub(crate) model: Option<String>,

    /// Directory where the downloaded .aimod file will be saved (defaults to current directory)
    #[arg(long, value_name = "DIR")]
    pub(crate) destination_dir: Option<PathBuf>,

    /// Remote registry URL — overrides KAPSL_REMOTE_URL for this call
    #[arg(long, value_name = "URL")]
    pub(crate) remote_url: Option<String>,

    /// Bearer token for the remote registry (env: KAPSL_REMOTE_TOKEN)
    #[arg(long, value_name = "TOKEN")]
    pub(crate) remote_token: Option<String>,
}

#[derive(clap::Args, Debug)]
#[command(next_help_heading = "Login Options")]
pub(crate) struct LoginCommandArgs {
    /// Backend base URL (defaults to KAPSL_REMOTE_URL or https://api.kapsl.net/v1)
    #[arg(long, value_name = "URL")]
    pub(crate) remote_url: Option<String>,

    /// OAuth provider to use
    #[arg(long, value_enum, default_value_t = OAuthProvider::GitHub)]
    pub(crate) provider: OAuthProvider,

    /// Local callback host for browser redirect
    #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
    pub(crate) callback_host: String,

    /// Local callback port (0 picks an ephemeral free port)
    #[arg(long, value_name = "PORT", default_value_t = 0)]
    pub(crate) callback_port: u16,

    /// Max time to wait for browser login callback
    #[arg(long, value_name = "SECONDS", default_value_t = 180)]
    pub(crate) timeout_seconds: u64,

    /// Print login URL instead of opening a browser automatically
    #[arg(long, default_value_t = false)]
    pub(crate) no_browser: bool,

    /// Use OAuth Device Code flow for headless/SSH environments (GitHub provider only)
    #[arg(
        long = "device-code",
        visible_alias = "headless",
        default_value_t = false
    )]
    pub(crate) device_code: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum OAuthProvider {
    #[value(name = "github", alias = "git-hub")]
    GitHub,
    #[value(name = "google")]
    Google,
}

impl OAuthProvider {
    pub(crate) fn route_segment(self) -> &'static str {
        match self {
            Self::GitHub => "github",
            Self::Google => "google",
        }
    }
}
