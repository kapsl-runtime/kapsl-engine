//! Command-line schema for the Kapsl executable.

use clap::{Parser, Subcommand};

use crate::backend::BackendPlanCommandArgs;

mod commands;
mod help;
mod invocation;
mod run;

pub(crate) use commands::*;
pub(crate) use help::*;
pub(crate) use invocation::*;
pub(crate) use run::*;

#[derive(Parser, Debug)]
#[command(
    name = "kapsl",
    author,
    version,
    about = "Run, package, and distribute AI models",
    long_about = "Kapsl is a high-performance AI inference runtime and packaging tool.\n\
                  \n\
                  Use `kapsl run` to serve one or more model packages, `kapsl build` to\n\
                  create a portable .aimod package from an ONNX or GGUF model, and\n\
                  `kapsl push`/`pull` to sync packages with a remote registry.",
    after_help = cli_after_help(),
    styles(kapsl_help_styles()),
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<KapslCommand>,

    #[command(flatten)]
    pub(crate) run: Args,
}

#[derive(Subcommand, Debug)]
pub(crate) enum KapslCommand {
    /// Start the inference server and load one or more model packages
    Run(Args),
    /// Package a model file or directory into a portable .aimod archive
    Build(BuildCommandArgs),
    /// Create a self-contained model and backend bundle for offline deployment
    Bundle(BundleCommandArgs),
    /// Resolve a package's deployment backend policy for this host
    BackendPlan(BackendPlanCommandArgs),
    /// Inspect and manage lazily installed inference backends
    Backend(BackendCommandArgs),
    /// Upload a .aimod package to a remote registry
    Push(PushCommandArgs),
    /// Download a .aimod package from a remote registry
    Pull(PullCommandArgs),
    /// Log in to a remote registry and save credentials locally
    Login(LoginCommandArgs),
    /// Manage extensions in a running Kapsl Engine
    Extension(ExtensionCommandArgs),
    /// Install optional hardware acceleration support
    Provider(ProviderCommandArgs),
    /// Hot-load a model into an already-running runtime (no restart required)
    AddModel(AddModelCommandArgs),
    /// List models loaded in a running Kapsl Engine
    List(ListCommandArgs),
    /// Stop, unload, and unregister a model from a running Kapsl Engine
    RemoveModel(RemoveModelCommandArgs),
}

#[cfg(test)]
mod tests;
