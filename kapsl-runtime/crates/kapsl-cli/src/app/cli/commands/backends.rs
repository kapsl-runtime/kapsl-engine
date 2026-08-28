//! Backend-pack and provider-pack command options.

use std::path::PathBuf;

use clap::{Subcommand, ValueEnum};

#[derive(clap::Args, Debug)]
pub(crate) struct BackendCommandArgs {
    #[command(subcommand)]
    pub(crate) command: BackendSubcommand,
}

#[derive(Subcommand, Debug)]
pub(crate) enum BackendSubcommand {
    /// Resolve, verify, and cache every backend required by a model
    Ensure(BackendEnsureCommandArgs),
    /// List installed backend packs
    List(BackendListCommandArgs),
    /// Remove interrupted staging data and, optionally, old runtime caches
    Prune(BackendPruneCommandArgs),
}

#[derive(clap::Args, Debug)]
pub(crate) struct BackendEnsureCommandArgs {
    /// Model package(s) whose backend should be prepared
    #[arg(required = true, value_name = "MODEL")]
    pub(crate) model: Vec<PathBuf>,

    /// Refuse network access and report any missing pack
    #[arg(long)]
    pub(crate) offline: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct BackendListCommandArgs {
    /// Emit a JSON array instead of a table
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct BackendPruneCommandArgs {
    /// Also remove caches belonging to Kapsl runtime versions other than this one
    #[arg(long)]
    pub(crate) old_versions: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct ProviderCommandArgs {
    #[command(subcommand)]
    pub(crate) command: ProviderSubcommand,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ProviderSubcommand {
    /// Download and install a provider pack matching this Kapsl release
    Install(ProviderInstallCommandArgs),
}

#[derive(clap::Args, Debug)]
pub(crate) struct ProviderInstallCommandArgs {
    /// Accelerator runtime to install
    #[arg(value_enum)]
    pub(crate) provider: ProviderPackage,

    /// Reinstall the pack even when a complete matching pack is present
    #[arg(long)]
    pub(crate) force: bool,

    /// Override the destination directory containing kapsl.exe
    #[arg(long, value_name = "DIR", hide = true)]
    pub(crate) install_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ProviderPackage {
    #[value(name = "cuda12", alias = "cuda")]
    Cuda12,
    #[value(name = "tensorrt10", alias = "tensorrt")]
    TensorRt10,
}
