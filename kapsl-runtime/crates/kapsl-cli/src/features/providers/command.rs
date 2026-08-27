use super::installer::ProviderInstaller;
use super::pack::{configured_download_base_url, release_version, ProviderPack};
use super::transfer::NativeProviderPackTransfer;
use crate::app::{
    ProviderCommandArgs, ProviderInstallCommandArgs, ProviderPackage, ProviderSubcommand,
};
use crate::DynError;

/// Dispatches provider subcommands while leaving room for future operations.
pub(crate) fn execute_provider_command(args: ProviderCommandArgs) -> Result<(), DynError> {
    match args.command {
        ProviderSubcommand::Install(args) => execute_provider_install_command(args),
    }
}

fn execute_provider_install_command(args: ProviderInstallCommandArgs) -> Result<(), DynError> {
    if !cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        return Err(
            "`kapsl provider install` currently supports Windows x86_64. On Linux, install the matching provider .deb or tar.gz package from the release."
                .into(),
        );
    }

    let ProviderInstallCommandArgs {
        provider,
        force,
        install_dir,
    } = args;
    let install_dir = match install_dir {
        Some(path) => path,
        None => std::env::current_exe()?
            .parent()
            .ok_or("Could not determine the directory containing kapsl.exe")?
            .to_path_buf(),
    };
    if !install_dir.is_dir() {
        return Err(format!(
            "The Kapsl installation directory does not exist: {}",
            install_dir.display()
        )
        .into());
    }

    let requested: &[ProviderPack] = match provider {
        ProviderPackage::Cuda12 => &[ProviderPack::Cuda12],
        ProviderPackage::TensorRt10 => &[ProviderPack::Cuda12, ProviderPack::TensorRt10],
    };
    let version = release_version();
    let base_url = configured_download_base_url();
    let transfer = NativeProviderPackTransfer;
    let installer = ProviderInstaller::new(&version, &base_url, &install_dir, force, &transfer);

    for &pack in requested {
        installer.install(pack)?;
    }

    println!();
    println!("Provider installation complete.");
    match provider {
        ProviderPackage::Cuda12 => {
            println!("CUDA 12 is now available to Kapsl's automatic provider selection.");
        }
        ProviderPackage::TensorRt10 => {
            println!(
                "TensorRT 10 is now available to packages that declare `preferred_provider: tensorrt`."
            );
            println!(
                "Set KAPSL_PROVIDER_POLICY=manifest to require the package-declared provider."
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{Cli, KapslCommand};
    use clap::Parser;

    #[test]
    fn provider_install_command_uses_friendly_names() {
        let cli = Cli::try_parse_from(["kapsl", "provider", "install", "cuda12"])
            .expect("parse CUDA provider install");
        assert!(matches!(
            cli.command,
            Some(KapslCommand::Provider(ProviderCommandArgs {
                command: ProviderSubcommand::Install(ProviderInstallCommandArgs {
                    provider: ProviderPackage::Cuda12,
                    ..
                })
            }))
        ));

        let cli = Cli::try_parse_from(["kapsl", "provider", "install", "tensorrt"])
            .expect("parse TensorRT provider alias");
        assert!(matches!(
            cli.command,
            Some(KapslCommand::Provider(ProviderCommandArgs {
                command: ProviderSubcommand::Install(ProviderInstallCommandArgs {
                    provider: ProviderPackage::TensorRt10,
                    ..
                })
            }))
        ));
    }
}
