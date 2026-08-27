use super::{is_valid_extension_id, ExtensionInstallRequest, ExtensionInstallResponse};
use crate::app::{
    dyn_error_from_message, Ansi, ExtensionCommandArgs, ExtensionInstallCommandArgs,
    ExtensionSubcommand,
};
use crate::features::auth::format_authorization_header;
use crate::features::packaging::{format_remote_http_error, native_tls_http_agent_with_timeout};
use crate::DynError;
use std::time::Duration;

/// Dispatches extension subcommands while leaving room for future operations.
pub(crate) fn execute_extension_command(args: ExtensionCommandArgs) -> Result<(), DynError> {
    match args.command {
        ExtensionSubcommand::Install(args) => request_extension_install(args),
    }
}

/// Requests installation from a running engine. Marketplace download and
/// registry mutation deliberately remain server-side.
fn request_extension_install(args: ExtensionInstallCommandArgs) -> Result<(), DynError> {
    let extension_id = args.extension_id.trim();
    if !is_valid_extension_id(extension_id) {
        return Err(dyn_error_from_message(format!(
            "Invalid extension ID `{}`. Use letters, numbers, dots, underscores, or hyphens.",
            args.extension_id
        )));
    }

    let base_url = args
        .http_url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(|url| url.trim_end_matches('/').to_string())
        .unwrap_or_else(|| format!("http://{}:{}", args.http_host, args.http_port));
    let install_url = format!("{base_url}/api/extensions/install");
    let timeout = Duration::from_millis(args.timeout_ms.max(1));
    let agent = native_tls_http_agent_with_timeout(timeout);
    let payload = ExtensionInstallRequest {
        path: None,
        extension_id: Some(extension_id.to_string()),
        marketplace_url: args
            .marketplace_url
            .as_deref()
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .map(str::to_owned),
    };
    let payload = serde_json::to_string(&payload)
        .map_err(|error| dyn_error_from_message(format!("Failed to serialize request: {error}")))?;

    let mut request = agent
        .post(&install_url)
        .header("Content-Type", "application/json");
    if let Some(header) = format_authorization_header(args.auth_token.as_deref()) {
        request = request.header("Authorization", &header);
    }

    let mut response = request.send(payload).map_err(|error| {
        dyn_error_from_message(format!(
            "Kapsl Engine could not install `{extension_id}` via {install_url}: {}",
            format_remote_http_error(error)
        ))
    })?;
    let body = response.body_mut().read_to_string().map_err(|error| {
        dyn_error_from_message(format!(
            "Failed to read installation response from {install_url}: {error}"
        ))
    })?;
    let installed: ExtensionInstallResponse = serde_json::from_str(&body).map_err(|error| {
        dyn_error_from_message(format!(
            "Kapsl Engine returned an invalid installation response from {install_url}: {error}"
        ))
    })?;
    if !installed.is_success() {
        return Err(dyn_error_from_message(format!(
            "Kapsl Engine returned an unexpected installation status from {install_url}"
        )));
    }

    let display_name = installed.display_name();
    let version = installed.version().trim();
    let a = Ansi::new();
    eprintln!();
    eprintln!(
        "  {}  {}{}",
        a.green("✓"),
        a.bold(display_name),
        if version.is_empty() {
            String::new()
        } else {
            format!(" {}", a.dim(&format!("v{version}")))
        }
    );
    eprintln!("     {}", a.dim("Installed in the running Kapsl Engine"));
    eprintln!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::app::{
        Cli, ExtensionCommandArgs, ExtensionInstallCommandArgs, ExtensionSubcommand, KapslCommand,
    };
    use clap::Parser;

    #[test]
    fn extension_install_command_parses_hub_extension_id() {
        let cli = Cli::try_parse_from([
            "kapsl",
            "extension",
            "install",
            "connector.s3",
            "--http-port",
            "9195",
        ])
        .expect("parse extension install command");

        assert!(matches!(
            cli.command,
            Some(KapslCommand::Extension(ExtensionCommandArgs {
                command: ExtensionSubcommand::Install(ExtensionInstallCommandArgs {
                    extension_id,
                    http_port: 9195,
                    ..
                })
            })) if extension_id == "connector.s3"
        ));
    }
}
