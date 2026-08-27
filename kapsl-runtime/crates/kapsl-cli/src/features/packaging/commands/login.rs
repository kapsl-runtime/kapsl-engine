use super::super::{
    is_likely_headless_session, perform_browser_login_flow, perform_device_code_login_flow,
    resolved_login_remote_url,
};
use crate::app::{dyn_error_from_message, Ansi, LoginCommandArgs, OAuthProvider};
use crate::DynError;

pub(crate) fn execute_login_command(args: LoginCommandArgs) -> Result<(), DynError> {
    let remote_url = resolved_login_remote_url(args.remote_url.as_deref());
    let auto_headless = args.no_browser || is_likely_headless_session();
    let use_device_code =
        args.device_code || (auto_headless && args.provider == OAuthProvider::GitHub);

    let response = if use_device_code {
        perform_device_code_login_flow(
            &remote_url,
            args.provider,
            args.timeout_seconds,
            args.no_browser,
        )
    } else {
        perform_browser_login_flow(
            &remote_url,
            args.provider,
            args.callback_host.trim(),
            args.callback_port,
            args.timeout_seconds,
            args.no_browser,
        )
    }
    .map_err(dyn_error_from_message)?;

    let ansi = Ansi::new();
    eprintln!();
    eprintln!(
        "  {}  {}",
        ansi.green("✓"),
        ansi.bold("Authenticated successfully")
    );
    eprintln!("     {}  {}", ansi.dim("Provider"), response.provider);
    eprintln!(
        "     {}    {}",
        ansi.dim("Remote"),
        ansi.teal(&response.remote_url)
    );
    eprintln!(
        "     {}    {}",
        ansi.dim("Token"),
        ansi.dim(&response.token_store_path)
    );
    eprintln!();
    Ok(())
}
