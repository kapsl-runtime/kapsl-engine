use super::super::{
    parse_model_target, pull_kapsl_from_remote, push_kapsl_to_remote, PullKapslRequest,
    PushKapslRequest,
};
use super::output::print_transfer_summary;
use crate::app::{dyn_error_from_message, run_with_loading, PullCommandArgs, PushCommandArgs};
use crate::DynError;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn discover_package_in_current_directory() -> Result<PathBuf, String> {
    let current_directory = std::env::current_dir()
        .map_err(|error| format!("Failed to read current directory: {error}"))?;
    let entries = fs::read_dir(&current_directory).map_err(|error| {
        format!(
            "Failed to read current directory {}: {}",
            current_directory.display(),
            error
        )
    })?;

    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("Failed to read directory entry: {error}"))?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("aimod"))
        {
            candidates.push(path);
        }
    }

    candidates.sort();
    if candidates.is_empty() {
        return Err("No .aimod files found in the current directory. Pass an explicit package path via `kapsl push <repo>/<model>:<label> <PATH>` or `--model <PATH>`.".to_string());
    }
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }

    if let Some(directory_name) = current_directory
        .file_name()
        .and_then(|value| value.to_str())
    {
        let expected = current_directory.join(format!("{directory_name}.aimod"));
        if expected.is_file() {
            return Ok(expected);
        }
    }

    Err(format!(
        "Multiple .aimod files found in the current directory. Pass an explicit path.\nFound: {}",
        candidates
            .iter()
            .map(|path| path.file_name().unwrap_or_default().to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

pub(crate) fn execute_push_command(args: PushCommandArgs) -> Result<(), DynError> {
    if args.kapsl.is_some() && args.model.is_some() {
        return Err(dyn_error_from_message(
            "Push expects a single `.aimod` argument. Use either `kapsl push <repo>/<model>:<label> <KAPSL>` or `kapsl push <repo>/<model>:<label> --model <PATH>`.",
        ));
    }
    let target = parse_model_target(&args.target).map_err(dyn_error_from_message)?;
    let package_path = match args.kapsl.as_ref().or(args.model.as_ref()) {
        Some(path) => path.clone(),
        None => discover_package_in_current_directory().map_err(dyn_error_from_message)?,
    };
    let request = PushKapslRequest {
        kapsl_path: package_path.to_string_lossy().to_string(),
        target: target.as_string(),
        remote_url: args.remote_url,
        remote_token: args.remote_token,
        interactive_login: true,
    };

    let started_at = Instant::now();
    let response = run_with_loading("Uploading package", || {
        push_kapsl_to_remote(&request).map_err(dyn_error_from_message)
    })?;
    print_transfer_summary(
        "Uploaded",
        response.bytes_uploaded,
        started_at.elapsed(),
        &response.artifact_url,
    );
    Ok(())
}

pub(crate) fn execute_pull_command(args: PullCommandArgs) -> Result<(), DynError> {
    if args.target.is_some() && args.model.is_some() {
        return Err(dyn_error_from_message(
            "Pull expects a single target argument. Use either `kapsl pull <repo>/<model>:<label>` or `kapsl pull --model <repo>/<model>:<label>`.",
        ));
    }
    let target = args.target.or(args.model).ok_or_else(|| {
        dyn_error_from_message("Target is required. Usage: `kapsl pull <repo>/<model>:<label>`.")
    })?;
    let target = parse_model_target(&target).map_err(dyn_error_from_message)?;
    let request = PullKapslRequest {
        target: target.as_string(),
        destination_dir: args
            .destination_dir
            .map(|path| path.to_string_lossy().to_string()),
        remote_url: args.remote_url,
        remote_token: args.remote_token,
        interactive_login: true,
    };

    let started_at = Instant::now();
    let response = run_with_loading("Downloading package", || {
        pull_kapsl_from_remote(&request).map_err(dyn_error_from_message)
    })?;
    print_transfer_summary(
        "Downloaded",
        response.bytes_downloaded,
        started_at.elapsed(),
        &response.kapsl_path,
    );
    Ok(())
}
