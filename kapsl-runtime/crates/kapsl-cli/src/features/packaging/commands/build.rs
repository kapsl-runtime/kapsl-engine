use super::super::{
    create_kapsl_package, create_kapsl_package_from_context, looks_like_model_file_path,
    AxisOverrides, ContextPackageRequest, PackageKapslRequest, PackageKapslResponse,
};
use super::output::print_build_summary;
use crate::app::{cli_stdin_is_tty, dyn_error_from_message, run_with_loading, BuildCommandArgs};
use crate::DynError;
use std::path::{Path, PathBuf};

fn context_metadata_missing(context_path: &Path) -> bool {
    context_path.is_dir() && !context_path.join("metadata.json").exists()
}

fn execute_context_build(
    mut request: ContextPackageRequest<'_>,
) -> Result<PackageKapslResponse, DynError> {
    request.interactive_metadata_setup =
        cli_stdin_is_tty() && context_metadata_missing(request.context_path);
    let interactive_metadata_setup = request.interactive_metadata_setup;
    let build = || create_kapsl_package_from_context(request).map_err(dyn_error_from_message);

    if interactive_metadata_setup {
        build()
    } else {
        run_with_loading("Building package", build)
    }
}

fn context_package_request<'a>(
    context_path: &'a Path,
    args: &'a BuildCommandArgs,
    metadata: Option<&'a serde_json::Value>,
) -> ContextPackageRequest<'a> {
    ContextPackageRequest {
        model_override: args.model.as_deref(),
        output_override: args.output.as_deref(),
        project_name_override: args.project_name.as_deref(),
        framework_override: args.framework.as_deref(),
        version_override: args.version.as_deref(),
        metadata_override: metadata,
        serving_backend_override: args.serving_backend,
        axes: AxisOverrides {
            format: args.format.as_deref(),
            model_type: args.model_type.as_deref(),
            task: args.task.as_deref(),
        },
        ..ContextPackageRequest::new(context_path)
    }
}

fn file_package_request(
    model_path: &Path,
    args: &BuildCommandArgs,
    metadata: Option<&serde_json::Value>,
) -> PackageKapslRequest {
    PackageKapslRequest {
        model_path: model_path.to_string_lossy().to_string(),
        output_path: args
            .output
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        project_name: args.project_name.clone(),
        framework: args.framework.clone(),
        format: args.format.clone(),
        model_type: args.model_type.clone(),
        task: args.task.clone(),
        serving_backend: args.serving_backend,
        version: args.version.clone(),
        metadata: metadata.cloned(),
    }
}

fn model_file_metadata_missing(model_path: &Path) -> bool {
    let metadata_directory = match model_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    !metadata_directory.join("metadata.json").exists()
}

fn execute_model_file_build(
    request: &PackageKapslRequest,
) -> Result<PackageKapslResponse, DynError> {
    let model_path = PathBuf::from(request.model_path.trim());
    let interactive_metadata_setup = cli_stdin_is_tty() && model_file_metadata_missing(&model_path);
    let build = || {
        create_kapsl_package(request, interactive_metadata_setup).map_err(dyn_error_from_message)
    };

    if interactive_metadata_setup {
        build()
    } else {
        run_with_loading("Building package", build)
    }
}

pub(crate) fn execute_build_command(args: BuildCommandArgs) -> Result<(), DynError> {
    let metadata = match args.metadata_json.as_deref() {
        Some(raw) => Some(
            serde_json::from_str::<serde_json::Value>(raw).map_err(|error| {
                dyn_error_from_message(format!("Invalid --metadata-json: {error}"))
            })?,
        ),
        None => None,
    };

    let response = match args.context.as_ref() {
        Some(context_or_model_path) if context_or_model_path.is_dir() => execute_context_build(
            context_package_request(context_or_model_path, &args, metadata.as_ref()),
        )?,
        Some(context_or_model_path)
            if looks_like_model_file_path(context_or_model_path)
                || context_or_model_path.is_file() =>
        {
            if args.model.is_some() {
                return Err(dyn_error_from_message(
                    "When CONTEXT is a model file, do not also pass --model.",
                ));
            }
            let request = file_package_request(context_or_model_path, &args, metadata.as_ref());
            execute_model_file_build(&request)?
        }
        Some(context_path) => execute_context_build(context_package_request(
            context_path,
            &args,
            metadata.as_ref(),
        ))?,
        None => match args.model.as_ref() {
            Some(model_path) => {
                let request = file_package_request(model_path, &args, metadata.as_ref());
                execute_model_file_build(&request)?
            }
            None => {
                // Docker-style default: `kapsl build` means build the current directory.
                let context_directory = PathBuf::from(".");
                execute_context_build(context_package_request(
                    &context_directory,
                    &args,
                    metadata.as_ref(),
                ))?
            }
        },
    };
    print_build_summary(&response.kapsl_path, response.metadata_path.as_deref());
    Ok(())
}
