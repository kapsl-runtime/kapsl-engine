use super::*;

mod archive;
mod auth;
mod build;
mod commands;
mod context;
mod remote;
mod target;
mod temp;
mod transfer;
mod types;

#[cfg(test)]
pub(crate) use auth::{auth_base_url_from_remote_url, store_remote_token_for_remote};
pub(crate) use auth::{
    is_likely_headless_session, perform_device_code_login_flow, resolved_login_remote_url,
    resolved_remote_token,
};
use auth::{maybe_auto_login_for_remote, perform_browser_login_flow, resolved_remote_url};
use build::infer_framework_from_model_path;
pub(crate) use build::{
    create_kapsl_package, find_model_file_in_context, looks_like_model_file_path,
    resolve_package_loader,
};
pub(crate) use commands::{
    execute_build_command, execute_login_command, execute_pull_command, execute_push_command,
};
pub(crate) use context::{
    create_kapsl_package_from_context, prompt_non_empty_with_default, prompt_provider_with_default,
    resolve_axis_triple, AxisOverrides, ContextPackageRequest,
};
use context::{
    create_source_metadata_if_missing, default_model_type_for_format, infer_format_from_model_path,
    legacy_framework_for, prompt_select_with_default, prompt_task_for_model_type, FORMAT_OPTIONS,
    MODEL_TYPE_OPTIONS,
};
#[cfg(test)]
pub(crate) use context::{prompt_model_file_with_default, prompt_with_default};
pub(crate) use remote::{pull_kapsl_from_remote, push_kapsl_to_remote};
pub(crate) use target::parse_model_target;
use target::ModelTargetRef;
#[cfg(test)]
pub(crate) use temp::TempDirGuard;
use temp::{replace_output_file, staged_output_path};
pub(crate) use transfer::fetch_remote_artifact_inventory;
use transfer::{
    artifact_url_for_remote, ArtifactTransport, HttpArtifactTransport, RemoteHttpRequestError,
};
pub(crate) use types::{
    PackageKapslRequest, PackageKapslResponse, PullKapslRequest, PullKapslResponse,
    PushKapslRequest, PushKapslResponse, RemoteArtifactInventoryResponse,
    RuntimeRemoteArtifactInventoryResponse,
};
