use super::*;

mod axes;
mod builder;
mod files;
mod manifest;
mod prompts;

pub(crate) use axes::{
    default_model_type_for_format, default_task_for_model_type, infer_format_from_model_path,
    legacy_framework_for, resolve_axis_triple, task_options_for_model_type, AxisOverrides,
    FORMAT_OPTIONS, MODEL_TYPE_OPTIONS,
};
pub(crate) use builder::{create_kapsl_package_from_context, ContextPackageRequest};
pub(crate) use files::{
    collect_context_files, collect_existing_file_references_from_metadata,
    normalize_output_path_for_context,
};
pub(crate) use manifest::{create_source_metadata_if_missing, parse_context_manifest};
#[cfg(test)]
pub(crate) use prompts::prompt_with_default;
pub(crate) use prompts::{
    prompt_model_file_with_default, prompt_non_empty_with_default, prompt_provider_with_default,
    prompt_select_with_default, prompt_task_for_model_type,
};
