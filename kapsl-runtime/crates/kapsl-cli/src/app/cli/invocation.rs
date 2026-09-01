//! Parsing and normalization of runtime command invocations.

use clap::{ArgMatches, FromArgMatches};

use super::Args;
use crate::app::dyn_error_from_message;
use crate::DynError;

pub(crate) fn parse_runtime_args_and_matches(
    argv: &[String],
) -> Result<(Args, ArgMatches), DynError> {
    let cmd = <Args as clap::Args>::augment_args(clap::Command::new("kapsl"));
    let matches = cmd
        .try_get_matches_from(argv)
        .map_err(|error| dyn_error_from_message(error.to_string()))?;
    let args = Args::from_arg_matches(&matches)?;
    Ok((args, matches))
}

pub(crate) fn runtime_argv_from_invocation(raw_argv: &[String]) -> Vec<String> {
    if matches!(raw_argv.get(1).map(String::as_str), Some("run")) {
        let mut runtime_argv = Vec::with_capacity(raw_argv.len().saturating_sub(1));
        runtime_argv.push(raw_argv[0].clone());
        runtime_argv.extend(raw_argv.iter().skip(2).cloned());
        runtime_argv
    } else {
        raw_argv.to_vec()
    }
}
