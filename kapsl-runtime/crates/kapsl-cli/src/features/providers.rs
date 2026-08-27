//! Provider-pack command and installation support.
//!
//! The command layer resolves CLI configuration, while the installer owns the
//! verified download and rollback-capable activation transaction.

mod activation;
mod command;
mod installer;
mod pack;
mod transfer;

pub(crate) use command::execute_provider_command;
