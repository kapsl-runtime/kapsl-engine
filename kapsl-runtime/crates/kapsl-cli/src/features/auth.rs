//! Authentication domain facade.
//!
//! The implementation is split by responsibility while this module keeps the
//! existing crate-level API stable for runtime, HTTP, packaging, and tests.

use super::*;

mod contracts;
mod credentials;
mod model;
mod policy;
mod session;
mod state;
mod store;

pub(crate) use contracts::*;
pub(crate) use credentials::*;
pub(crate) use model::*;
pub(crate) use policy::*;
pub(crate) use session::*;
pub(crate) use state::*;
pub(crate) use store::*;
