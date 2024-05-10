//! Function space

mod common;
mod serial;

pub(crate) use common::assign_dofs;
pub use serial::SerialFunctionSpace;
