//! Function space

mod common;
mod parallel;
mod serial;

use common::assign_dofs;
pub use parallel::ParallelFunctionSpace;
pub use serial::SerialFunctionSpace;
