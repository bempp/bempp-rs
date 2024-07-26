//! Function space

mod common;
//#[cfg(feature = "mpi")]
//mod parallel;
mod serial;

pub(crate) use common::assign_dofs;
//#[cfg(feature = "mpi")]
//pub use parallel::ParallelFunctionSpace;
pub use serial::SerialFunctionSpace;
