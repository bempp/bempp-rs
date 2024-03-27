//! Function space

#[cfg(feature = "mpi")]
mod parallel;
mod serial;

#[cfg(feature = "mpi")]
pub use parallel::ParallelFunctionSpace;
pub use serial::SerialFunctionSpace;
