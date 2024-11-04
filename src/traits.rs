//! Trait definitions

mod function;

#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
