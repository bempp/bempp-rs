//! Trait definitions

mod function;

pub use function::FunctionSpace;
#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
