//! Trait definitions

mod assembly;
mod function;

pub use assembly::{CellAssembler, PotentialAssembly, PotentialIntegrand};
pub use function::FunctionSpace;
#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
