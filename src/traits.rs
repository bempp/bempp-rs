//! Trait definitions

mod assembly;
mod function;

pub use assembly::{
    CellAssembler, CellGeometry, KernelEvaluator, PotentialAssembly, PotentialIntegrand,
};
pub use function::FunctionSpace;
#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
