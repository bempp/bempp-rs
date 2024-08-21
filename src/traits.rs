//! Trait definitions

mod assembly;
mod function;

pub use assembly::{BoundaryIntegrand, CellGeometry, CellPairAssembler, KernelEvaluator};
pub use function::FunctionSpace;
#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
