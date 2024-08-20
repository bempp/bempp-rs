//! Trait definitions

mod assembly;
mod function;

pub use assembly::{BoundaryIntegrand, CellGeometry, KernelEvaluator};
pub use function::FunctionSpace;
#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
