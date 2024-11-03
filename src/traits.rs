//! Trait definitions

mod assembly;
mod function;

pub use assembly::{
    Access1D, Access2D, BoundaryIntegrand, CellAssembler, CellGeometry, GeometryAccess,
    KernelEvaluator, PotentialAssembly, PotentialIntegrand,
};
pub use function::FunctionSpace;
#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
