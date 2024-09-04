//! Trait definitions

mod assembly;
mod function;

#[cfg(feature = "mpi")]
pub use assembly::ParallelBoundaryAssembly;
pub use assembly::{
    Access1D, Access2D, BoundaryAssembly, BoundaryIntegrand, CellAssembler, CellGeometry,
    CellPairAssembler, GeometryAccess, KernelEvaluator, PotentialAssembly, PotentialIntegrand,
};
pub use function::FunctionSpace;
#[cfg(feature = "mpi")]
pub use function::ParallelFunctionSpace;
