//! Assembly
mod boundary;
mod potential;
use crate::assembly::common::RlstArray;
use rlst::RlstScalar;

#[cfg(feature = "mpi")]
pub use boundary::ParallelBoundaryAssembly;
pub use boundary::{
    Access1D, Access2D, BoundaryAssembly, BoundaryIntegrand, CellPairAssembler, GeometryAccess,
};
pub use potential::{CellAssembler, PotentialAssembly, PotentialIntegrand};

pub trait CellGeometry {
    //! Cell geometry
    /// Scalar type
    type T: RlstScalar<Real = Self::T>;
    /// Points
    fn points(&self) -> &RlstArray<Self::T, 2>;
    /// Normals
    fn normals(&self) -> &RlstArray<Self::T, 2>;
    /// Jacobians
    fn jacobians(&self) -> &RlstArray<Self::T, 2>;
    /// Determinants of jacobians
    fn jdets(&self) -> &[Self::T];
}

pub trait KernelEvaluator {
    //! Kernel evaluator
    /// Scalar type
    type T: RlstScalar;

    /// Evaluate the kernel values for all source and target pairs
    ///
    /// For each source, the kernel is evaluated for exactly one target. This is equivalent to taking the diagonal of the matrix assembled by `assemble_st`
    fn assemble_pairwise_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Evaluate the kernel values for all sources and all targets
    ///
    /// For every source, the kernel is evaluated for every target.
    fn assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );
}
