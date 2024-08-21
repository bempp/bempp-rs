//! Assembly
use crate::assembly::common::RlstArray;
use rlst::RlstScalar;

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

pub trait BoundaryIntegrand {
    //! Integrand
    /// Scalar type
    type T: RlstScalar;

    /// Evaluate integrand for a singular quadrature rule
    unsafe fn evaluate_nonsingular(
        &self,
        test_table: &RlstArray<Self::T, 4>,
        trial_table: &RlstArray<Self::T, 4>,
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 3>,
        test_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
        trial_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
    ) -> Self::T;

    /// Evaluate integrand for a non-singular quadrature rule
    unsafe fn evaluate_singular(
        &self,
        test_table: &RlstArray<Self::T, 4>,
        trial_table: &RlstArray<Self::T, 4>,
        point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 2>,
        test_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
        trial_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
    ) -> Self::T;
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

pub trait CellPairAssembler<T: RlstScalar> {
    //! Assembler for the contributions from a pair of cells
    /// Assemble contributions into `local_mat`
    fn assemble(&mut self, local_mat: &mut RlstArray<T, 2>);
    /// Set the test cell
    fn set_test_cell(&mut self, test_cell: usize);
    /// Set the trial cell
    fn set_trial_cell(&mut self, trial_cell: usize);
}
