//! Assembly
use crate::assembly::common::RlstArray;
use crate::traits::FunctionSpace;
use ndelement::{traits::FiniteElement, types::ReferenceCellType};
use ndgrid::traits::Grid;
use rlst::{CsrMatrix, RlstScalar};
use std::collections::HashMap;

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

pub trait CellPairAssembler {
    //! Assembler for the contributions from a pair of cells
    /// Scalar type
    type T: RlstScalar;

    /// Assemble contributions into `local_mat`
    fn assemble(&mut self, local_mat: &mut RlstArray<Self::T, 2>);
    /// Set the test cell
    fn set_test_cell(&mut self, test_cell: usize);
    /// Set the trial cell
    fn set_trial_cell(&mut self, trial_cell: usize);
}

pub trait BoundaryAssembly {
    //! Functions for boundary assembly
    /// Scalar type
    type T: RlstScalar;

    /// Assemble the singular contributions into a dense matrix
    fn assemble_singular_into_dense<Space: FunctionSpace<T = Self::T> + Sync>(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &Space,
        test_space: &Space,
    );

    /// Assemble the singular contributions into a CSR sparse matrix
    fn assemble_singular_into_csr<Space: FunctionSpace<T = Self::T> + Sync>(
        &self,
        trial_space: &Space,
        test_space: &Space,
    ) -> CsrMatrix<Self::T>;

    /// Assemble the singular correction into a dense matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_dense<Space: FunctionSpace<T = Self::T> + Sync>(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &Space,
        test_space: &Space,
    );

    /// Assemble the singular correction into a CSR matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_csr<Space: FunctionSpace<T = Self::T> + Sync>(
        &self,
        trial_space: &Space,
        test_space: &Space,
    ) -> CsrMatrix<Self::T>;

    /// Assemble into a dense matrix
    fn assemble_into_dense<Space: FunctionSpace<T = Self::T> + Sync>(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &Space,
        test_space: &Space,
    );

    /// Assemble the non-singular contributions into a dense matrix
    fn assemble_nonsingular_into_dense<Space: FunctionSpace<T = Self::T> + Sync>(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &Space,
        test_space: &Space,
        trial_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
        test_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
    );
}

#[cfg(feature = "mpi")]
pub trait ParallelBoundaryAssembly: BoundaryAssembly {
    //! Functions for parallel boundary assembly

    /// Assemble the singular contributions into a CSR sparse matrix, indexed by global DOF numbers
    fn parallel_assemble_singular_into_csr<
        'a,
        C: Communicator,
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
        SerialTestSpace: FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync + 'a,
        SerialTrialSpace: FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync + 'a,
    >(
        &self,
        trial_space: &'a (impl ParallelFunctionSpace<C, LocalSpace<'a> = SerialTrialSpace> + 'a),
        test_space: &'a (impl ParallelFunctionSpace<C, LocalSpace<'a> = SerialTestSpace> + 'a),
    ) -> CsrMatrix<Self::T>;

    /// Assemble the singular contributions into a CSR sparse matrix, indexed by global DOF numbers
    fn parallel_assemble_singular_correction_into_csr<
        'a,
        C: Communicator,
        TestGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = <Self::T as RlstScalar>::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
        SerialTestSpace: FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync + 'a,
        SerialTrialSpace: FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync + 'a,
    >(
        &self,
        trial_space: &'a (impl ParallelFunctionSpace<C, LocalSpace<'a> = SerialTrialSpace> + 'a),
        test_space: &'a (impl ParallelFunctionSpace<C, LocalSpace<'a> = SerialTestSpace> + 'a),
    ) -> CsrMatrix<Self::T>;
}
