//! Traits for boundary assembly
use super::CellGeometry;
use crate::assembly::common::RlstArray;
use crate::traits::FunctionSpace;
#[cfg(feature = "mpi")]
use crate::traits::ParallelFunctionSpace;
#[cfg(feature = "mpi")]
use mpi::traits::Communicator;
use ndelement::types::ReferenceCellType;
use rlst::{CsrMatrix, RlstScalar};
use std::collections::HashMap;

pub trait BoundaryIntegrand {
    //! Integrand
    /// Scalar type
    type T: RlstScalar;

    #[allow(clippy::too_many_arguments)]
    /// Evaluate integrand for a singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` to be used
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

    #[allow(clippy::too_many_arguments)]
    /// Evaluate integrand for a non-singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` to be used
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
        C: Communicator,
        Space: ParallelFunctionSpace<C, T = Self::T>,
    >(
        &self,
        trial_space: &Space,
        test_space: &Space,
    ) -> CsrMatrix<Self::T>;

    /// Assemble the singular contributions into a CSR sparse matrix, indexed by global DOF numbers
    fn parallel_assemble_singular_correction_into_csr<
        C: Communicator,
        Space: ParallelFunctionSpace<C, T = Self::T>,
    >(
        &self,
        trial_space: &Space,
        test_space: &Space,
    ) -> CsrMatrix<Self::T>;
}
