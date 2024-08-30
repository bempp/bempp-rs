//! Traits for potential assembly
use super::CellGeometry;
use crate::assembly::common::RlstArray;
use crate::traits::FunctionSpace;
use rlst::RlstScalar;

pub unsafe trait PotentialIntegrand {
    //! Integrand
    //!
    //! # Safety
    //! This trait's methods use unsafe access

    /// Scalar type
    type T: RlstScalar;

    /// Evaluate integrand
    fn evaluate(
        &self,
        table: &RlstArray<Self::T, 4>,
        point_index: usize,
        eval_index: usize,
        basis_index: usize,
        k: &RlstArray<Self::T, 3>,
        geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
    ) -> Self::T;
}

pub trait CellAssembler {
    //! Assembler for the contributions from a cell
    /// Scalar type
    type T: RlstScalar;

    /// Assemble contributions into `local_mat`
    fn assemble(&mut self, local_mat: &mut RlstArray<Self::T, 2>);
    /// Set the cell
    fn set_cell(&mut self, cell: usize);
}

pub trait PotentialAssembly {
    //! Functions for boundary assembly
    /// Scalar type
    type T: RlstScalar;

    /// Assemble into a dense matrix
    fn assemble_into_dense<Space: FunctionSpace<T = Self::T> + Sync>(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        space: &Space,
        points: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
    );
}
