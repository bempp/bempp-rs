//! Single layer integrand
use crate::assembly::common::RlstArray;
use crate::traits::{BoundaryIntegrand, CellGeometry};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

pub struct SingleLayerBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> SingleLayerBoundaryIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

unsafe impl<T: RlstScalar> BoundaryIntegrand for SingleLayerBoundaryIntegrand<T> {
    type T = T;

    fn evaluate_nonsingular(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<T, 3>,
        _test_geometry: &impl CellGeometry<T = T::Real>,
        _trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        unsafe {
            *k.get_unchecked([0, test_point_index, trial_point_index])
                * *test_table.get_unchecked([0, test_point_index, test_basis_index, 0])
                * *trial_table.get_unchecked([0, trial_point_index, trial_basis_index, 0])
        }
    }

    fn evaluate_singular(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 2>,
        _test_geometry: &impl CellGeometry<T = T::Real>,
        _trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        unsafe {
            *k.get_unchecked([0, point_index])
                * *test_table.get_unchecked([0, point_index, test_basis_index, 0])
                * *trial_table.get_unchecked([0, point_index, trial_basis_index, 0])
        }
    }
}

impl<T: RlstScalar> Default for SingleLayerBoundaryIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}
