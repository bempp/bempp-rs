//! Hypersingular integrand
use crate::assembly::common::RlstArray;
use crate::traits::{BoundaryIntegrand, CellGeometry};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

pub struct HypersingularBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> HypersingularBoundaryIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

impl<T: RlstScalar> BoundaryIntegrand for HypersingularBoundaryIntegrand<T> {
    type T = T;

    unsafe fn evaluate_nonsingular(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<T, 3>,
        _test_geometry: &impl CellGeometry<T = T::Real>,
        trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        panic!();
    }

    unsafe fn evaluate_singular(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 2>,
        _test_geometry: &impl CellGeometry<T = T::Real>,
        trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        panic!();
    }
}
