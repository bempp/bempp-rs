//! Single layer integrand
use crate::assembly::common::RlstArray;
use crate::traits::{CellGeometry, PotentialIntegrand};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

pub struct SingleLayerPotentialIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> SingleLayerPotentialIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

impl<T: RlstScalar> PotentialIntegrand for SingleLayerPotentialIntegrand<T> {
    type T = T;

    unsafe fn evaluate(
        &self,
        table: &RlstArray<T, 4>,
        point_index: usize,
        eval_index: usize,
        basis_index: usize,
        k: &RlstArray<T, 3>,
        _geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        *k.get_unchecked([0, basis_index, eval_index])
            * *table.get_unchecked([0, point_index, basis_index, 0])
    }
}

impl<T: RlstScalar> Default for SingleLayerPotentialIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}
