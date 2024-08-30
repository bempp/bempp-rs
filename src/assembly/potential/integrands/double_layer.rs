//! Double layer integrand
use crate::assembly::common::RlstArray;
use crate::traits::{CellGeometry, PotentialIntegrand};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

pub struct DoubleLayerPotentialIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> DoubleLayerPotentialIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

unsafe impl<T: RlstScalar> PotentialIntegrand for DoubleLayerPotentialIntegrand<T> {
    type T = T;

    fn evaluate(
        &self,
        table: &RlstArray<T, 4>,
        point_index: usize,
        eval_index: usize,
        basis_index: usize,
        k: &RlstArray<T, 3>,
        geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        unsafe {
            -(*k.get_unchecked([1, point_index, eval_index])
                * num::cast::<T::Real, T>(*geometry.normals().get_unchecked([0, point_index]))
                    .unwrap()
                + *k.get_unchecked([2, point_index, eval_index])
                    * num::cast::<T::Real, T>(*geometry.normals().get_unchecked([1, point_index]))
                        .unwrap()
                + *k.get_unchecked([3, point_index, eval_index])
                    * num::cast::<T::Real, T>(*geometry.normals().get_unchecked([2, point_index]))
                        .unwrap())
                * *table.get_unchecked([0, point_index, basis_index, 0])
        }
    }
}

impl<T: RlstScalar> Default for DoubleLayerPotentialIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}
