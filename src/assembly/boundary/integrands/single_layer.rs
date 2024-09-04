//! Single layer integrand
use crate::traits::{Access1D, Access2D, BoundaryIntegrand, GeometryAccess};
use rlst::RlstScalar;

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

    fn evaluate(
        &self,
        k: &impl Access1D<T = T>,
        test_table: &impl Access2D<T = T>,
        trial_table: &impl Access2D<T = T>,
        _test_geometry: &impl GeometryAccess<T = T>,
        _trial_geometry: &impl GeometryAccess<T = T>,
    ) -> T {
        unsafe { k.get(0) * test_table.get(0, 0) * trial_table.get(0, 0) }
    }
}

impl<T: RlstScalar> Default for SingleLayerBoundaryIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}
