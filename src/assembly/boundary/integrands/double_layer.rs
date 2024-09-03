//! Double layer integrand
use crate::traits::{Access1D, Access2D, BoundaryIntegrand, GeometryAccess};
use rlst::RlstScalar;

pub struct DoubleLayerBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> DoubleLayerBoundaryIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

unsafe impl<T: RlstScalar> BoundaryIntegrand for DoubleLayerBoundaryIntegrand<T> {
    type T = T;

    fn evaluate(
        &self,
        k: &impl Access1D<T = T>,
        test_table: &impl Access2D<T = T>,
        trial_table: &impl Access2D<T = T>,
        _test_geometry: &impl GeometryAccess<T = T>,
        trial_geometry: &impl GeometryAccess<T = T>,
    ) -> T {
        unsafe {
            (k.get(1) * trial_geometry.normal(0)
                + k.get(2) * trial_geometry.normal(1)
                + k.get(3) * trial_geometry.normal(2))
                * test_table.get(0, 0)
                * trial_table.get(0, 0)
        }
    }
}

impl<T: RlstScalar> Default for DoubleLayerBoundaryIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}
