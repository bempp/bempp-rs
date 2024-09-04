//! Adjoint double layer integrand
use crate::traits::{Access1D, Access2D, BoundaryIntegrand, GeometryAccess};
use rlst::RlstScalar;

pub struct AdjointDoubleLayerBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> AdjointDoubleLayerBoundaryIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

unsafe impl<T: RlstScalar> BoundaryIntegrand for AdjointDoubleLayerBoundaryIntegrand<T> {
    type T = T;

    fn evaluate(
        &self,
        k: &impl Access1D<T = T>,
        test_table: &impl Access2D<T = T>,
        trial_table: &impl Access2D<T = T>,
        test_geometry: &impl GeometryAccess<T = T>,
        _trial_geometry: &impl GeometryAccess<T = T>,
    ) -> T {
        unsafe {
            -(k.get(1) * test_geometry.normal(0)
                + k.get(2) * test_geometry.normal(1)
                + k.get(3) * test_geometry.normal(2))
                * test_table.get(0, 0)
                * trial_table.get(0, 0)
        }
    }
}

impl<T: RlstScalar> Default for AdjointDoubleLayerBoundaryIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}
