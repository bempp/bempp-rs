//! Hypersingular integrand
use crate::traits::{Access1D, Access2D, BoundaryIntegrand, GeometryAccess};
use rlst::RlstScalar;

pub struct HypersingularCurlCurlBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> HypersingularCurlCurlBoundaryIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

impl<T: RlstScalar> Default for HypersingularCurlCurlBoundaryIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: RlstScalar> BoundaryIntegrand for HypersingularCurlCurlBoundaryIntegrand<T> {
    type T = T;

    fn evaluate(
        &self,
        k: &impl Access1D<T = T>,
        test_table: &impl Access2D<T = T>,
        trial_table: &impl Access2D<T = T>,
        test_geometry: &impl GeometryAccess<T = T>,
        trial_geometry: &impl GeometryAccess<T = T>,
    ) -> T {
        unsafe {
            let test0 = test_table.get(1, 0);
            let test1 = test_table.get(2, 0);
            let trial0 = trial_table.get(1, 0);
            let trial1 = trial_table.get(2, 0);

            k.get(0)
                * ((test_geometry.jacobian(3) * test0 - test_geometry.jacobian(0) * test1)
                    * (trial_geometry.jacobian(3) * trial0 - trial_geometry.jacobian(0) * trial1)
                    + (test_geometry.jacobian(4) * test0 - test_geometry.jacobian(1) * test1)
                        * (trial_geometry.jacobian(4) * trial0
                            - trial_geometry.jacobian(1) * trial1)
                    + (test_geometry.jacobian(5) * test0 - test_geometry.jacobian(2) * test1)
                        * (trial_geometry.jacobian(5) * trial0
                            - trial_geometry.jacobian(2) * trial1))
                / test_geometry.jdet()
                / trial_geometry.jdet()
        }
    }
}

pub struct HypersingularNormalNormalBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> HypersingularNormalNormalBoundaryIntegrand<T> {
    pub fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

impl<T: RlstScalar> Default for HypersingularNormalNormalBoundaryIntegrand<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: RlstScalar> BoundaryIntegrand for HypersingularNormalNormalBoundaryIntegrand<T> {
    type T = T;

    fn evaluate(
        &self,
        k: &impl Access1D<T = T>,
        test_table: &impl Access2D<T = T>,
        trial_table: &impl Access2D<T = T>,
        test_geometry: &impl GeometryAccess<T = T>,
        trial_geometry: &impl GeometryAccess<T = T>,
    ) -> T {
        unsafe {
            k.get(0)
                * (trial_geometry.normal(0) * test_geometry.normal(0)
                    + trial_geometry.normal(1) * test_geometry.normal(1)
                    + trial_geometry.normal(2) * test_geometry.normal(2))
                * test_table.get(0, 0)
                * trial_table.get(0, 0)
        }
    }
}
