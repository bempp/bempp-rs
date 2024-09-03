//! Integrands
mod adjoint_double_layer;
mod double_layer;
mod hypersingular;
mod single_layer;

pub use adjoint_double_layer::AdjointDoubleLayerBoundaryIntegrand;
pub use double_layer::DoubleLayerBoundaryIntegrand;
pub use hypersingular::{
    HypersingularCurlCurlBoundaryIntegrand, HypersingularNormalNormalBoundaryIntegrand,
};
pub use single_layer::SingleLayerBoundaryIntegrand;

use crate::traits::{Access1D, Access2D, BoundaryIntegrand, GeometryAccess};
use rlst::RlstScalar;

/// The sum of two integrands
pub struct BoundaryIntegrandSum<
    T: RlstScalar,
    I0: BoundaryIntegrand<T = T>,
    I1: BoundaryIntegrand<T = T>,
> {
    integrand0: I0,
    integrand1: I1,
}

impl<T: RlstScalar, I0: BoundaryIntegrand<T = T>, I1: BoundaryIntegrand<T = T>>
    BoundaryIntegrandSum<T, I0, I1>
{
    pub fn new(integrand0: I0, integrand1: I1) -> Self {
        Self {
            integrand0,
            integrand1,
        }
    }
}

unsafe impl<T: RlstScalar, I0: BoundaryIntegrand<T = T>, I1: BoundaryIntegrand<T = T>>
    BoundaryIntegrand for BoundaryIntegrandSum<T, I0, I1>
{
    type T = T;

    fn evaluate(
        &self,
        k: &impl Access1D<T = T>,
        test_table: &impl Access2D<T = T>,
        trial_table: &impl Access2D<T = T>,
        test_geometry: &impl GeometryAccess<T = T>,
        trial_geometry: &impl GeometryAccess<T = T>,
    ) -> T {
        self.integrand0
            .evaluate(k, test_table, trial_table, test_geometry, trial_geometry)
            + self
                .integrand1
                .evaluate(k, test_table, trial_table, test_geometry, trial_geometry)
    }
}

/// An integrand multiplied by a scalar
pub struct BoundaryIntegrandScalarProduct<T: RlstScalar, I: BoundaryIntegrand<T = T>> {
    scalar: T,
    integrand: I,
}

impl<T: RlstScalar, I: BoundaryIntegrand<T = T>> BoundaryIntegrandScalarProduct<T, I> {
    pub fn new(scalar: T, integrand: I) -> Self {
        Self { scalar, integrand }
    }
}

unsafe impl<T: RlstScalar, I: BoundaryIntegrand<T = T>> BoundaryIntegrand
    for BoundaryIntegrandScalarProduct<T, I>
{
    type T = T;

    fn evaluate(
        &self,
        k: &impl Access1D<T = T>,
        test_table: &impl Access2D<T = T>,
        trial_table: &impl Access2D<T = T>,
        test_geometry: &impl GeometryAccess<T = T>,
        trial_geometry: &impl GeometryAccess<T = T>,
    ) -> T {
        self.scalar
            * self
                .integrand
                .evaluate(k, test_table, trial_table, test_geometry, trial_geometry)
    }
}
