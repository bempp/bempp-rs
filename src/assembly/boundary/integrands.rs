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

use crate::assembly::common::RlstArray;
use crate::traits::{BoundaryIntegrand, CellGeometry};
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

impl<T: RlstScalar, I0: BoundaryIntegrand<T = T>, I1: BoundaryIntegrand<T = T>> BoundaryIntegrand
    for BoundaryIntegrandSum<T, I0, I1>
{
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
        test_geometry: &impl CellGeometry<T = T::Real>,
        trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        self.integrand0.evaluate_nonsingular(
            test_table,
            trial_table,
            test_point_index,
            trial_point_index,
            test_basis_index,
            trial_basis_index,
            k,
            test_geometry,
            trial_geometry,
        ) + self.integrand1.evaluate_nonsingular(
            test_table,
            trial_table,
            test_point_index,
            trial_point_index,
            test_basis_index,
            trial_basis_index,
            k,
            test_geometry,
            trial_geometry,
        )
    }

    unsafe fn evaluate_singular(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 2>,
        test_geometry: &impl CellGeometry<T = T::Real>,
        trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        self.integrand0.evaluate_singular(
            test_table,
            trial_table,
            point_index,
            test_basis_index,
            trial_basis_index,
            k,
            test_geometry,
            trial_geometry,
        ) + self.integrand1.evaluate_singular(
            test_table,
            trial_table,
            point_index,
            test_basis_index,
            trial_basis_index,
            k,
            test_geometry,
            trial_geometry,
        )
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

impl<T: RlstScalar, I: BoundaryIntegrand<T = T>> BoundaryIntegrand
    for BoundaryIntegrandScalarProduct<T, I>
{
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
        test_geometry: &impl CellGeometry<T = T::Real>,
        trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        self.scalar
            * self.integrand.evaluate_nonsingular(
                test_table,
                trial_table,
                test_point_index,
                trial_point_index,
                test_basis_index,
                trial_basis_index,
                k,
                test_geometry,
                trial_geometry,
            )
    }

    unsafe fn evaluate_singular(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 2>,
        test_geometry: &impl CellGeometry<T = T::Real>,
        trial_geometry: &impl CellGeometry<T = T::Real>,
    ) -> T {
        self.scalar
            * self.integrand.evaluate_singular(
                test_table,
                trial_table,
                point_index,
                test_basis_index,
                trial_basis_index,
                k,
                test_geometry,
                trial_geometry,
            )
    }
}
