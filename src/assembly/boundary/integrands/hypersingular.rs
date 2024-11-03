//! Hypersingular integrand
use rlst::RlstScalar;

use super::{Access1D, Access2D, BoundaryIntegrand, GeometryAccess};

/// Integrand for the curl curl term of a hypersingular boundary operator
pub struct HypersingularCurlCurlBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> HypersingularCurlCurlBoundaryIntegrand<T> {
    /// Create new
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

/// Integrand for the normal normal term of a hypersingular boundary operator
pub struct HypersingularNormalNormalBoundaryIntegrand<T: RlstScalar> {
    _t: std::marker::PhantomData<T>,
}

impl<T: RlstScalar> HypersingularNormalNormalBoundaryIntegrand<T> {
    /// Create new
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

    fn evaluate_nonsingular(
        &self,
        test_table: &crate::assembly::common::RlstArray<Self::T, 4>,
        trial_table: &crate::assembly::common::RlstArray<Self::T, 4>,
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &crate::assembly::common::RlstArray<Self::T, 3>,
        test_geometry: &impl crate::traits::CellGeometry<T = <Self::T as RlstScalar>::Real>,
        trial_geometry: &impl crate::traits::CellGeometry<T = <Self::T as RlstScalar>::Real>,
    ) -> Self::T {
        self.evaluate(
            &super::NonSingularKernel::new(k, test_point_index, trial_point_index),
            &super::Table::new(test_table, test_point_index, test_basis_index),
            &super::Table::new(trial_table, trial_point_index, trial_basis_index),
            &super::Geometry::new(test_geometry, test_point_index),
            &super::Geometry::new(trial_geometry, trial_point_index),
        )
    }

    fn evaluate_singular(
        &self,
        test_table: &crate::assembly::common::RlstArray<Self::T, 4>,
        trial_table: &crate::assembly::common::RlstArray<Self::T, 4>,
        point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &crate::assembly::common::RlstArray<Self::T, 2>,
        test_geometry: &impl crate::traits::CellGeometry<T = <Self::T as RlstScalar>::Real>,
        trial_geometry: &impl crate::traits::CellGeometry<T = <Self::T as RlstScalar>::Real>,
    ) -> Self::T {
        self.evaluate(
            &super::SingularKernel::new(k, point_index),
            &super::Table::new(test_table, point_index, test_basis_index),
            &super::Table::new(trial_table, point_index, trial_basis_index),
            &super::Geometry::new(test_geometry, point_index),
            &super::Geometry::new(trial_geometry, point_index),
        )
    }
}
