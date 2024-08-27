//! Hypersingular integrand
use crate::assembly::common::RlstArray;
use crate::traits::{BoundaryIntegrand, CellGeometry};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

#[allow(clippy::too_many_arguments)]
unsafe fn hyp_test_trial_product<T: RlstScalar>(
    test_table: &RlstArray<T, 4>,
    trial_table: &RlstArray<T, 4>,
    test_jacobians: &RlstArray<T::Real, 2>,
    trial_jacobians: &RlstArray<T::Real, 2>,
    test_jdets: &[T::Real],
    trial_jdets: &[T::Real],
    test_point_index: usize,
    trial_point_index: usize,
    test_basis_index: usize,
    trial_basis_index: usize,
) -> T {
    let test0 = *test_table.get_unchecked([1, test_point_index, test_basis_index, 0]);
    let test1 = *test_table.get_unchecked([2, test_point_index, test_basis_index, 0]);
    let trial0 = *trial_table.get_unchecked([1, trial_point_index, trial_basis_index, 0]);
    let trial1 = *trial_table.get_unchecked([2, trial_point_index, trial_basis_index, 0]);

    ((num::cast::<T::Real, T>(*test_jacobians.get_unchecked([3, test_point_index])).unwrap()
        * test0
        - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([0, test_point_index])).unwrap()
            * test1)
        * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([3, trial_point_index]))
            .unwrap()
            * trial0
            - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([0, trial_point_index]))
                .unwrap()
                * trial1)
        + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([4, test_point_index])).unwrap()
            * test0
            - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([1, test_point_index]))
                .unwrap()
                * test1)
            * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([4, trial_point_index]))
                .unwrap()
                * trial0
                - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([1, trial_point_index]))
                    .unwrap()
                    * trial1)
        + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([5, test_point_index])).unwrap()
            * test0
            - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([2, test_point_index]))
                .unwrap()
                * test1)
            * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([5, trial_point_index]))
                .unwrap()
                * trial0
                - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([2, trial_point_index]))
                    .unwrap()
                    * trial1))
        / num::cast::<T::Real, T>(test_jdets[test_point_index] * trial_jdets[trial_point_index])
            .unwrap()
}

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

impl<T: RlstScalar> BoundaryIntegrand for HypersingularCurlCurlBoundaryIntegrand<T> {
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
        hyp_test_trial_product(
            test_table,
            trial_table,
            test_geometry.jacobians(),
            trial_geometry.jacobians(),
            test_geometry.jdets(),
            trial_geometry.jdets(),
            test_point_index,
            trial_point_index,
            test_basis_index,
            trial_basis_index,
        ) * *k.get_unchecked([0, test_point_index, trial_point_index])
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
        hyp_test_trial_product(
            test_table,
            trial_table,
            test_geometry.jacobians(),
            trial_geometry.jacobians(),
            test_geometry.jdets(),
            trial_geometry.jdets(),
            point_index,
            point_index,
            test_basis_index,
            trial_basis_index,
        ) * *k.get_unchecked([0, point_index])
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

impl<T: RlstScalar> BoundaryIntegrand for HypersingularNormalNormalBoundaryIntegrand<T> {
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
        *k.get_unchecked([0, test_point_index, trial_point_index])
            * num::cast::<T::Real, T>(
                *trial_geometry
                    .normals()
                    .get_unchecked([0, trial_point_index])
                    * *test_geometry.normals().get_unchecked([0, test_point_index])
                    + *trial_geometry
                        .normals()
                        .get_unchecked([1, trial_point_index])
                        * *test_geometry.normals().get_unchecked([1, test_point_index])
                    + *trial_geometry
                        .normals()
                        .get_unchecked([2, trial_point_index])
                        * *test_geometry.normals().get_unchecked([2, test_point_index]),
            )
            .unwrap()
            * *test_table.get_unchecked([0, test_point_index, test_basis_index, 0])
            * *trial_table.get_unchecked([0, trial_point_index, trial_basis_index, 0])
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
        *k.get_unchecked([0, point_index])
            * num::cast::<T::Real, T>(
                *trial_geometry.normals().get_unchecked([0, point_index])
                    * *test_geometry.normals().get_unchecked([0, point_index])
                    + *trial_geometry.normals().get_unchecked([1, point_index])
                        * *test_geometry.normals().get_unchecked([1, point_index])
                    + *trial_geometry.normals().get_unchecked([2, point_index])
                        * *test_geometry.normals().get_unchecked([2, point_index]),
            )
            .unwrap()
            * *test_table.get_unchecked([0, point_index, test_basis_index, 0])
            * *trial_table.get_unchecked([0, point_index, trial_basis_index, 0])
    }
}
