//! Integrands
mod adjoint_double_layer;
mod double_layer;
mod hypersingular;
mod single_layer;

use std::marker::PhantomData;

pub use adjoint_double_layer::AdjointDoubleLayerBoundaryIntegrand;
pub use double_layer::DoubleLayerBoundaryIntegrand;
pub use hypersingular::{
    HypersingularCurlCurlBoundaryIntegrand, HypersingularNormalNormalBoundaryIntegrand,
};
pub use single_layer::SingleLayerBoundaryIntegrand;

use crate::boundary_assemblers::helpers::{CellGeometry, RlstArray};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

/// 1D access
pub trait Access1D {
    /// Value tyoe
    type T;
    /// Get value
    ///
    /// # Safety
    /// This function uses unsafe memory access
    unsafe fn get(&self, i: usize) -> Self::T;
}

/// 2D access
pub trait Access2D {
    /// Value tyoe
    type T;
    /// Get value
    ///
    /// # Safety
    /// This function uses unsafe memory access
    unsafe fn get(&self, i: usize, j: usize) -> Self::T;
}

/// Geometry access
pub trait GeometryAccess {
    /// Value tyoe
    type T;
    /// Get component of the point
    ///
    /// # Safety
    /// This function uses unsafe memory access
    unsafe fn point(&self, i: usize) -> Self::T;
    /// Get component of the normal
    ///
    /// # Safety
    /// This function uses unsafe memory access
    unsafe fn normal(&self, i: usize) -> Self::T;
    /// Get component of the jacobian
    ///
    /// # Safety
    /// This function uses unsafe memory access
    unsafe fn jacobian(&self, i: usize) -> Self::T;
    /// Get determinant of the jacobian
    ///
    /// # Safety
    /// This function uses unsafe memory access
    unsafe fn jdet(&self) -> Self::T;
}

/// Non-singular kernel with 1D access
struct NonSingularKernel<'a, T: RlstScalar> {
    k: &'a RlstArray<T, 3>,
    test_point_index: usize,
    trial_point_index: usize,
}

impl<'a, T: RlstScalar> NonSingularKernel<'a, T> {
    /// Create new
    fn new(k: &'a RlstArray<T, 3>, test_point_index: usize, trial_point_index: usize) -> Self {
        Self {
            k,
            test_point_index,
            trial_point_index,
        }
    }
}

impl<T: RlstScalar> Access1D for NonSingularKernel<'_, T> {
    type T = T;

    unsafe fn get(&self, i: usize) -> Self::T {
        *self
            .k
            .get_unchecked([i, self.test_point_index, self.trial_point_index])
    }
}

/// Singular kernel with 1D access
struct SingularKernel<'a, T: RlstScalar> {
    k: &'a RlstArray<T, 2>,
    point_index: usize,
}

impl<'a, T: RlstScalar> SingularKernel<'a, T> {
    fn new(k: &'a RlstArray<T, 2>, point_index: usize) -> Self {
        Self { k, point_index }
    }
}

impl<T: RlstScalar> Access1D for SingularKernel<'_, T> {
    type T = T;

    unsafe fn get(&self, i: usize) -> Self::T {
        *self.k.get_unchecked([i, self.point_index])
    }
}

/// Entry in tabulated data
struct Table<'a, T: RlstScalar> {
    table: &'a RlstArray<T, 4>,
    point_index: usize,
    basis_index: usize,
}

impl<'a, T: RlstScalar> Table<'a, T> {
    fn new(table: &'a RlstArray<T, 4>, point_index: usize, basis_index: usize) -> Self {
        Self {
            table,
            point_index,
            basis_index,
        }
    }
}
impl<T: RlstScalar> Access2D for Table<'_, T> {
    type T = T;
    unsafe fn get(&self, i: usize, j: usize) -> Self::T {
        *self
            .table
            .get_unchecked([i, self.point_index, self.basis_index, j])
    }
}

/// Geometry for a point
struct Geometry<'a, T: RlstScalar, G: CellGeometry<T = T::Real>> {
    geometry: &'a G,
    point_index: usize,
    _t: PhantomData<T>,
}

impl<'a, T: RlstScalar, G: CellGeometry<T = T::Real>> Geometry<'a, T, G> {
    fn new(geometry: &'a G, point_index: usize) -> Self {
        Self {
            geometry,
            point_index,
            _t: PhantomData,
        }
    }
}
impl<T: RlstScalar, G: CellGeometry<T = T::Real>> GeometryAccess for Geometry<'_, T, G> {
    type T = T;
    unsafe fn point(&self, i: usize) -> Self::T {
        T::from(*self.geometry.points().get_unchecked([i, self.point_index])).unwrap()
    }
    unsafe fn normal(&self, i: usize) -> Self::T {
        T::from(*self.geometry.normals().get_unchecked([i, self.point_index])).unwrap()
    }
    unsafe fn jacobian(&self, i: usize) -> Self::T {
        T::from(
            *self
                .geometry
                .jacobians()
                .get_unchecked([i, self.point_index]),
        )
        .unwrap()
    }
    unsafe fn jdet(&self) -> Self::T {
        T::from(*self.geometry.jdets().get_unchecked(self.point_index)).unwrap()
    }
}

pub unsafe trait BoundaryIntegrand: Sync {
    //! Integrand
    //!
    //! # Safety
    //! This trait's methods use unsafe access

    /// Scalar type
    type T: RlstScalar;

    /// Evaluate integrand
    fn evaluate(
        &self,
        k: &impl Access1D<T = Self::T>,
        test_table: &impl Access2D<T = Self::T>,
        trial_table: &impl Access2D<T = Self::T>,
        test_geometry: &impl GeometryAccess<T = Self::T>,
        trial_geometry: &impl GeometryAccess<T = Self::T>,
    ) -> Self::T;

    #[allow(clippy::too_many_arguments)]
    /// Evaluate integrand for a singular quadrature rule
    fn evaluate_nonsingular(
        &self,
        test_table: &RlstArray<Self::T, 4>,
        trial_table: &RlstArray<Self::T, 4>,
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 3>,
        test_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
        trial_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
    ) -> Self::T {
        self.evaluate(
            &NonSingularKernel::new(k, test_point_index, trial_point_index),
            &Table::new(test_table, test_point_index, test_basis_index),
            &Table::new(trial_table, trial_point_index, trial_basis_index),
            &Geometry::new(test_geometry, test_point_index),
            &Geometry::new(trial_geometry, trial_point_index),
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Evaluate integrand for a non-singular quadrature rule
    fn evaluate_singular(
        &self,
        test_table: &RlstArray<Self::T, 4>,
        trial_table: &RlstArray<Self::T, 4>,
        point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
        k: &RlstArray<Self::T, 2>,
        test_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
        trial_geometry: &impl CellGeometry<T = <Self::T as RlstScalar>::Real>,
    ) -> Self::T {
        self.evaluate(
            &SingularKernel::new(k, point_index),
            &Table::new(test_table, point_index, test_basis_index),
            &Table::new(trial_table, point_index, trial_basis_index),
            &Geometry::new(test_geometry, point_index),
            &Geometry::new(trial_geometry, point_index),
        )
    }
}

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
    /// Create new
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
pub struct BoundaryIntegrandTimesScalar<T: RlstScalar, I: BoundaryIntegrand<T = T>> {
    scalar: T,
    integrand: I,
}

impl<T: RlstScalar, I: BoundaryIntegrand<T = T>> BoundaryIntegrandTimesScalar<T, I> {
    /// Create new
    pub fn new(scalar: T, integrand: I) -> Self {
        Self { scalar, integrand }
    }
}

unsafe impl<T: RlstScalar, I: BoundaryIntegrand<T = T>> BoundaryIntegrand
    for BoundaryIntegrandTimesScalar<T, I>
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
