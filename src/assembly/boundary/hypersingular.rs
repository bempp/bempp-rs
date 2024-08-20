//! Hypersingular assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::{
    boundary::integrands::HypersingularBoundaryIntegrand, common::GreenKernelEvalType,
    kernels::KernelEvaluator,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

/*
#[allow(clippy::too_many_arguments)]
unsafe fn hyp_test_trial_product<T: RlstScalar + MatrixInverse>(
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
*/
/// Assembler for a hypersingular operator
pub struct HypersingularAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    kernel: KernelEvaluator<T, K>,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> HypersingularAssembler<T, K> {
    /// Create a new hypersingular assembler
    pub fn new(kernel: KernelEvaluator<T, K>) -> Self {
        Self {
            kernel,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> HypersingularAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace hypersingular assembler
    pub fn new_laplace() -> Self {
        Self::new(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HypersingularAssembler<T, Helmholtz3dKernel<T>> {
    /// Create a new Helmholtz hypersingular assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}

impl<T: RlstScalar + MatrixInverse> BoundaryAssembler
    for HypersingularAssembler<T, Laplace3dKernel<T>>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Laplace3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Laplace3dKernel<T>> {
        &self.kernel
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        &mut self.options
    }
}

/// Assembler for curl-curl term of Helmholtz hypersingular operator
struct HelmholtzHypersingularCurlCurlAssembler<'a, T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
    options: &'a BoundaryAssemblerOptions,
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse>
    HelmholtzHypersingularCurlCurlAssembler<'a, T>
{
    /// Create a new assembler
    pub fn new(
        kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
        options: &'a BoundaryAssemblerOptions,
    ) -> Self {
        Self { kernel, options }
    }
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse> BoundaryAssembler
    for HelmholtzHypersingularCurlCurlAssembler<'a, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Helmholtz3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Helmholtz3dKernel<T>> {
        panic!();
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        panic!("Cannot get mutable options")
    }
}

/// Assembler for normal normal term of Helmholtz hypersingular boundary operator
struct HelmholtzHypersingularNormalNormalAssembler<'a, T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
    options: &'a BoundaryAssemblerOptions,
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse>
    HelmholtzHypersingularNormalNormalAssembler<'a, T>
{
    /// Create a new assembler
    pub fn new(
        kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
        options: &'a BoundaryAssemblerOptions,
    ) -> Self {
        Self { kernel, options }
    }
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse> BoundaryAssembler
    for HelmholtzHypersingularNormalNormalAssembler<'a, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Helmholtz3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Helmholtz3dKernel<T>> {
        panic!();
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        panic!("Cannot get mutable options")
    }
}

impl<T: RlstScalar<Complex = T> + MatrixInverse> BoundaryAssembler
    for HypersingularAssembler<T, Helmholtz3dKernel<T>>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Helmholtz3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Helmholtz3dKernel<T>> {
        panic!();
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        &mut self.options
    }
}
