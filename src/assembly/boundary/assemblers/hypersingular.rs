//! Hypersingular assemblers
use super::BoundaryAssembler;
use crate::assembly::{
    boundary::integrands::{
        BoundaryIntegrandScalarProduct, BoundaryIntegrandSum,
        HypersingularCurlCurlBoundaryIntegrand, HypersingularNormalNormalBoundaryIntegrand,
    },
    common::GreenKernelEvalType,
    kernels::KernelEvaluator,
};
use crate::traits::BoundaryIntegrand;
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

type HelmholtzIntegrand<T> = BoundaryIntegrandSum<
    T,
    HypersingularCurlCurlBoundaryIntegrand<T>,
    BoundaryIntegrandScalarProduct<T, HypersingularNormalNormalBoundaryIntegrand<T>>,
>;

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>, I: BoundaryIntegrand<T = T>>
    BoundaryAssembler<T, I, KernelEvaluator<T, K>>
{
    /// Create a new adjoint double layer assembler
    pub fn new_hypersingular(integrand: I, kernel: KernelEvaluator<T, K>) -> Self {
        Self::new(integrand, kernel, 4, 1)
    }
}
impl<T: RlstScalar + MatrixInverse>
    BoundaryAssembler<
        T,
        HypersingularCurlCurlBoundaryIntegrand<T>,
        KernelEvaluator<T, Laplace3dKernel<T>>,
    >
{
    /// Create a new Laplace adjoint double layer assembler
    pub fn new_laplace_hypersingular() -> Self {
        Self::new_hypersingular(
            HypersingularCurlCurlBoundaryIntegrand::new(),
            KernelEvaluator::new_laplace(GreenKernelEvalType::ValueDeriv),
        )
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    BoundaryAssembler<T, HelmholtzIntegrand<T>, KernelEvaluator<T, Helmholtz3dKernel<T>>>
{
    /// Create a new Helmholtz adjoint double layer assembler
    pub fn new_helmholtz_hypersingular(wavenumber: T::Real) -> Self {
        Self::new_hypersingular(
            BoundaryIntegrandSum::new(
                HypersingularCurlCurlBoundaryIntegrand::new(),
                BoundaryIntegrandScalarProduct::new(
                    num::cast::<T::Real, T>(-wavenumber.powi(2)).unwrap(),
                    HypersingularNormalNormalBoundaryIntegrand::new(),
                ),
            ),
            KernelEvaluator::new_helmholtz(wavenumber, GreenKernelEvalType::ValueDeriv),
        )
    }
}
