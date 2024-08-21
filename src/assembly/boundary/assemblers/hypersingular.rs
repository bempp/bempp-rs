//! Hypersingular assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::{
    boundary::integrands::{
        BoundaryIntegrandSum, HypersingularCurlCurlBoundaryIntegrand,
        HypersingularNormalNormalBoundaryIntegrand,
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
    HypersingularNormalNormalBoundaryIntegrand<T>,
>;

/// Assembler for a hypersingular operator
pub struct HypersingularAssembler<
    T: RlstScalar + MatrixInverse,
    K: Kernel<T = T>,
    I: BoundaryIntegrand<T = T> + Sync,
> {
    kernel: KernelEvaluator<T, K>,
    integrand: I,
    options: BoundaryAssemblerOptions,
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>, I: BoundaryIntegrand<T = T> + Sync>
    HypersingularAssembler<T, K, I>
{
    /// Create a new hypersingular assembler
    pub fn new(kernel: KernelEvaluator<T, K>, integrand: I) -> Self {
        Self {
            kernel,
            integrand,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse>
    HypersingularAssembler<T, Laplace3dKernel<T>, HypersingularCurlCurlBoundaryIntegrand<T>>
{
    /// Create a new Laplace hypersingular assembler
    pub fn new_laplace() -> Self {
        Self::new(
            KernelEvaluator::new_laplace(GreenKernelEvalType::ValueDeriv),
            HypersingularCurlCurlBoundaryIntegrand::new(),
        )
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    HypersingularAssembler<T, Helmholtz3dKernel<T>, HelmholtzIntegrand<T>>
{
    /// Create a new Helmholtz hypersingular assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(
            KernelEvaluator::new_helmholtz(wavenumber, GreenKernelEvalType::ValueDeriv),
            BoundaryIntegrandSum::new(
                HypersingularCurlCurlBoundaryIntegrand::new(),
                HypersingularNormalNormalBoundaryIntegrand::new(wavenumber),
            ),
        )
    }
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>, I: BoundaryIntegrand<T = T> + Sync>
    BoundaryAssembler for HypersingularAssembler<T, K, I>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = I;
    type Kernel = KernelEvaluator<T, K>;
    fn integrand(&self) -> &I {
        &self.integrand
    }
    fn kernel(&self) -> &KernelEvaluator<T, K> {
        &self.kernel
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        &mut self.options
    }
}
