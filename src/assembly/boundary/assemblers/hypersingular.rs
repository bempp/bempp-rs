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
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

/// Assembler for a hypersingular operator
pub struct HypersingularAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    kernel: KernelEvaluator<T, K>,
    integrand: HypersingularCurlCurlBoundaryIntegrand<T>,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> HypersingularAssembler<T, K> {
    /// Create a new hypersingular assembler
    pub fn new(kernel: KernelEvaluator<T, K>) -> Self {
        Self {
            kernel,
            integrand: HypersingularCurlCurlBoundaryIntegrand::new(),
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
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularCurlCurlBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Laplace3dKernel<T>>;
    fn integrand(&self) -> &HypersingularCurlCurlBoundaryIntegrand<T> {
        &self.integrand
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

impl<T: RlstScalar<Complex = T> + MatrixInverse> BoundaryAssembler
    for HypersingularAssembler<T, Helmholtz3dKernel<T>>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularCurlCurlBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Helmholtz3dKernel<T>>;
    fn integrand(&self) -> &HypersingularCurlCurlBoundaryIntegrand<T> {
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
