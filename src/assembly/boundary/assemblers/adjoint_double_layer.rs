//! Adjoint double layer assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::{
    boundary::integrands::AdjointDoubleLayerBoundaryIntegrand, common::GreenKernelEvalType,
    kernels::KernelEvaluator,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

/// Assembler for a adjoint double layer operator
pub struct AdjointDoubleLayerAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    integrand: AdjointDoubleLayerBoundaryIntegrand<T>,
    kernel: KernelEvaluator<T, K>,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> AdjointDoubleLayerAssembler<T, K> {
    /// Create a new adjoint double layer assembler
    pub fn new(kernel: KernelEvaluator<T, K>) -> Self {
        Self {
            integrand: AdjointDoubleLayerBoundaryIntegrand::new(),
            kernel,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> AdjointDoubleLayerAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace adjoint double layer assembler
    pub fn new_laplace() -> Self {
        Self::new(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    AdjointDoubleLayerAssembler<T, Helmholtz3dKernel<T>>
{
    /// Create a new Helmholtz adjoint double layer assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> BoundaryAssembler
    for AdjointDoubleLayerAssembler<T, K>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    type Integrand = AdjointDoubleLayerBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, K>;
    fn integrand(&self) -> &AdjointDoubleLayerBoundaryIntegrand<T> {
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
