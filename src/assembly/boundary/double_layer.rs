//! Double layer assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::{
    boundary::integrands::DoubleLayerBoundaryIntegrand, common::GreenKernelEvalType,
    kernels::KernelEvaluator,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

/// Assembler for a double layer operator
pub struct DoubleLayerAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    integrand: DoubleLayerBoundaryIntegrand<T>,
    kernel: KernelEvaluator<T, K>,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> DoubleLayerAssembler<T, K> {
    /// Create a new double layer assembler
    pub fn new(kernel: KernelEvaluator<T, K>) -> Self {
        Self {
            integrand: DoubleLayerBoundaryIntegrand::new(),
            kernel,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> DoubleLayerAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace double layer assembler
    pub fn new_laplace() -> Self {
        Self::new(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> DoubleLayerAssembler<T, Helmholtz3dKernel<T>> {
    /// Create a new Helmholtz double layer assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> BoundaryAssembler
    for DoubleLayerAssembler<T, K>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    type Integrand = DoubleLayerBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, K>;
    fn integrand(&self) -> &DoubleLayerBoundaryIntegrand<T> {
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
