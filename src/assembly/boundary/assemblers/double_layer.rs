//! Double layer assemblers
use super::BoundaryAssembler;
use crate::assembly::{
    boundary::integrands::DoubleLayerBoundaryIntegrand, common::GreenKernelEvalType,
    kernels::KernelEvaluator,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>>
    BoundaryAssembler<T, DoubleLayerBoundaryIntegrand<T>, KernelEvaluator<T, K>>
{
    /// Create a new double layer assembler
    pub fn new_double_layer(kernel: KernelEvaluator<T, K>) -> Self {
        Self::new(DoubleLayerBoundaryIntegrand::new(), kernel, 4, 0)
    }
}
impl<T: RlstScalar + MatrixInverse>
    BoundaryAssembler<T, DoubleLayerBoundaryIntegrand<T>, KernelEvaluator<T, Laplace3dKernel<T>>>
{
    /// Create a new Laplace double layer assembler
    pub fn new_laplace_double_layer() -> Self {
        Self::new_double_layer(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    BoundaryAssembler<T, DoubleLayerBoundaryIntegrand<T>, KernelEvaluator<T, Helmholtz3dKernel<T>>>
{
    /// Create a new Helmholtz double layer assembler
    pub fn new_helmholtz_double_layer(wavenumber: T::Real) -> Self {
        Self::new_double_layer(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
