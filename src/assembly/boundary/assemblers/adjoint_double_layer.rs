//! Adjoint double layer assemblers
use super::BoundaryAssembler;
use crate::assembly::{
    boundary::integrands::AdjointDoubleLayerBoundaryIntegrand, common::GreenKernelEvalType,
    kernels::KernelEvaluator,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>>
    BoundaryAssembler<T, AdjointDoubleLayerBoundaryIntegrand<T>, KernelEvaluator<T, K>>
{
    /// Create a new adjoint double layer assembler
    pub fn new_adjoint_double_layer(kernel: KernelEvaluator<T, K>) -> Self {
        Self::new(AdjointDoubleLayerBoundaryIntegrand::new(), kernel, 4, 0)
    }
}
impl<T: RlstScalar + MatrixInverse>
    BoundaryAssembler<
        T,
        AdjointDoubleLayerBoundaryIntegrand<T>,
        KernelEvaluator<T, Laplace3dKernel<T>>,
    >
{
    /// Create a new Laplace adjoint double layer assembler
    pub fn new_laplace_adjoint_double_layer() -> Self {
        Self::new_adjoint_double_layer(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    BoundaryAssembler<
        T,
        AdjointDoubleLayerBoundaryIntegrand<T>,
        KernelEvaluator<T, Helmholtz3dKernel<T>>,
    >
{
    /// Create a new Helmholtz adjoint double layer assembler
    pub fn new_helmholtz_adjoint_double_layer(wavenumber: T::Real) -> Self {
        Self::new_adjoint_double_layer(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
