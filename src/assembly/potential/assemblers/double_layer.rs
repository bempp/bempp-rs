//! Double layer assemblers
use super::PotentialAssembler;
use crate::assembly::{
    common::GreenKernelEvalType, kernels::KernelEvaluator,
    potential::integrands::DoubleLayerPotentialIntegrand,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>>
    PotentialAssembler<T, DoubleLayerPotentialIntegrand<T>, KernelEvaluator<T, K>>
{
    /// Create a new Double layer assembler
    pub fn new_double_layer(kernel: KernelEvaluator<T, K>) -> Self {
        Self::new(DoubleLayerPotentialIntegrand::new(), kernel, 4)
    }
}
impl<T: RlstScalar + MatrixInverse>
    PotentialAssembler<T, DoubleLayerPotentialIntegrand<T>, KernelEvaluator<T, Laplace3dKernel<T>>>
{
    /// Create a new Laplace Double layer assembler
    pub fn new_laplace_double_layer() -> Self {
        Self::new_double_layer(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    PotentialAssembler<
        T,
        DoubleLayerPotentialIntegrand<T>,
        KernelEvaluator<T, Helmholtz3dKernel<T>>,
    >
{
    /// Create a new Helmholtz Double layer assembler
    pub fn new_helmholtz_double_layer(wavenumber: T::Real) -> Self {
        Self::new_double_layer(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
