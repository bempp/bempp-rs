//! Single layer assemblers
use super::PotentialAssembler;
use crate::assembly::{
    common::GreenKernelEvalType, kernels::KernelEvaluator,
    potential::integrands::SingleLayerPotentialIntegrand,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar};

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>>
    PotentialAssembler<T, SingleLayerPotentialIntegrand<T>, KernelEvaluator<T, K>>
{
    /// Create a new single layer assembler
    pub fn new_single_layer(kernel: KernelEvaluator<T, K>) -> Self {
        Self::new(SingleLayerPotentialIntegrand::new(), kernel, 1)
    }
}
impl<T: RlstScalar + MatrixInverse>
    PotentialAssembler<T, SingleLayerPotentialIntegrand<T>, KernelEvaluator<T, Laplace3dKernel<T>>>
{
    /// Create a new Laplace single layer assembler
    pub fn new_laplace_single_layer() -> Self {
        Self::new_single_layer(KernelEvaluator::new_laplace(GreenKernelEvalType::Value))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    PotentialAssembler<
        T,
        SingleLayerPotentialIntegrand<T>,
        KernelEvaluator<T, Helmholtz3dKernel<T>>,
    >
{
    /// Create a new Helmholtz single layer assembler
    pub fn new_helmholtz_single_layer(wavenumber: T::Real) -> Self {
        Self::new_single_layer(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::Value,
        ))
    }
}
