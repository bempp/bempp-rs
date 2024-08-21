//! Green's function kernels
use crate::assembly::common::GreenKernelEvalType;
use crate::traits::KernelEvaluator as KernelEvaluatorTrait;
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::RlstScalar;

/// Kernel evaluator
pub struct KernelEvaluator<T: RlstScalar, K: Kernel<T = T>> {
    pub(crate) kernel: K,
    eval_type: GreenKernelEvalType,
}

impl<T: RlstScalar, K: Kernel<T = T>> KernelEvaluator<T, K> {
    /// Create new
    pub fn new(kernel: K, eval_type: GreenKernelEvalType) -> Self {
        Self { kernel, eval_type }
    }
}
impl<T: RlstScalar> KernelEvaluator<T, Laplace3dKernel<T>> {
    /// Create new Laplace kernel
    pub fn new_laplace(eval_type: GreenKernelEvalType) -> Self {
        Self::new(Laplace3dKernel::<T>::new(), eval_type)
    }
}
impl<T: RlstScalar<Complex = T>> KernelEvaluator<T, Helmholtz3dKernel<T>> {
    /// Create new Helmholtz kernel
    pub fn new_helmholtz(wavenumber: T::Real, eval_type: GreenKernelEvalType) -> Self {
        Self::new(Helmholtz3dKernel::<T>::new(wavenumber), eval_type)
    }
}
impl<T: RlstScalar, K: Kernel<T = T>> KernelEvaluatorTrait for KernelEvaluator<T, K> {
    type T = T;

    fn assemble_pairwise_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_pairwise_st(self.eval_type, sources, targets, result);
    }

    fn assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(self.eval_type, sources, targets, result);
    }
}
