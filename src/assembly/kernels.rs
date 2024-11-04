//! Green's function kernels
use crate::assembly::common::GreenKernelEvalType;
use green_kernels::traits::Kernel;
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

    /// Assemble pairwise.
    pub fn assemble_pairwise_st(
        &self,
        sources: &[<T as RlstScalar>::Real],
        targets: &[<T as RlstScalar>::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_pairwise_st(self.eval_type, sources, targets, result);
    }

    /// Assemble all sources against all targets.
    pub fn assemble_st(
        &self,
        sources: &[<T as RlstScalar>::Real],
        targets: &[<T as RlstScalar>::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_st(self.eval_type, sources, targets, result);
    }
}
