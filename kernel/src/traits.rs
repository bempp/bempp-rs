//! Trait for Green's function kernels
use crate::types::EvalType;
use crate::types::KernelType;
use bempp_traits::types::Scalar;

use rayon::ThreadPool;

pub trait Kernel {
    type T: Scalar;

    fn evaluate_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    );

    fn evaluate_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
        thread_pool: &ThreadPool,
    );

    fn kernel_type(&self) -> &KernelType;

    fn domain_component_count(&self) -> usize;

    fn space_dimension(&self) -> usize;

    fn range_component_count(&self, eval_type: EvalType) -> usize;
}
