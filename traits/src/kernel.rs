//! Trait for Green's function kernels
use crate::types::EvalType;
use crate::types::KernelType;
use crate::types::Scalar;

/// Interface to evaluating Green's functions for given sources and targets.
pub trait Kernel: Sync {
    type T: Scalar;

    /// Single threaded evaluation of Green's functions.
    ///
    /// - `eval_type`: Either [EvalType::Value] to only return Green's function values
    ///              or [EvalType::ValueDeriv] to return values and derivatives.
    /// - `sources`: A slice defining the source points. The points must be given in the form
    ///            `[x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N]`, that is
    ///            the value for each dimension must be continuously contained in the slice.
    /// - `targets`: A slice defining the targets. The memory layout is the same as for sources.
    /// - `charges`: A slice defining the charges. For each source point there needs to be one charge.
    /// - `result`: The result array. If the kernel is scalar and `eval_type` has the value [EvalType::Value]
    ///           then `result` has the same number of elemens as there are targets. For a scalar kernel
    ///           in three dimensional space if [EvalType::ValueDeriv] was chosen then `result` contains
    ///           for each target in consecutive order the value of the kernel and the three components
    ///           of its derivative.
    ///
    fn evaluate_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    );

    /// Multi-threaded evaluation of a Green's function kernel.
    ///
    /// The method parallelizes over the given targets. It expects a Rayon `ThreadPool`
    /// in which the multi-threaded execution can be scheduled.
    fn evaluate_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        charges: &[Self::T],
        result: &mut [Self::T],
    );

    /// Single threaded assembly of a kernel matrix.
    ///
    /// - `eval_type`: Either [EvalType::Value] to only return Green's function values
    ///              or [EvalType::ValueDeriv] to return values and derivatives.
    /// - `sources`: A slice defining the source points. The points must be given in the form
    ///            `[x_1, x_2, ... x_N, y_1, y_2, ..., y_N, z_1, z_2, ..., z_N]`, that is
    ///            the value for each dimension must be continuously contained in the slice.
    /// - `targets`: A slice defining the targets. The memory layout is the same as for sources.
    /// - `result`: The result array. If the kernel is scalar and `eval_type` has the value [EvalType::Value]
    ///           then `result` has MxN elements with M the number of targets and N the number of targets.
    ///           For a scalar kernel in three dimensional space if [EvalType::ValueDeriv] was chosen then `result` contains
    ///           in consecutive order the interaction of all sources with the first target and then the corresponding derivatives,
    ///           followed by the interactions with the second target, and so on. See the example for illustration.
    ///
    fn assemble_st(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        result: &mut [Self::T],
    );

    /// Multi-threaded version of kernel matrix assembly.
    fn assemble_mt(
        &self,
        eval_type: EvalType,
        sources: &[<Self::T as Scalar>::Real],
        targets: &[<Self::T as Scalar>::Real],
        result: &mut [Self::T],
    );

    /// Return the type of the kernel.
    fn kernel_type(&self) -> &KernelType;

    /// Return the domain component count of the Green's fct.
    ///
    /// For a scalar kernel this is `1`.
    fn domain_component_count(&self) -> usize;

    /// Return the space dimension.
    fn space_dimension(&self) -> usize;

    /// Return the range component count of the Green's fct.
    ///
    /// For a scalar kernel this is `1` if [EvalType::Value] is
    /// given, and `4` if [EvalType::ValueDeriv] is given.
    fn range_component_count(&self, eval_type: EvalType) -> usize;
}

/// Scaling required by the FMM to apply kernel to each octree level.
pub trait KernelScale {
    /// The kernel is generic over data type.
    type T: Scalar<Real = Self::T>;

    /// # Warning
    /// Scaling by level is kernel dependent, only applicable to homogenous kernels, staged to be deprecated
    ///
    /// # Arguments
    /// * `level` - Level of octree at which the kernel is being applied.
    fn scale(&self, level: u64) -> Self::T;
}
