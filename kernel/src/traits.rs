//! Trait for Green's function kernels
use crate::types::EvalType;
use crate::types::KernelType;
use bempp_traits::types::Scalar;

/// Interface to evaluating Green's functions for given sources and targets.
pub trait Kernel {
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
    /// The following code gives an example of how to use it together with the [rlst] dense matrix type.
    /// ```
    /// use rlst::dense::*;
    /// use rlst_dense;
    /// use bempp_kernel::traits::*;
    /// use bempp_kernel::laplace_3d::Laplace3dKernel;
    /// use bempp_kernel::types::*;
    /// let nsources = 5;
    /// let ntargets = 10;
    ///
    /// let sources = rlst::dense::rlst_rand_mat![f64, (nsources, 3)];
    /// let targets = rlst::dense::rlst_rand_mat![f64, (ntargets, 3)];
    /// let charges = rlst::dense::rlst_col_vec![f64, nsources];
    /// let mut interactions = rlst_dense::rlst_dynamic_mat!(f64, (4, ntargets));
    ///
    /// Laplace3dKernel::<f64>::new().evaluate_st(EvalType::ValueDeriv, sources.data(), targets.data(), charges.data(), interactions.data_mut());
    ///
    /// println!("The value of the potential at the second target is {}", interactions[[0, 1]]);
    /// println!("The target derivative of the potential at the second target is ({}, {}, {})", interactions[[1, 1]], interactions[[2, 1]], interactions[[3, 1]]);
    ///```
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
    /// The method parallelizes over the given targets. It expects a Rayon [ThreadPool]
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
    /// The following code gives an example of how to use it together with the [rlst] dense matrix type.
    /// ```
    /// use rlst::dense::*;
    /// use rlst_dense;
    /// use bempp_kernel::traits::*;
    /// use bempp_kernel::laplace_3d::Laplace3dKernel;
    /// use bempp_kernel::types::*;
    /// let nsources = 5;
    /// let ntargets = 10;
    ///
    /// let sources = rlst::dense::rlst_rand_mat![f64, (nsources, 3)];
    /// let targets = rlst::dense::rlst_rand_mat![f64, (ntargets, 3)];
    /// let mut interactions = rlst_dense::rlst_dynamic_mat![f64, (nsources, 4 * ntargets)];
    ///
    /// Laplace3dKernel::<f64>::new().assemble_st(EvalType::ValueDeriv, sources.data(), targets.data(), interactions.data_mut());
    ///
    /// // The column index of the third target interaction is 8 = 2 * 4, since each
    /// // target is associated with its interaction value plus 3 derivatives, i.e. 4 values.
    /// // The derivatives correspondingly have the column indices 9, 10, 11.
    /// // If EvalType::Value is chosen then the column index would be 2 as then each target
    /// // is only associated with 1 value.
    /// println!("The interaction of the second source with the third target is {}", interactions[[1, 8]]);
    /// println!("The target derivative of the potential at the second target is ({}, {}, {})", interactions[[1, 9]], interactions[[1, 10]], interactions[[1, 11]]);
    ///```
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
