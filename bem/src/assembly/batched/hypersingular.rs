//! Hypersingular assemblers
use super::{BatchedAssembler, EvalType, RlstArray};
use bempp_kernel::laplace_3d::Laplace3dKernel;
use bempp_traits::kernel::Kernel;
use rlst_dense::{traits::UnsafeRandomAccessByRef, types::RlstScalar};

/// Assembler for a Laplace hypersingular operator
pub struct LaplaceHypersingularAssembler<const BATCHSIZE: usize, T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<const BATCHSIZE: usize, T: RlstScalar> Default
    for LaplaceHypersingularAssembler<BATCHSIZE, T>
{
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
impl<const BATCHSIZE: usize, T: RlstScalar> BatchedAssembler
    for LaplaceHypersingularAssembler<BATCHSIZE, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    const BATCHSIZE: usize = BATCHSIZE;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        _test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        *k.get_unchecked([0, index])
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        _test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        *k.get_unchecked([test_index, 0, trial_index])
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn test_trial_product(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        test_jacobians: &RlstArray<T::Real, 2>,
        trial_jacobians: &RlstArray<T::Real, 2>,
        test_jdets: &[T::Real],
        trial_jdets: &[T::Real],
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
    ) -> T {
        let test0 = *test_table.get_unchecked([1, test_point_index, test_basis_index, 0]);
        let test1 = *test_table.get_unchecked([2, test_point_index, test_basis_index, 0]);
        let trial0 = *trial_table.get_unchecked([1, trial_point_index, trial_basis_index, 0]);
        let trial1 = *trial_table.get_unchecked([2, trial_point_index, trial_basis_index, 0]);

        ((num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 3])).unwrap()
            * test0
            - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 0]))
                .unwrap()
                * test1)
            * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 3]))
                .unwrap()
                * trial0
                - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 0]))
                    .unwrap()
                    * trial1)
            + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 4]))
                .unwrap()
                * test0
                - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 1]))
                    .unwrap()
                    * test1)
                * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 4]))
                    .unwrap()
                    * trial0
                    - num::cast::<T::Real, T>(
                        *trial_jacobians.get_unchecked([trial_point_index, 1]),
                    )
                    .unwrap()
                        * trial1)
            + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 5]))
                .unwrap()
                * test0
                - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 2]))
                    .unwrap()
                    * test1)
                * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 5]))
                    .unwrap()
                    * trial0
                    - num::cast::<T::Real, T>(
                        *trial_jacobians.get_unchecked([trial_point_index, 2]),
                    )
                    .unwrap()
                        * trial1))
            / num::cast::<T::Real, T>(test_jdets[test_point_index] * trial_jdets[trial_point_index])
                .unwrap()
    }
}
