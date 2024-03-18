//! Adjoint double layer assemblers
use super::{BatchedAssembler, EvalType, RlstArray};
use bempp_kernel::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use bempp_traits::kernel::Kernel;
use rlst_dense::{traits::UnsafeRandomAccessByRef, types::RlstScalar};

/// Assembler for a Laplace adjoint double layer operator
pub struct LaplaceAdjointDoubleLayerAssembler<const BATCHSIZE: usize, T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<const BATCHSIZE: usize, T: RlstScalar> Default
    for LaplaceAdjointDoubleLayerAssembler<BATCHSIZE, T>
{
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
impl<const BATCHSIZE: usize, T: RlstScalar> BatchedAssembler
    for LaplaceAdjointDoubleLayerAssembler<BATCHSIZE, T>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    const BATCHSIZE: usize = BATCHSIZE;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        -*k.get_unchecked([1, index])
            * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 0])).unwrap()
            - *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 1])).unwrap()
            - *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 2])).unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        -*k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 0])).unwrap()
            - *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 1])).unwrap()
            - *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 2])).unwrap()
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}

/// Assembler for a Helmholtz adjoint double layer boundary operator
pub struct HelmholtzAdjointDoubleLayerAssembler<const BATCHSIZE: usize, T: RlstScalar<Complex = T>>
{
    kernel: Helmholtz3dKernel<T>,
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>>
    HelmholtzAdjointDoubleLayerAssembler<BATCHSIZE, T>
{
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
        }
    }
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>> BatchedAssembler
    for HelmholtzAdjointDoubleLayerAssembler<BATCHSIZE, T>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    const BATCHSIZE: usize = BATCHSIZE;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        -*k.get_unchecked([1, index])
            * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 0])).unwrap()
            - *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 1])).unwrap()
            - *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 2])).unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        -*k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 0])).unwrap()
            - *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 1])).unwrap()
            - *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 2])).unwrap()
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}
