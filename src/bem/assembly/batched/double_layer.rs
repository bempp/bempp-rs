//! Double layer assemblers
use super::{BatchedAssembler, BatchedAssemblerOptions, EvalType, RlstArray};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a Laplace double layer operator
pub struct LaplaceDoubleLayerAssembler<T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar> Default for LaplaceDoubleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar> BatchedAssembler for LaplaceDoubleLayerAssembler<T> {
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    fn options(&self) -> &BatchedAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BatchedAssemblerOptions {
        &mut self.options
    }
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        _test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        *k.get_unchecked([1, index])
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 0])).unwrap()
            + *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 1])).unwrap()
            + *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 2])).unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        _test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        *k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 0])).unwrap()
            + *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 1])).unwrap()
            + *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 2])).unwrap()
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

/// Assembler for a Helmholtz double layer boundary operator
pub struct HelmholtzDoubleLayerAssembler<T: RlstScalar<Complex = T>> {
    kernel: Helmholtz3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar<Complex = T>> HelmholtzDoubleLayerAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar<Complex = T>> BatchedAssembler for HelmholtzDoubleLayerAssembler<T> {
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    fn options(&self) -> &BatchedAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BatchedAssemblerOptions {
        &mut self.options
    }
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        _test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        *k.get_unchecked([1, index])
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 0])).unwrap()
            + *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 1])).unwrap()
            + *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 2])).unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        _test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        *k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 0])).unwrap()
            + *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 1])).unwrap()
            + *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 2])).unwrap()
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
