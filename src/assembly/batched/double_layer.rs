//! Double layer assemblers
use super::{BatchedAssembler, BatchedAssemblerOptions, GreenKernelEvalType, RlstArray};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a Laplace double layer operator
pub struct LaplaceDoubleLayerAssembler<T: RlstScalar + MatrixInverse> {
    kernel: Laplace3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse> Default for LaplaceDoubleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> BatchedAssembler for LaplaceDoubleLayerAssembler<T> {
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
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([0, index])).unwrap()
            + *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([1, index])).unwrap()
            + *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([2, index])).unwrap()
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
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([0, trial_index])).unwrap()
            + *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([1, trial_index])).unwrap()
            + *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([2, trial_index])).unwrap()
    }
    fn kernel_assemble_pairwise_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_pairwise_st(GreenKernelEvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(GreenKernelEvalType::ValueDeriv, sources, targets, result);
    }
}

/// Assembler for a Helmholtz double layer boundary operator
pub struct HelmholtzDoubleLayerAssembler<T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: Helmholtz3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HelmholtzDoubleLayerAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> BatchedAssembler
    for HelmholtzDoubleLayerAssembler<T>
{
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
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([0, index])).unwrap()
            + *k.get_unchecked([2, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([1, index])).unwrap()
            + *k.get_unchecked([3, index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([2, index])).unwrap()
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
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([0, trial_index])).unwrap()
            + *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([1, trial_index])).unwrap()
            + *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([2, trial_index])).unwrap()
    }
    fn kernel_assemble_pairwise_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_pairwise_st(GreenKernelEvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(GreenKernelEvalType::ValueDeriv, sources, targets, result);
    }
}
