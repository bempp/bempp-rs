//! Single layer assemblers
use super::{BatchedAssembler, BatchedAssemblerOptions, EvalType, RlstArray};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a Laplace single layer operator
pub struct LaplaceSingleLayerAssembler<T: RlstScalar + MatrixInverse> {
    kernel: Laplace3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse> Default for LaplaceSingleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> BatchedAssembler for LaplaceSingleLayerAssembler<T> {
    const DERIV_SIZE: usize = 1;
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
    fn kernel_assemble_pairwise_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_pairwise_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
    }
}

/// Assembler for a Helmholtz single layer boundary operator
pub struct HelmholtzSingleLayerAssembler<T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: Helmholtz3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HelmholtzSingleLayerAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> BatchedAssembler
    for HelmholtzSingleLayerAssembler<T>
{
    const DERIV_SIZE: usize = 1;
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
    fn kernel_assemble_pairwise_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_pairwise_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
    }
}
