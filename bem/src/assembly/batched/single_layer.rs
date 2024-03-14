//! Single layer assemblers
use super::{BatchedAssembler, EvalType, RlstArray};
use bempp_kernel::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use bempp_traits::kernel::Kernel;
use rlst_dense::{traits::UnsafeRandomAccessByRef, types::RlstScalar};

/// Assembler for a Laplace single layer operator
pub struct LaplaceSingleLayerAssembler<T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<T: RlstScalar> Default for LaplaceSingleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
impl<T: RlstScalar> BatchedAssembler for LaplaceSingleLayerAssembler<T> {
    const DERIV_SIZE: usize = 1;
    type RealT = T::Real;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        _trial_normals: &RlstArray<Self::RealT, 2>,
        index: usize,
    ) -> Self::T {
        *k.get_unchecked([0, index])
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        _trial_normals: &RlstArray<Self::RealT, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T {
        *k.get_unchecked([test_index, 0, trial_index])
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
    }
}

/// Assembler for a Helmholtz single layer boundary operator
pub struct HelmholtzSingleLayerAssembler<T: RlstScalar<Complex = T>> {
    kernel: Helmholtz3dKernel<T>,
}
impl<T: RlstScalar<Complex = T>> HelmholtzSingleLayerAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
        }
    }
}
impl<T: RlstScalar<Complex = T>> BatchedAssembler for HelmholtzSingleLayerAssembler<T> {
    const DERIV_SIZE: usize = 1;
    type RealT = T::Real;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        _trial_normals: &RlstArray<Self::RealT, 2>,
        index: usize,
    ) -> Self::T {
        *k.get_unchecked([0, index])
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        _trial_normals: &RlstArray<Self::RealT, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T {
        *k.get_unchecked([test_index, 0, trial_index])
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
    }
}
