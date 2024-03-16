//! Double layer assemblers
use super::{BatchedAssembler, EvalType, RlstArray};
use bempp_kernel::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel};
use bempp_traits::kernel::Kernel;
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a Laplace double layer operator
pub struct LaplaceDoubleLayerAssembler<T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<T: RlstScalar> Default for LaplaceDoubleLayerAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
impl<T: RlstScalar> BatchedAssembler for LaplaceDoubleLayerAssembler<T> {
    const DERIV_SIZE: usize = 4;
    type RealT = T::Real;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        trial_normals: &RlstArray<Self::RealT, 2>,
        index: usize,
    ) -> Self::T {
        *k.get_unchecked([1, index])
            * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([index, 0])).unwrap()
            + *k.get_unchecked([2, index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([index, 1]))
                    .unwrap()
            + *k.get_unchecked([3, index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([index, 2]))
                    .unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        trial_normals: &RlstArray<Self::RealT, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T {
        *k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([trial_index, 0]))
                .unwrap()
            + *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([trial_index, 1]))
                    .unwrap()
            + *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([trial_index, 2]))
                    .unwrap()
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}

/// Assembler for a Helmholtz double layer boundary operator
pub struct HelmholtzDoubleLayerAssembler<T: RlstScalar<Complex = T>> {
    kernel: Helmholtz3dKernel<T>,
}
impl<T: RlstScalar<Complex = T>> HelmholtzDoubleLayerAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
        }
    }
}
impl<T: RlstScalar<Complex = T>> BatchedAssembler for HelmholtzDoubleLayerAssembler<T> {
    const DERIV_SIZE: usize = 4;
    type RealT = T::Real;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        trial_normals: &RlstArray<Self::RealT, 2>,
        index: usize,
    ) -> Self::T {
        *k.get_unchecked([1, index])
            * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([index, 0])).unwrap()
            + *k.get_unchecked([2, index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([index, 1]))
                    .unwrap()
            + *k.get_unchecked([3, index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([index, 2]))
                    .unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        _test_normals: &RlstArray<Self::RealT, 2>,
        trial_normals: &RlstArray<Self::RealT, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T {
        *k.get_unchecked([test_index, 1, trial_index])
            * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([trial_index, 0]))
                .unwrap()
            + *k.get_unchecked([test_index, 2, trial_index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([trial_index, 1]))
                    .unwrap()
            + *k.get_unchecked([test_index, 3, trial_index])
                * num::cast::<Self::RealT, Self::T>(*trial_normals.get_unchecked([trial_index, 2]))
                    .unwrap()
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::ValueDeriv, sources, targets, result);
    }
    fn kernel_assemble_st(
        &self,
        sources: &[Self::RealT],
        targets: &[Self::RealT],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}
