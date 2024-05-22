//! Batched dense assembly of boundary operators
use super::{BatchedPotentialAssembler, BatchedPotentialAssemblerOptions, EvalType, RlstArray};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a Laplace double layer potential operator
pub struct LaplaceDoubleLayerPotentialAssembler<T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
    options: BatchedPotentialAssemblerOptions,
}
impl<T: RlstScalar> Default for LaplaceDoubleLayerPotentialAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
            options: BatchedPotentialAssemblerOptions::default(),
        }
    }
}

impl<T: RlstScalar> BatchedPotentialAssembler for LaplaceDoubleLayerPotentialAssembler<T> {
    const DERIV_SIZE: usize = 4;
    // TODO: remove TABLE_DERIVS, always 0
    const TABLE_DERIVS: usize = 0;
    type T = T;

    fn options(&self) -> &BatchedPotentialAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BatchedPotentialAssemblerOptions {
        &mut self.options
    }

    unsafe fn kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        normals: &RlstArray<T::Real, 2>,
        index: usize,
        point_index: usize,
    ) -> T {
        -*k.get_unchecked([index, 1, point_index])
            * num::cast::<T::Real, T>(*normals.get_unchecked([index, 0])).unwrap()
            - *k.get_unchecked([index, 2, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([index, 1])).unwrap()
            - *k.get_unchecked([index, 3, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([index, 2])).unwrap()
    }

    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}

/// Assembler for a Helmholtz double layer potential operator
pub struct HelmholtzDoubleLayerPotentialAssembler<T: RlstScalar<Complex = T>> {
    kernel: Helmholtz3dKernel<T>,
    options: BatchedPotentialAssemblerOptions,
}
impl<T: RlstScalar<Complex = T>> HelmholtzDoubleLayerPotentialAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            options: BatchedPotentialAssemblerOptions::default(),
        }
    }
}

impl<T: RlstScalar<Complex = T>> BatchedPotentialAssembler
    for HelmholtzDoubleLayerPotentialAssembler<T>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    type T = T;

    fn options(&self) -> &BatchedPotentialAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BatchedPotentialAssemblerOptions {
        &mut self.options
    }

    unsafe fn kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        normals: &RlstArray<T::Real, 2>,
        index: usize,
        point_index: usize,
    ) -> T {
        -*k.get_unchecked([index, 1, point_index])
            * num::cast::<T::Real, T>(*normals.get_unchecked([index, 0])).unwrap()
            - *k.get_unchecked([index, 2, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([index, 1])).unwrap()
            - *k.get_unchecked([index, 3, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([index, 2])).unwrap()
    }

    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(EvalType::ValueDeriv, sources, targets, result);
    }
}
