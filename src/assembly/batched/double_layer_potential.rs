//! Batched dense assembly of boundary operators
use super::{
    BatchedPotentialAssembler, BatchedPotentialAssemblerOptions, GreenKernelEvalType, RlstArray,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a Laplace double layer potential operator
pub struct LaplaceDoubleLayerPotentialAssembler<T: RlstScalar + MatrixInverse> {
    kernel: Laplace3dKernel<T>,
    options: BatchedPotentialAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse> Default for LaplaceDoubleLayerPotentialAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
            options: BatchedPotentialAssemblerOptions::default(),
        }
    }
}

impl<T: RlstScalar + MatrixInverse> BatchedPotentialAssembler
    for LaplaceDoubleLayerPotentialAssembler<T>
{
    const DERIV_SIZE: usize = 4;
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
        -*k.get_unchecked([1, index, point_index])
            * num::cast::<T::Real, T>(*normals.get_unchecked([0, index])).unwrap()
            - *k.get_unchecked([2, index, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([1, index])).unwrap()
            - *k.get_unchecked([3, index, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([2, index])).unwrap()
    }

    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(GreenKernelEvalType::ValueDeriv, sources, targets, result);
    }
}

/// Assembler for a Helmholtz double layer potential operator
pub struct HelmholtzDoubleLayerPotentialAssembler<T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: Helmholtz3dKernel<T>,
    options: BatchedPotentialAssemblerOptions,
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HelmholtzDoubleLayerPotentialAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            options: BatchedPotentialAssemblerOptions::default(),
        }
    }
}

impl<T: RlstScalar<Complex = T> + MatrixInverse> BatchedPotentialAssembler
    for HelmholtzDoubleLayerPotentialAssembler<T>
{
    const DERIV_SIZE: usize = 4;
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
        -*k.get_unchecked([1, index, point_index])
            * num::cast::<T::Real, T>(*normals.get_unchecked([0, index])).unwrap()
            - *k.get_unchecked([2, index, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([1, index])).unwrap()
            - *k.get_unchecked([3, index, point_index])
                * num::cast::<T::Real, T>(*normals.get_unchecked([2, index])).unwrap()
    }

    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(GreenKernelEvalType::ValueDeriv, sources, targets, result);
    }
}
