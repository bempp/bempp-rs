//! Double layer potential assemblers
use super::{PotentialAssembler, PotentialAssemblerOptions};
use crate::assembly::common::{GreenKernelEvalType, RlstArray};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a Laplace single layer potential operator
pub struct LaplaceSingleLayerPotentialAssembler<T: RlstScalar + MatrixInverse> {
    kernel: Laplace3dKernel<T>,
    options: PotentialAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse> Default for LaplaceSingleLayerPotentialAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
            options: PotentialAssemblerOptions::default(),
        }
    }
}

impl<T: RlstScalar + MatrixInverse> PotentialAssembler for LaplaceSingleLayerPotentialAssembler<T> {
    const DERIV_SIZE: usize = 1;
    type T = T;

    fn options(&self) -> &PotentialAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut PotentialAssemblerOptions {
        &mut self.options
    }

    unsafe fn kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        _normals: &RlstArray<T::Real, 2>,
        index: usize,
        point_index: usize,
    ) -> T {
        *k.get_unchecked([0, index, point_index])
    }

    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(GreenKernelEvalType::Value, sources, targets, result);
    }
}

/// Assembler for a Helmholtz single layer potential operator
pub struct HelmholtzSingleLayerPotentialAssembler<T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: Helmholtz3dKernel<T>,
    options: PotentialAssemblerOptions,
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HelmholtzSingleLayerPotentialAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            options: PotentialAssemblerOptions::default(),
        }
    }
}

impl<T: RlstScalar<Complex = T> + MatrixInverse> PotentialAssembler
    for HelmholtzSingleLayerPotentialAssembler<T>
{
    const DERIV_SIZE: usize = 1;
    type T = T;

    fn options(&self) -> &PotentialAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut PotentialAssemblerOptions {
        &mut self.options
    }

    unsafe fn kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        _normals: &RlstArray<T::Real, 2>,
        index: usize,
        point_index: usize,
    ) -> T {
        *k.get_unchecked([0, index, point_index])
    }

    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    ) {
        self.kernel
            .assemble_st(GreenKernelEvalType::Value, sources, targets, result);
    }
}
