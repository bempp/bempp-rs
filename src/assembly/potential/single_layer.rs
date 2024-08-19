//! Single layer potential assemblers
use super::{PotentialAssembler, PotentialAssemblerOptions};
use crate::assembly::common::{GreenKernelEvalType, RlstArray};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a single layer potential operator
pub struct SingleLayerPotentialAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    kernel: K,
    options: PotentialAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> SingleLayerPotentialAssembler<T, K> {
    /// Create a new single layer potential assembler
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            options: PotentialAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> SingleLayerPotentialAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace single layer potential assembler
    pub fn new_laplace() -> Self {
        Self::new(Laplace3dKernel::<T>::new())
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse>
    SingleLayerPotentialAssembler<T, Helmholtz3dKernel<T>>
{
    /// Create a new Helmholtz single layer potential assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(Helmholtz3dKernel::<T>::new(wavenumber))
    }
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> PotentialAssembler
    for SingleLayerPotentialAssembler<T, K>
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
