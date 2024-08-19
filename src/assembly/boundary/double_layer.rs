//! Double layer assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::common::{GreenKernelEvalType, RlstArray};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a double layer operator
pub struct DoubleLayerAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    kernel: K,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> DoubleLayerAssembler<T, K> {
    /// Create a new double layer assembler
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> DoubleLayerAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace double layer assembler
    pub fn new_laplace() -> Self {
        Self::new(Laplace3dKernel::<T>::new())
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> DoubleLayerAssembler<T, Helmholtz3dKernel<T>> {
    /// Create a new Helmholtz double layer assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(Helmholtz3dKernel::<T>::new(wavenumber))
    }
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> BoundaryAssembler
    for DoubleLayerAssembler<T, K>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    fn options(&self) -> &BoundaryAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
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
        *k.get_unchecked([1, test_index, trial_index])
            * num::cast::<T::Real, T>(*trial_normals.get_unchecked([0, trial_index])).unwrap()
            + *k.get_unchecked([2, test_index, trial_index])
                * num::cast::<T::Real, T>(*trial_normals.get_unchecked([1, trial_index])).unwrap()
            + *k.get_unchecked([3, test_index, trial_index])
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
