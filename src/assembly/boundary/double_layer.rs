//! Double layer assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::{
    boundary::integrands::DoubleLayerBoundaryIntegrand,
    common::{GreenKernelEvalType, RlstArray},
    kernels::KernelEvaluator,
};
use crate::traits::KernelEvaluator as KernelEvaluatorTrait;
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a double layer operator
pub struct DoubleLayerAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    integrand: DoubleLayerBoundaryIntegrand<T>,
    kernel: KernelEvaluator<T, K>,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> DoubleLayerAssembler<T, K> {
    /// Create a new double layer assembler
    pub fn new(kernel: KernelEvaluator<T, K>) -> Self {
        Self {
            integrand: DoubleLayerBoundaryIntegrand::new(),
            kernel,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> DoubleLayerAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace double layer assembler
    pub fn new_laplace() -> Self {
        Self::new(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> DoubleLayerAssembler<T, Helmholtz3dKernel<T>> {
    /// Create a new Helmholtz double layer assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> BoundaryAssembler
    for DoubleLayerAssembler<T, K>
{
    const DERIV_SIZE: usize = 4;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    type Integrand = DoubleLayerBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, K>;
    fn integrand(&self) -> &DoubleLayerBoundaryIntegrand<T> {
        &self.integrand
    }
    fn kernel(&self) -> &KernelEvaluator<T, K> {
        &self.kernel
    }
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
        self.kernel.assemble_pairwise_st(sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel.assemble_st(sources, targets, result);
    }
}
