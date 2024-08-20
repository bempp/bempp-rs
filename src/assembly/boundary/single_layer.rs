//! Single layer assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::{
    boundary::integrands::SingleLayerBoundaryIntegrand,
    common::{GreenKernelEvalType, RlstArray},
    kernels::KernelEvaluator,
};
use crate::traits::KernelEvaluator as KernelEvaluatorTrait;
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{MatrixInverse, RlstScalar, UnsafeRandomAccessByRef};

/// Assembler for a single layer operator
pub struct SingleLayerAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    integrand: SingleLayerBoundaryIntegrand<T>,
    kernel: KernelEvaluator<T, K>,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> SingleLayerAssembler<T, K> {
    /// Create a new single layer assembler
    pub fn new(kernel: KernelEvaluator<T, K>) -> Self {
        Self {
            integrand: SingleLayerBoundaryIntegrand::new(),
            kernel,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> SingleLayerAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace single layer assembler
    pub fn new_laplace() -> Self {
        Self::new(KernelEvaluator::new_laplace(GreenKernelEvalType::Value))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> SingleLayerAssembler<T, Helmholtz3dKernel<T>> {
    /// Create a new Helmholtz single layer assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::Value,
        ))
    }
}

impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> BoundaryAssembler
    for SingleLayerAssembler<T, K>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    type Integrand = SingleLayerBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, K>;
    fn integrand(&self) -> &SingleLayerBoundaryIntegrand<T> {
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
        *k.get_unchecked([0, test_index, trial_index])
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
