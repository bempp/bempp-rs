//! Hypersingular assemblers
use super::{BoundaryAssembler, BoundaryAssemblerOptions};
use crate::assembly::{
    boundary::integrands::HypersingularBoundaryIntegrand,
    common::{equal_grids, GreenKernelEvalType, RlstArray, SparseMatrixData},
    kernels::KernelEvaluator,
};
use crate::traits::FunctionSpace;
use crate::traits::KernelEvaluator as KernelEvaluatorTrait;
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use ndelement::traits::FiniteElement;
use ndelement::types::ReferenceCellType;
use ndgrid::traits::Grid;
use rlst::{MatrixInverse, RlstScalar, Shape, UnsafeRandomAccessByRef};
use std::collections::HashMap;

#[allow(clippy::too_many_arguments)]
unsafe fn hyp_test_trial_product<T: RlstScalar + MatrixInverse>(
    test_table: &RlstArray<T, 4>,
    trial_table: &RlstArray<T, 4>,
    test_jacobians: &RlstArray<T::Real, 2>,
    trial_jacobians: &RlstArray<T::Real, 2>,
    test_jdets: &[T::Real],
    trial_jdets: &[T::Real],
    test_point_index: usize,
    trial_point_index: usize,
    test_basis_index: usize,
    trial_basis_index: usize,
) -> T {
    let test0 = *test_table.get_unchecked([1, test_point_index, test_basis_index, 0]);
    let test1 = *test_table.get_unchecked([2, test_point_index, test_basis_index, 0]);
    let trial0 = *trial_table.get_unchecked([1, trial_point_index, trial_basis_index, 0]);
    let trial1 = *trial_table.get_unchecked([2, trial_point_index, trial_basis_index, 0]);

    ((num::cast::<T::Real, T>(*test_jacobians.get_unchecked([3, test_point_index])).unwrap()
        * test0
        - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([0, test_point_index])).unwrap()
            * test1)
        * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([3, trial_point_index]))
            .unwrap()
            * trial0
            - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([0, trial_point_index]))
                .unwrap()
                * trial1)
        + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([4, test_point_index])).unwrap()
            * test0
            - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([1, test_point_index]))
                .unwrap()
                * test1)
            * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([4, trial_point_index]))
                .unwrap()
                * trial0
                - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([1, trial_point_index]))
                    .unwrap()
                    * trial1)
        + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([5, test_point_index])).unwrap()
            * test0
            - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([2, test_point_index]))
                .unwrap()
                * test1)
            * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([5, trial_point_index]))
                .unwrap()
                * trial0
                - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([2, trial_point_index]))
                    .unwrap()
                    * trial1))
        / num::cast::<T::Real, T>(test_jdets[test_point_index] * trial_jdets[trial_point_index])
            .unwrap()
}

/// Assembler for a hypersingular operator
pub struct HypersingularAssembler<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> {
    kernel: KernelEvaluator<T, K>,
    options: BoundaryAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse, K: Kernel<T = T>> HypersingularAssembler<T, K> {
    /// Create a new hypersingular assembler
    pub fn new(kernel: KernelEvaluator<T, K>) -> Self {
        Self {
            kernel,
            options: BoundaryAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> HypersingularAssembler<T, Laplace3dKernel<T>> {
    /// Create a new Laplace hypersingular assembler
    pub fn new_laplace() -> Self {
        Self::new(KernelEvaluator::new_laplace(
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HypersingularAssembler<T, Helmholtz3dKernel<T>> {
    /// Create a new Helmholtz hypersingular assembler
    pub fn new_helmholtz(wavenumber: T::Real) -> Self {
        Self::new(KernelEvaluator::new_helmholtz(
            wavenumber,
            GreenKernelEvalType::ValueDeriv,
        ))
    }
}

impl<T: RlstScalar + MatrixInverse> BoundaryAssembler
    for HypersingularAssembler<T, Laplace3dKernel<T>>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Laplace3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Laplace3dKernel<T>> {
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
    #[allow(clippy::too_many_arguments)]
    unsafe fn test_trial_product(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        test_jacobians: &RlstArray<T::Real, 2>,
        trial_jacobians: &RlstArray<T::Real, 2>,
        test_jdets: &[T::Real],
        trial_jdets: &[T::Real],
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
    ) -> T {
        hyp_test_trial_product::<T>(
            test_table,
            trial_table,
            test_jacobians,
            trial_jacobians,
            test_jdets,
            trial_jdets,
            test_point_index,
            trial_point_index,
            test_basis_index,
            trial_basis_index,
        )
    }
}

/// Assembler for curl-curl term of Helmholtz hypersingular operator
struct HelmholtzHypersingularCurlCurlAssembler<'a, T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
    options: &'a BoundaryAssemblerOptions,
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse>
    HelmholtzHypersingularCurlCurlAssembler<'a, T>
{
    /// Create a new assembler
    pub fn new(
        kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
        options: &'a BoundaryAssemblerOptions,
    ) -> Self {
        Self { kernel, options }
    }
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse> BoundaryAssembler
    for HelmholtzHypersingularCurlCurlAssembler<'a, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Helmholtz3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Helmholtz3dKernel<T>> {
        panic!();
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        panic!("Cannot get mutable options")
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
    #[allow(clippy::too_many_arguments)]
    unsafe fn test_trial_product(
        &self,
        test_table: &RlstArray<T, 4>,
        trial_table: &RlstArray<T, 4>,
        test_jacobians: &RlstArray<T::Real, 2>,
        trial_jacobians: &RlstArray<T::Real, 2>,
        test_jdets: &[T::Real],
        trial_jdets: &[T::Real],
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
    ) -> T {
        hyp_test_trial_product::<T>(
            test_table,
            trial_table,
            test_jacobians,
            trial_jacobians,
            test_jdets,
            trial_jdets,
            test_point_index,
            trial_point_index,
            test_basis_index,
            trial_basis_index,
        )
    }
}

/// Assembler for normal normal term of Helmholtz hypersingular boundary operator
struct HelmholtzHypersingularNormalNormalAssembler<'a, T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
    options: &'a BoundaryAssemblerOptions,
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse>
    HelmholtzHypersingularNormalNormalAssembler<'a, T>
{
    /// Create a new assembler
    pub fn new(
        kernel: &'a KernelEvaluator<T, Helmholtz3dKernel<T>>,
        options: &'a BoundaryAssemblerOptions,
    ) -> Self {
        Self { kernel, options }
    }
}
impl<'a, T: RlstScalar<Complex = T> + MatrixInverse> BoundaryAssembler
    for HelmholtzHypersingularNormalNormalAssembler<'a, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Helmholtz3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Helmholtz3dKernel<T>> {
        panic!();
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        panic!("Cannot get mutable options")
    }
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        -*k.get_unchecked([0, index])
            * num::cast::<T::Real, T>(
                self.kernel.kernel.wavenumber.powi(2)
                    * (*trial_normals.get_unchecked([0, index])
                        * *test_normals.get_unchecked([0, index])
                        + *trial_normals.get_unchecked([1, index])
                            * *test_normals.get_unchecked([1, index])
                        + *trial_normals.get_unchecked([2, index])
                            * *test_normals.get_unchecked([2, index])),
            )
            .unwrap()
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        -num::cast::<T::Real, T>(self.kernel.kernel.wavenumber.powi(2)).unwrap()
            * *k.get_unchecked([0, test_index, trial_index])
            * (num::cast::<T::Real, T>(*trial_normals.get_unchecked([0, trial_index])).unwrap()
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([0, test_index])).unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([1, trial_index])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([1, test_index]))
                        .unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([2, trial_index])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([2, test_index]))
                        .unwrap())
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

impl<T: RlstScalar<Complex = T> + MatrixInverse> BoundaryAssembler
    for HypersingularAssembler<T, Helmholtz3dKernel<T>>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    type Integrand = HypersingularBoundaryIntegrand<T>;
    type Kernel = KernelEvaluator<T, Helmholtz3dKernel<T>>;
    fn integrand(&self) -> &HypersingularBoundaryIntegrand<T> {
        panic!();
    }
    fn kernel(&self) -> &KernelEvaluator<T, Helmholtz3dKernel<T>> {
        panic!();
    }
    fn options(&self) -> &BoundaryAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BoundaryAssemblerOptions {
        &mut self.options
    }

    unsafe fn singular_kernel_value(
        &self,
        _k: &RlstArray<T, 2>,
        _test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        _index: usize,
    ) -> T {
        panic!("Cannot directly use HypersingularAssembler for Helmholtz");
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        _k: &RlstArray<T, 3>,
        _test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        _test_index: usize,
        _trial_index: usize,
    ) -> T {
        panic!("Cannot directly use HypersingularAssembler for Helmholtz");
    }
    fn kernel_assemble_pairwise_st(
        &self,
        _sources: &[T::Real],
        _targets: &[T::Real],
        _result: &mut [T],
    ) {
        panic!("Cannot directly use HypersingularAssembler for Helmholtz");
    }
    fn kernel_assemble_st(&self, _sources: &[T::Real], _targets: &[T::Real], _result: &mut [T]) {
        panic!("Cannot directly use HypersingularAssembler for Helmholtz");
    }

    fn assemble_singular<
        TestGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = T> + Sync,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> SparseMatrixData<T> {
        let curl_curl_assembler =
            HelmholtzHypersingularCurlCurlAssembler::<T>::new(&self.kernel, &self.options);
        let normal_normal_assembler =
            HelmholtzHypersingularNormalNormalAssembler::<T>::new(&self.kernel, &self.options);

        let mut curlcurl = curl_curl_assembler.assemble_singular::<TestGrid, TrialGrid, Element>(
            shape,
            trial_space,
            test_space,
        );
        curlcurl.add(
            normal_normal_assembler.assemble_singular::<TestGrid, TrialGrid, Element>(
                shape,
                trial_space,
                test_space,
            ),
        );
        curlcurl
    }

    fn assemble_singular_correction<
        TestGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = T> + Sync,
    >(
        &self,
        shape: [usize; 2],
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> SparseMatrixData<T> {
        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return SparseMatrixData::new(shape);
        }

        let curl_curl_assembler =
            HelmholtzHypersingularCurlCurlAssembler::<T>::new(&self.kernel, &self.options);
        let normal_normal_assembler =
            HelmholtzHypersingularNormalNormalAssembler::<T>::new(&self.kernel, &self.options);

        let mut curlcurl = curl_curl_assembler
            .assemble_singular_correction::<TestGrid, TrialGrid, Element>(
                shape,
                trial_space,
                test_space,
            );
        curlcurl.add(
            normal_normal_assembler.assemble_singular_correction::<TestGrid, TrialGrid, Element>(
                shape,
                trial_space,
                test_space,
            ),
        );
        curlcurl
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_nonsingular_into_dense<
        TestGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
        TrialGrid: Grid<T = T::Real, EntityDescriptor = ReferenceCellType> + Sync,
        Element: FiniteElement<T = T> + Sync,
    >(
        &self,
        output: &mut RlstArray<T, 2>,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
        trial_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
        test_colouring: &HashMap<ReferenceCellType, Vec<Vec<usize>>>,
    ) {
        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if output.shape()[0] != test_space.global_size()
            || output.shape()[1] != trial_space.global_size()
        {
            panic!("Matrix has wrong shape");
        }

        let curl_curl_assembler =
            HelmholtzHypersingularCurlCurlAssembler::<T>::new(&self.kernel, &self.options);
        let normal_normal_assembler =
            HelmholtzHypersingularNormalNormalAssembler::<T>::new(&self.kernel, &self.options);

        curl_curl_assembler.assemble_nonsingular_into_dense::<TestGrid, TrialGrid, Element>(
            output,
            trial_space,
            test_space,
            trial_colouring,
            test_colouring,
        );
        normal_normal_assembler.assemble_nonsingular_into_dense::<TestGrid, TrialGrid, Element>(
            output,
            trial_space,
            test_space,
            trial_colouring,
            test_colouring,
        );
    }
}
