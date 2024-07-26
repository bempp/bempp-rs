//! Hypersingular assemblers
use super::{BatchedAssembler, BatchedAssemblerOptions, EvalType, RlstArray, SparseMatrixData};
use crate::assembly::common::equal_grids;
use crate::traits::function::FunctionSpace;
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

/// Assembler for a Laplace hypersingular operator
pub struct LaplaceHypersingularAssembler<T: RlstScalar + MatrixInverse> {
    kernel: Laplace3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar + MatrixInverse> Default for LaplaceHypersingularAssembler<T> {
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar + MatrixInverse> BatchedAssembler for LaplaceHypersingularAssembler<T> {
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    fn options(&self) -> &BatchedAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BatchedAssemblerOptions {
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
        *k.get_unchecked([test_index, 0, trial_index])
    }
    fn kernel_assemble_pairwise_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_pairwise_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
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
struct HelmholtzHypersingularCurlCurlAssembler<T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: Helmholtz3dKernel<T>,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HelmholtzHypersingularCurlCurlAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> BatchedAssembler
    for HelmholtzHypersingularCurlCurlAssembler<T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    fn options(&self) -> &BatchedAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BatchedAssemblerOptions {
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
        *k.get_unchecked([test_index, 0, trial_index])
    }
    fn kernel_assemble_pairwise_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_pairwise_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
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
struct HelmholtzHypersingularNormalNormalAssembler<T: RlstScalar<Complex = T> + MatrixInverse> {
    kernel: Helmholtz3dKernel<T>,
    wavenumber: T::Real,
    options: BatchedAssemblerOptions,
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HelmholtzHypersingularNormalNormalAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            wavenumber,
            options: BatchedAssemblerOptions::default(),
        }
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> BatchedAssembler
    for HelmholtzHypersingularNormalNormalAssembler<T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 0;
    type T = T;
    fn options(&self) -> &BatchedAssemblerOptions {
        &self.options
    }
    fn options_mut(&mut self) -> &mut BatchedAssemblerOptions {
        &mut self.options
    }
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        -num::cast::<T::Real, T>(self.wavenumber.powi(2)).unwrap()
            * *k.get_unchecked([0, index])
            * (num::cast::<T::Real, T>(*trial_normals.get_unchecked([0, index])).unwrap()
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([0, index])).unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([1, index])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([1, index])).unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([2, index])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([2, index])).unwrap())
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<T, 3>,
        test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> T {
        -num::cast::<T::Real, T>(self.wavenumber.powi(2)).unwrap()
            * *k.get_unchecked([test_index, 0, trial_index])
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
        self.kernel
            .assemble_pairwise_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
    }
}

/// Assembler for curl-curl term of Helmholtz hypersingular operator
pub struct HelmholtzHypersingularAssembler<T: RlstScalar<Complex = T> + MatrixInverse> {
    curl_curl_assembler: HelmholtzHypersingularCurlCurlAssembler<T>,
    normal_normal_assembler: HelmholtzHypersingularNormalNormalAssembler<T>,
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> HelmholtzHypersingularAssembler<T> {
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            curl_curl_assembler: HelmholtzHypersingularCurlCurlAssembler::<T>::new(wavenumber),
            normal_normal_assembler: HelmholtzHypersingularNormalNormalAssembler::<T>::new(
                wavenumber,
            ),
        }
    }
}
impl<T: RlstScalar<Complex = T> + MatrixInverse> BatchedAssembler
    for HelmholtzHypersingularAssembler<T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    type T = T;
    fn options(&self) -> &BatchedAssemblerOptions {
        panic!("Cannot directly use HelmholtzHypersingularAssembler");
    }
    fn options_mut(&mut self) -> &mut BatchedAssemblerOptions {
        panic!("Cannot directly use HelmholtzHypersingularAssembler");
    }
    fn quadrature_degree(&mut self, cell: ReferenceCellType, degree: usize) {
        self.curl_curl_assembler.quadrature_degree(cell, degree);
        self.normal_normal_assembler.quadrature_degree(cell, degree);
    }
    fn singular_quadrature_degree(
        &mut self,
        cells: (ReferenceCellType, ReferenceCellType),
        degree: usize,
    ) {
        self.curl_curl_assembler
            .singular_quadrature_degree(cells, degree);
        self.normal_normal_assembler
            .singular_quadrature_degree(cells, degree);
    }
    fn batch_size(&mut self, size: usize) {
        self.curl_curl_assembler.batch_size(size);
        self.normal_normal_assembler.batch_size(size);
    }

    unsafe fn singular_kernel_value(
        &self,
        _k: &RlstArray<T, 2>,
        _test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        _index: usize,
    ) -> T {
        panic!("Cannot directly use HelmholtzHypersingularAssembler");
    }
    unsafe fn nonsingular_kernel_value(
        &self,
        _k: &RlstArray<T, 3>,
        _test_normals: &RlstArray<T::Real, 2>,
        _trial_normals: &RlstArray<T::Real, 2>,
        _test_index: usize,
        _trial_index: usize,
    ) -> T {
        panic!("Cannot directly use HelmholtzHypersingularAssembler");
    }
    fn kernel_assemble_pairwise_st(
        &self,
        _sources: &[T::Real],
        _targets: &[T::Real],
        _result: &mut [T],
    ) {
        panic!("Cannot directly use HelmholtzHypersingularAssembler");
    }
    fn kernel_assemble_st(&self, _sources: &[T::Real], _targets: &[T::Real], _result: &mut [T]) {
        panic!("Cannot directly use HelmholtzHypersingularAssembler");
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
        let mut curlcurl = self
            .curl_curl_assembler
            .assemble_singular::<TestGrid, TrialGrid, Element>(shape, trial_space, test_space);
        curlcurl.add(
            self.normal_normal_assembler
                .assemble_singular::<TestGrid, TrialGrid, Element>(shape, trial_space, test_space),
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

        let mut curlcurl = self
            .curl_curl_assembler
            .assemble_singular_correction::<TestGrid, TrialGrid, Element>(
                shape,
                trial_space,
                test_space,
            );
        curlcurl.add(
            self.normal_normal_assembler
                .assemble_singular_correction::<TestGrid, TrialGrid, Element>(
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

        self.curl_curl_assembler
            .assemble_nonsingular_into_dense::<TestGrid, TrialGrid, Element>(
                output,
                trial_space,
                test_space,
                trial_colouring,
                test_colouring,
            );
        self.normal_normal_assembler
            .assemble_nonsingular_into_dense::<TestGrid, TrialGrid, Element>(
                output,
                trial_space,
                test_space,
                trial_colouring,
                test_colouring,
            );
    }
}
