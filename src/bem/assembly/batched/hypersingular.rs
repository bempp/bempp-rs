//! Hypersingular assemblers
use super::{equal_grids, BatchedAssembler, EvalType, RlstArray, SparseMatrixData};
use crate::traits::{
    bem::FunctionSpace, element::FiniteElement, grid::GridType, types::ReferenceCellType,
};
use green_kernels::{helmholtz_3d::Helmholtz3dKernel, laplace_3d::Laplace3dKernel, traits::Kernel};
use rlst::{RlstScalar, Shape, UnsafeRandomAccessByRef};
use std::collections::HashMap;

#[allow(clippy::too_many_arguments)]
unsafe fn hyp_test_trial_product<T: RlstScalar>(
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

    ((num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 3])).unwrap()
        * test0
        - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 0])).unwrap()
            * test1)
        * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 3]))
            .unwrap()
            * trial0
            - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 0]))
                .unwrap()
                * trial1)
        + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 4])).unwrap()
            * test0
            - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 1]))
                .unwrap()
                * test1)
            * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 4]))
                .unwrap()
                * trial0
                - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 1]))
                    .unwrap()
                    * trial1)
        + (num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 5])).unwrap()
            * test0
            - num::cast::<T::Real, T>(*test_jacobians.get_unchecked([test_point_index, 2]))
                .unwrap()
                * test1)
            * (num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 5]))
                .unwrap()
                * trial0
                - num::cast::<T::Real, T>(*trial_jacobians.get_unchecked([trial_point_index, 2]))
                    .unwrap()
                    * trial1))
        / num::cast::<T::Real, T>(test_jdets[test_point_index] * trial_jdets[trial_point_index])
            .unwrap()
}

/// Assembler for a Laplace hypersingular operator
pub struct LaplaceHypersingularAssembler<const BATCHSIZE: usize, T: RlstScalar> {
    kernel: Laplace3dKernel<T>,
}
impl<const BATCHSIZE: usize, T: RlstScalar> Default
    for LaplaceHypersingularAssembler<BATCHSIZE, T>
{
    fn default() -> Self {
        Self {
            kernel: Laplace3dKernel::<T>::new(),
        }
    }
}
impl<const BATCHSIZE: usize, T: RlstScalar> BatchedAssembler
    for LaplaceHypersingularAssembler<BATCHSIZE, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    const BATCHSIZE: usize = BATCHSIZE;
    type T = T;
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
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::Value, sources, targets, result);
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
struct HelmholtzHypersingularCurlCurlAssembler<const BATCHSIZE: usize, T: RlstScalar<Complex = T>> {
    kernel: Helmholtz3dKernel<T>,
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>>
    HelmholtzHypersingularCurlCurlAssembler<BATCHSIZE, T>
{
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
        }
    }
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>> BatchedAssembler
    for HelmholtzHypersingularCurlCurlAssembler<BATCHSIZE, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    const BATCHSIZE: usize = BATCHSIZE;
    type T = T;
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
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::Value, sources, targets, result);
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
struct HelmholtzHypersingularNormalNormalAssembler<
    const BATCHSIZE: usize,
    T: RlstScalar<Complex = T>,
> {
    kernel: Helmholtz3dKernel<T>,
    wavenumber: T::Real,
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>>
    HelmholtzHypersingularNormalNormalAssembler<BATCHSIZE, T>
{
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            kernel: Helmholtz3dKernel::<T>::new(wavenumber),
            wavenumber,
        }
    }
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>> BatchedAssembler
    for HelmholtzHypersingularNormalNormalAssembler<BATCHSIZE, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 0;
    const BATCHSIZE: usize = BATCHSIZE;
    type T = T;
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<T, 2>,
        test_normals: &RlstArray<T::Real, 2>,
        trial_normals: &RlstArray<T::Real, 2>,
        index: usize,
    ) -> T {
        -num::cast::<T::Real, T>(self.wavenumber.powi(2)).unwrap()
            * *k.get_unchecked([0, index])
            * (num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 0])).unwrap()
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 0])).unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 1])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 1])).unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([index, 2])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([index, 2])).unwrap())
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
            * (num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 0])).unwrap()
                * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 0])).unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 1])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 1]))
                        .unwrap()
                + num::cast::<T::Real, T>(*trial_normals.get_unchecked([trial_index, 2])).unwrap()
                    * num::cast::<T::Real, T>(*test_normals.get_unchecked([test_index, 2]))
                        .unwrap())
    }
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[T::Real],
        targets: &[T::Real],
        result: &mut [T],
    ) {
        self.kernel
            .assemble_diagonal_st(EvalType::Value, sources, targets, result);
    }
    fn kernel_assemble_st(&self, sources: &[T::Real], targets: &[T::Real], result: &mut [T]) {
        self.kernel
            .assemble_st(EvalType::Value, sources, targets, result);
    }
}

/// Assembler for curl-curl term of Helmholtz hypersingular operator
pub struct HelmholtzHypersingularAssembler<const BATCHSIZE: usize, T: RlstScalar<Complex = T>> {
    curl_curl_assembler: HelmholtzHypersingularCurlCurlAssembler<BATCHSIZE, T>,
    normal_normal_assembler: HelmholtzHypersingularNormalNormalAssembler<BATCHSIZE, T>,
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>>
    HelmholtzHypersingularAssembler<BATCHSIZE, T>
{
    /// Create a new assembler
    pub fn new(wavenumber: T::Real) -> Self {
        Self {
            curl_curl_assembler: HelmholtzHypersingularCurlCurlAssembler::<BATCHSIZE, T>::new(
                wavenumber,
            ),
            normal_normal_assembler:
                HelmholtzHypersingularNormalNormalAssembler::<BATCHSIZE, T>::new(wavenumber),
        }
    }
}
impl<const BATCHSIZE: usize, T: RlstScalar<Complex = T>> BatchedAssembler
    for HelmholtzHypersingularAssembler<BATCHSIZE, T>
{
    const DERIV_SIZE: usize = 1;
    const TABLE_DERIVS: usize = 1;
    const BATCHSIZE: usize = BATCHSIZE;
    type T = T;
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
    fn kernel_assemble_diagonal_st(
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
        TestGrid: GridType<T = T::Real> + Sync,
        TrialGrid: GridType<T = T::Real> + Sync,
        Element: FiniteElement<T = T> + Sync,
    >(
        &self,
        qdegree: usize,
        shape: [usize; 2],
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> SparseMatrixData<T> {
        let mut curlcurl = self
            .curl_curl_assembler
            .assemble_singular::<TestGrid, TrialGrid, Element>(
                qdegree,
                shape,
                trial_space,
                test_space,
            );
        curlcurl.add(
            self.normal_normal_assembler
                .assemble_singular::<TestGrid, TrialGrid, Element>(
                    qdegree,
                    shape,
                    trial_space,
                    test_space,
                ),
        );
        curlcurl
    }

    fn assemble_singular_correction<
        TestGrid: GridType<T = T::Real> + Sync,
        TrialGrid: GridType<T = T::Real> + Sync,
        Element: FiniteElement<T = T> + Sync,
    >(
        &self,
        npts_test: usize,
        npts_trial: usize,
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
                npts_test,
                npts_trial,
                shape,
                trial_space,
                test_space,
            );
        curlcurl.add(
            self.normal_normal_assembler
                .assemble_singular_correction::<TestGrid, TrialGrid, Element>(
                    npts_test,
                    npts_trial,
                    shape,
                    trial_space,
                    test_space,
                ),
        );
        curlcurl
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_nonsingular_into_dense<
        TestGrid: GridType<T = T::Real> + Sync,
        TrialGrid: GridType<T = T::Real> + Sync,
        Element: FiniteElement<T = T> + Sync,
    >(
        &self,
        output: &mut RlstArray<T, 2>,
        npts_test: usize,
        npts_trial: usize,
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
                npts_test,
                npts_trial,
                trial_space,
                test_space,
                trial_colouring,
                test_colouring,
            );
        self.normal_normal_assembler
            .assemble_nonsingular_into_dense::<TestGrid, TrialGrid, Element>(
                output,
                npts_test,
                npts_trial,
                trial_space,
                test_space,
                trial_colouring,
                test_colouring,
            );
    }
}
