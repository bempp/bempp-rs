//! Batched dense assembly
use crate::assembly::common::{RawData2D, SparseMatrixData};
use bempp_element::reference_cell;
use bempp_grid::common::{compute_dets23, compute_normals_from_jacobians23};
use bempp_quadrature::duffy::{
    quadrilateral_duffy, quadrilateral_triangle_duffy, triangle_duffy, triangle_quadrilateral_duffy,
};
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_traits::bem::FunctionSpace;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{CellType, GridType, ReferenceMapType, TopologyType};
use bempp_traits::types::EvalType;
use bempp_traits::types::ReferenceCellType;
use rayon::prelude::*;
use rlst::CsrMatrix;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, rlst_dynamic_array4, Array, BaseArray,
    RandomAccessMut, RawAccess, RawAccessMut, RlstScalar, Shape, UnsafeRandomAccessByRef,
    VectorContainer,
};
use std::collections::HashMap;

mod adjoint_double_layer;
mod double_layer;
mod hypersingular;
mod single_layer;
pub use adjoint_double_layer::{
    HelmholtzAdjointDoubleLayerAssembler, LaplaceAdjointDoubleLayerAssembler,
};
pub use double_layer::{HelmholtzDoubleLayerAssembler, LaplaceDoubleLayerAssembler};
pub use hypersingular::{HelmholtzHypersingularAssembler, LaplaceHypersingularAssembler};
pub use single_layer::{HelmholtzSingleLayerAssembler, LaplaceSingleLayerAssembler};

type RlstArray<T, const DIM: usize> = Array<T, BaseArray<T, VectorContainer<T>, DIM>, DIM>;

fn equal_grids<TestGrid: GridType, TrialGrid: GridType>(
    test_grid: &TestGrid,
    trial_grid: &TrialGrid,
) -> bool {
    std::ptr::addr_of!(*test_grid) as usize == std::ptr::addr_of!(*trial_grid) as usize
}
fn neighbours<TestGrid: GridType, TrialGrid: GridType>(
    test_grid: &TestGrid,
    trial_grid: &TrialGrid,
    test_cell: usize,
    trial_cell: usize,
) -> bool {
    if !equal_grids(test_grid, trial_grid) {
        false
    } else {
        let test_vertices = trial_grid
            .cell_from_index(test_cell)
            .topology()
            .vertex_indices()
            .collect::<Vec<_>>();
        for v in trial_grid
            .cell_from_index(trial_cell)
            .topology()
            .vertex_indices()
        {
            if test_vertices.contains(&v) {
                return true;
            }
        }
        false
    }
}

fn get_singular_quadrature_rule(
    test_celltype: ReferenceCellType,
    trial_celltype: ReferenceCellType,
    pairs: &[(usize, usize)],
    npoints: usize,
) -> TestTrialNumericalQuadratureDefinition {
    if pairs.is_empty() {
        panic!("Non-singular rule requested.");
    }
    let con = CellToCellConnectivity {
        connectivity_dimension: match pairs.len() {
            1 => 0,
            2 => 1,
            _ => 2,
        },
        local_indices: pairs.to_vec(),
    };
    match test_celltype {
        ReferenceCellType::Triangle => match trial_celltype {
            ReferenceCellType::Triangle => triangle_duffy(&con, npoints).unwrap(),
            ReferenceCellType::Quadrilateral => {
                triangle_quadrilateral_duffy(&con, npoints).unwrap()
            }
            _ => {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
        },
        ReferenceCellType::Quadrilateral => match trial_celltype {
            ReferenceCellType::Triangle => quadrilateral_triangle_duffy(&con, npoints).unwrap(),
            ReferenceCellType::Quadrilateral => quadrilateral_duffy(&con, npoints).unwrap(),
            _ => {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
        },
        _ => {
            unimplemented!("Only triangles and quadrilaterals are currently supported");
        }
    }
}

/// Assemble the contribution to the terms of a matrix for a batch of pairs of adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular<
    T: RlstScalar,
    TestGrid: GridType<T = T::Real>,
    TrialGrid: GridType<T = T::Real>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedAssembler<T = T>,
    deriv_size: usize,
    shape: [usize; 2],
    trial_cell_type: ReferenceCellType,
    test_cell_type: ReferenceCellType,
    trial_space: &impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element>,
    test_space: &impl FunctionSpace<Grid = TestGrid, FiniteElement = Element>,
    cell_pairs: &[(usize, usize)],
    trial_points: &RlstArray<T::Real, 2>,
    test_points: &RlstArray<T::Real, 2>,
    weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> SparseMatrixData<T> {
    let mut output = SparseMatrixData::<T>::new_known_size(
        shape,
        cell_pairs.len()
            * trial_space.element(trial_cell_type).dim()
            * test_space.element(test_cell_type).dim(),
    );
    let npts = weights.len();
    debug_assert!(weights.len() == npts);
    debug_assert!(test_points.shape()[0] == npts);
    debug_assert!(trial_points.shape()[0] == npts);

    let grid = test_space.grid();
    assert_eq!(grid.physical_dimension(), 3);
    assert_eq!(grid.domain_dimension(), 2);

    // Memory assignment to be moved elsewhere as passed into here mutable?
    let mut k = rlst_dynamic_array2!(T, [deriv_size, npts]);
    let zero = num::cast::<f64, T::Real>(0.0).unwrap();
    let mut test_jdet = vec![zero; npts];
    let mut test_mapped_pts = rlst_dynamic_array2!(T::Real, [npts, 3]);
    let mut test_jacobians = rlst_dynamic_array2!(T::Real, [npts, 6]);
    let mut test_normals = rlst_dynamic_array2!(T::Real, [npts, 3]);

    let mut trial_jdet = vec![zero; npts];
    let mut trial_mapped_pts = rlst_dynamic_array2!(T::Real, [npts, 3]);
    let mut trial_jacobians = rlst_dynamic_array2!(T::Real, [npts, 6]);
    let mut trial_normals = rlst_dynamic_array2!(T::Real, [npts, 3]);

    let test_evaluator = grid.reference_to_physical_map(test_points.data());
    let trial_evaluator = grid.reference_to_physical_map(trial_points.data());

    for (test_cell, trial_cell) in cell_pairs {
        test_evaluator.jacobian(*test_cell, test_jacobians.data_mut());
        compute_normals_from_jacobians23(test_jacobians.data(), test_normals.data_mut());
        compute_dets23(test_jacobians.data(), &mut test_jdet);
        test_evaluator.reference_to_physical(*test_cell, test_mapped_pts.data_mut());

        trial_evaluator.jacobian(*trial_cell, trial_jacobians.data_mut());
        compute_normals_from_jacobians23(trial_jacobians.data(), trial_normals.data_mut());
        compute_dets23(trial_jacobians.data(), &mut trial_jdet);
        trial_evaluator.reference_to_physical(*trial_cell, trial_mapped_pts.data_mut());

        assembler.kernel_assemble_diagonal_st(
            test_mapped_pts.data(),
            trial_mapped_pts.data(),
            k.data_mut(),
        );

        let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
        let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();
        for (test_i, test_dof) in test_dofs.iter().enumerate() {
            for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                let mut sum = num::cast::<f64, T>(0.0).unwrap();

                for (index, wt) in weights.iter().enumerate() {
                    unsafe {
                        sum += assembler.singular_kernel_value(
                            &k,
                            &test_normals,
                            &trial_normals,
                            index,
                        ) * assembler.test_trial_product(
                            test_table,
                            trial_table,
                            &test_jacobians,
                            &trial_jacobians,
                            &test_jdet,
                            &trial_jdet,
                            index,
                            index,
                            test_i,
                            trial_i,
                        ) * num::cast::<T::Real, T>(
                            *wt * *test_jdet.get_unchecked(index)
                                * *trial_jdet.get_unchecked(index),
                        )
                        .unwrap();
                    }
                }
                output.rows.push(*test_dof);
                output.cols.push(*trial_dof);
                output.data.push(sum);
            }
        }
    }
    output
}

/// Assemble the contribution to the terms of a matrix for a batch of non-adjacent cells
#[allow(clippy::too_many_arguments)]
fn assemble_batch_nonadjacent<
    T: RlstScalar,
    TestGrid: GridType<T = T::Real>,
    TrialGrid: GridType<T = T::Real>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedAssembler<T = T>,
    deriv_size: usize,
    output: &RawData2D<T>,
    trial_space: &impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element>,
    trial_cells: &[usize],
    test_space: &impl FunctionSpace<Grid = TestGrid, FiniteElement = Element>,
    test_cells: &[usize],
    trial_points: &RlstArray<T::Real, 2>,
    trial_weights: &[T::Real],
    test_points: &RlstArray<T::Real, 2>,
    test_weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> usize {
    let npts_test = test_weights.len();
    let npts_trial = trial_weights.len();
    debug_assert!(test_points.shape()[0] == npts_test);
    debug_assert!(trial_points.shape()[0] == npts_trial);

    let test_grid = test_space.grid();
    let trial_grid = trial_space.grid();

    assert_eq!(test_grid.physical_dimension(), 3);
    assert_eq!(test_grid.domain_dimension(), 2);
    assert_eq!(trial_grid.physical_dimension(), 3);
    assert_eq!(trial_grid.domain_dimension(), 2);

    let mut k = rlst_dynamic_array3!(T, [npts_test, deriv_size, npts_trial]);
    let zero = num::cast::<f64, T::Real>(0.0).unwrap();
    let mut test_jdet = vec![zero; npts_test];
    let mut test_mapped_pts = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_normals = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_jacobians = rlst_dynamic_array2!(T::Real, [npts_test, 6]);

    let test_evaluator = test_grid.reference_to_physical_map(test_points.data());
    let trial_evaluator = trial_grid.reference_to_physical_map(trial_points.data());

    let mut trial_jdet = vec![vec![zero; npts_trial]; trial_cells.len()];
    let mut trial_mapped_pts = vec![];
    let mut trial_normals = vec![];
    let mut trial_jacobians = vec![];
    for _i in 0..trial_cells.len() {
        trial_mapped_pts.push(rlst_dynamic_array2!(T::Real, [npts_trial, 3]));
        trial_normals.push(rlst_dynamic_array2!(T::Real, [npts_trial, 3]));
        trial_jacobians.push(rlst_dynamic_array2!(T::Real, [npts_trial, 6]));
    }

    for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
        trial_evaluator.jacobian(*trial_cell, trial_jacobians[trial_cell_i].data_mut());
        compute_dets23(
            trial_jacobians[trial_cell_i].data(),
            &mut trial_jdet[trial_cell_i],
        );
        compute_normals_from_jacobians23(
            trial_jacobians[trial_cell_i].data(),
            trial_normals[trial_cell_i].data_mut(),
        );
        trial_evaluator
            .reference_to_physical(*trial_cell, trial_mapped_pts[trial_cell_i].data_mut());
    }

    let mut sum: T;
    let mut trial_integrands = vec![num::cast::<f64, T>(0.0).unwrap(); npts_trial];

    for test_cell in test_cells {
        test_evaluator.jacobian(*test_cell, test_jacobians.data_mut());
        compute_dets23(test_jacobians.data(), &mut test_jdet);
        compute_normals_from_jacobians23(test_jacobians.data(), test_normals.data_mut());
        test_evaluator.reference_to_physical(*test_cell, test_mapped_pts.data_mut());

        for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
            if neighbours(test_grid, trial_grid, *test_cell, *trial_cell) {
                continue;
            }

            assembler.kernel_assemble_st(
                test_mapped_pts.data(),
                trial_mapped_pts[trial_cell_i].data(),
                k.data_mut(),
            );

            let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
            let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();

            for (test_i, test_dof) in test_dofs.iter().enumerate() {
                for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                    for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                        trial_integrands[trial_index] = num::cast::<T::Real, T>(
                            *trial_wt * trial_jdet[trial_cell_i][trial_index],
                        )
                        .unwrap();
                    }
                    sum = num::cast::<f64, T>(0.0).unwrap();
                    for (test_index, test_wt) in test_weights.iter().enumerate() {
                        let test_integrand =
                            num::cast::<T::Real, T>(*test_wt * test_jdet[test_index]).unwrap();
                        for trial_index in 0..npts_trial {
                            sum += unsafe {
                                assembler.nonsingular_kernel_value(
                                    &k,
                                    &test_normals,
                                    &trial_normals[trial_cell_i],
                                    test_index,
                                    trial_index,
                                ) * test_integrand
                                    * *trial_integrands.get_unchecked(trial_index)
                                    * assembler.test_trial_product(
                                        test_table,
                                        trial_table,
                                        &test_jacobians,
                                        &trial_jacobians[trial_cell_i],
                                        &test_jdet,
                                        &trial_jdet[trial_cell_i],
                                        test_index,
                                        trial_index,
                                        test_i,
                                        trial_i,
                                    )
                            };
                        }
                    }
                    // TODO: should we write into a result array, then copy into output after this loop?
                    unsafe {
                        *output.data.add(*test_dof + output.shape[0] * *trial_dof) += sum;
                    }
                }
            }
        }
    }
    1
}

/// Assemble the contribution to the terms of a matrix for a batch of pairs of adjacent cells if an (incorrect) non-singular quadrature rule was used
#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular_correction<
    T: RlstScalar,
    TestGrid: GridType<T = T::Real>,
    TrialGrid: GridType<T = T::Real>,
    Element: FiniteElement<T = T> + Sync,
>(
    assembler: &impl BatchedAssembler<T = T>,
    deriv_size: usize,
    shape: [usize; 2],
    trial_cell_type: ReferenceCellType,
    test_cell_type: ReferenceCellType,
    trial_space: &impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element>,
    test_space: &impl FunctionSpace<Grid = TestGrid, FiniteElement = Element>,
    cell_pairs: &[(usize, usize)],
    trial_points: &RlstArray<T::Real, 2>,
    trial_weights: &[T::Real],
    test_points: &RlstArray<T::Real, 2>,
    test_weights: &[T::Real],
    trial_table: &RlstArray<T, 4>,
    test_table: &RlstArray<T, 4>,
) -> SparseMatrixData<T> {
    let mut output = SparseMatrixData::<T>::new_known_size(
        shape,
        cell_pairs.len()
            * trial_space.element(trial_cell_type).dim()
            * test_space.element(test_cell_type).dim(),
    );
    let npts_test = test_weights.len();
    let npts_trial = trial_weights.len();
    debug_assert!(test_points.shape()[0] == npts_test);
    debug_assert!(trial_points.shape()[0] == npts_trial);

    let grid = test_space.grid();
    assert_eq!(grid.physical_dimension(), 3);
    assert_eq!(grid.domain_dimension(), 2);

    let mut k = rlst_dynamic_array3!(T, [npts_test, deriv_size, npts_trial]);

    let zero = num::cast::<f64, T::Real>(0.0).unwrap();

    let mut test_jdet = vec![zero; npts_test];
    let mut test_mapped_pts = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_normals = rlst_dynamic_array2!(T::Real, [npts_test, 3]);
    let mut test_jacobians = rlst_dynamic_array2!(T::Real, [npts_test, 6]);

    let mut trial_jdet = vec![zero; npts_trial];
    let mut trial_mapped_pts = rlst_dynamic_array2!(T::Real, [npts_trial, 3]);
    let mut trial_normals = rlst_dynamic_array2!(T::Real, [npts_trial, 3]);
    let mut trial_jacobians = rlst_dynamic_array2!(T::Real, [npts_trial, 6]);

    let test_evaluator = grid.reference_to_physical_map(test_points.data());
    let trial_evaluator = grid.reference_to_physical_map(trial_points.data());

    let mut sum: T;
    let mut trial_integrands = vec![num::cast::<f64, T>(0.0).unwrap(); npts_trial];

    for (test_cell, trial_cell) in cell_pairs {
        test_evaluator.jacobian(*test_cell, test_jacobians.data_mut());
        compute_dets23(test_jacobians.data(), &mut test_jdet);
        compute_normals_from_jacobians23(test_jacobians.data(), test_normals.data_mut());
        test_evaluator.reference_to_physical(*test_cell, test_mapped_pts.data_mut());

        trial_evaluator.jacobian(*trial_cell, trial_jacobians.data_mut());
        compute_dets23(trial_jacobians.data(), &mut trial_jdet);
        compute_normals_from_jacobians23(trial_jacobians.data(), trial_normals.data_mut());
        trial_evaluator.reference_to_physical(*trial_cell, trial_mapped_pts.data_mut());

        assembler.kernel_assemble_st(
            test_mapped_pts.data(),
            trial_mapped_pts.data(),
            k.data_mut(),
        );

        let test_dofs = test_space.cell_dofs(*test_cell).unwrap();
        let trial_dofs = trial_space.cell_dofs(*trial_cell).unwrap();
        for (test_i, test_dof) in test_dofs.iter().enumerate() {
            for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                    trial_integrands[trial_index] =
                        num::cast::<T::Real, T>(*trial_wt * trial_jdet[trial_index]).unwrap();
                }
                sum = num::cast::<f64, T>(0.0).unwrap();
                for (test_index, test_wt) in test_weights.iter().enumerate() {
                    let test_integrand =
                        num::cast::<T::Real, T>(*test_wt * test_jdet[test_index]).unwrap();
                    for trial_index in 0..npts_trial {
                        sum += unsafe {
                            assembler.nonsingular_kernel_value(
                                &k,
                                &test_normals,
                                &trial_normals,
                                test_index,
                                trial_index,
                            ) * test_integrand
                                * *trial_integrands.get_unchecked(trial_index)
                                * assembler.test_trial_product(
                                    test_table,
                                    trial_table,
                                    &test_jacobians,
                                    &trial_jacobians,
                                    &test_jdet,
                                    &trial_jdet,
                                    test_index,
                                    trial_index,
                                    test_i,
                                    trial_i,
                                )
                        };
                    }
                }
                output.rows.push(*test_dof);
                output.cols.push(*trial_dof);
                output.data.push(sum);
            }
        }
    }
    output
}

pub trait BatchedAssembler: Sync + Sized {
    //! Batched assembler
    //!
    //! Assemble operators by processing batches of cells in parallel

    /// Scalar type
    type T: RlstScalar;
    /// Number of derivatives
    const DERIV_SIZE: usize;
    /// Number of derivatives needed in basis function tables
    const TABLE_DERIVS: usize;
    /// The number of cells in each batch
    const BATCHSIZE: usize;

    /// Return the kernel value to use in the integrand when using a singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` may be used
    unsafe fn singular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 2>,
        test_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        trial_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        index: usize,
    ) -> Self::T;

    /// Return the kernel value to use in the integrand when using a non-singular quadrature rule
    ///
    /// # Safety
    /// This method is unsafe to allow `get_unchecked` may be used
    unsafe fn nonsingular_kernel_value(
        &self,
        k: &RlstArray<Self::T, 3>,
        test_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        trial_normals: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        test_index: usize,
        trial_index: usize,
    ) -> Self::T;

    /// Evaluate the kernel values for all source and target pairs
    ///
    /// For each source, the kernel is evaluated for exactly one target. This is equivalent to taking the diagonal of the matrix assembled by `kernel_assemble_st`
    fn kernel_assemble_diagonal_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// Evaluate the kernel values for all sources and all targets
    ///
    /// For every source, the kernel is evaluated for every target.
    fn kernel_assemble_st(
        &self,
        sources: &[<Self::T as RlstScalar>::Real],
        targets: &[<Self::T as RlstScalar>::Real],
        result: &mut [Self::T],
    );

    /// The product of a test and trial function
    ///
    /// # Safety
    /// This function uses unchecked access into tables
    #[allow(clippy::too_many_arguments)]
    unsafe fn test_trial_product(
        &self,
        test_table: &RlstArray<Self::T, 4>,
        trial_table: &RlstArray<Self::T, 4>,
        _test_jacobians: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        _trial_jacobians: &RlstArray<<Self::T as RlstScalar>::Real, 2>,
        _test_jdets: &[<Self::T as RlstScalar>::Real],
        _trial_jdets: &[<Self::T as RlstScalar>::Real],
        test_point_index: usize,
        trial_point_index: usize,
        test_basis_index: usize,
        trial_basis_index: usize,
    ) -> Self::T {
        *test_table.get_unchecked([0, test_point_index, test_basis_index, 0])
            * *trial_table.get_unchecked([0, trial_point_index, trial_basis_index, 0])
    }

    /// Assemble the singular contributions
    fn assemble_singular<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        qdegree: usize,
        shape: [usize; 2],
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> SparseMatrixData<Self::T> {
        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return SparseMatrixData::new(shape);
        }

        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if shape[0] != test_space.global_size() || shape[1] != trial_space.global_size() {
            panic!("Matrix has wrong shape");
        }

        let grid = test_space.grid();

        let mut qweights = vec![];
        let mut trial_points = vec![];
        let mut test_points = vec![];
        let mut trial_tables = vec![];
        let mut test_tables = vec![];

        let mut cell_blocks = vec![];

        for test_cell_type in grid.cell_types() {
            for trial_cell_type in grid.cell_types() {
                let offset = qweights.len();

                let mut possible_pairs = vec![];
                // Vertex-adjacent
                for i in 0..reference_cell::entity_counts(*test_cell_type)[0] {
                    for j in 0..reference_cell::entity_counts(*trial_cell_type)[0] {
                        possible_pairs.push(vec![(i, j)]);
                    }
                }
                // edge-adjacent
                for test_e in reference_cell::edges(*test_cell_type) {
                    for trial_e in reference_cell::edges(*trial_cell_type) {
                        possible_pairs.push(vec![(test_e[0], trial_e[0]), (test_e[1], trial_e[1])]);
                        possible_pairs.push(vec![(test_e[1], trial_e[0]), (test_e[0], trial_e[1])]);
                    }
                }
                // Same cell
                if test_cell_type == trial_cell_type {
                    possible_pairs.push(
                        (0..reference_cell::entity_counts(*test_cell_type)[0])
                            .map(&|i| (i, i))
                            .collect::<Vec<_>>(),
                    );
                }

                let mut pair_indices: HashMap<Vec<(usize, usize)>, usize> = HashMap::new();
                for (i, pairs) in possible_pairs.iter().enumerate() {
                    pair_indices.insert(pairs.clone(), i);
                }

                for pairs in &possible_pairs {
                    let qrule = get_singular_quadrature_rule(
                        *test_cell_type,
                        *trial_cell_type,
                        pairs,
                        qdegree,
                    );
                    let npts = qrule.weights.len();

                    let mut points = rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts, 2]);
                    for i in 0..npts {
                        for j in 0..2 {
                            *points.get_mut([i, j]).unwrap() =
                                num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                    qrule.trial_points[2 * i + j],
                                )
                                .unwrap();
                        }
                    }
                    let trial_element = trial_space.element(*trial_cell_type);
                    let mut table = rlst_dynamic_array4!(
                        Self::T,
                        trial_element.tabulate_array_shape(Self::TABLE_DERIVS, points.shape()[0])
                    );
                    trial_element.tabulate(&points, Self::TABLE_DERIVS, &mut table);
                    trial_points.push(points);
                    trial_tables.push(table);

                    let mut points = rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts, 2]);
                    for i in 0..npts {
                        for j in 0..2 {
                            *points.get_mut([i, j]).unwrap() =
                                num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                    qrule.test_points[2 * i + j],
                                )
                                .unwrap();
                        }
                    }
                    let test_element = test_space.element(*test_cell_type);
                    let mut table = rlst_dynamic_array4!(
                        Self::T,
                        test_element.tabulate_array_shape(Self::TABLE_DERIVS, points.shape()[0])
                    );
                    test_element.tabulate(&points, Self::TABLE_DERIVS, &mut table);
                    test_points.push(points);
                    test_tables.push(table);
                    qweights.push(
                        qrule
                            .weights
                            .iter()
                            .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                            .collect::<Vec<_>>(),
                    );
                }
                let mut cell_pairs: Vec<Vec<(usize, usize)>> = vec![vec![]; possible_pairs.len()];
                for vertex in 0..grid.number_of_vertices() {
                    let test_cells = grid
                        .vertex_to_cells(vertex)
                        .iter()
                        .map(|c| c.cell)
                        // TODO: store quadrature by cell type and remove this filter
                        .filter(|c| {
                            grid.cell_from_index(*c).topology().cell_type() == *test_cell_type
                        })
                        .collect::<Vec<_>>();
                    let trial_cells = grid
                        .vertex_to_cells(vertex)
                        .iter()
                        .map(|c| c.cell)
                        // TODO: store quadrature by cell type and remove this filter
                        .filter(|c| {
                            grid.cell_from_index(*c).topology().cell_type() == *trial_cell_type
                        })
                        .collect::<Vec<_>>();
                    for test_cell in &test_cells {
                        let test_vertices = grid
                            .cell_from_index(*test_cell)
                            .topology()
                            .vertex_indices()
                            .collect::<Vec<_>>();
                        for trial_cell in &trial_cells {
                            let mut smallest = true;
                            let mut pairs = vec![];
                            for (trial_i, trial_v) in grid
                                .cell_from_index(*trial_cell)
                                .topology()
                                .vertex_indices()
                                .enumerate()
                            {
                                for (test_i, test_v) in test_vertices.iter().enumerate() {
                                    if *test_v == trial_v {
                                        if *test_v < vertex {
                                            smallest = false;
                                            break;
                                        }
                                        pairs.push((test_i, trial_i));
                                    }
                                }
                                if !smallest {
                                    break;
                                }
                            }
                            if smallest {
                                cell_pairs[pair_indices[&pairs]].push((*test_cell, *trial_cell));
                            }
                        }
                    }
                }
                for (i, cells) in cell_pairs.iter().enumerate() {
                    let mut start = 0;
                    while start < cells.len() {
                        let end = if start + Self::BATCHSIZE < cells.len() {
                            start + Self::BATCHSIZE
                        } else {
                            cells.len()
                        };
                        cell_blocks.push((
                            offset + i,
                            *trial_cell_type,
                            *test_cell_type,
                            cells[start..end].to_vec(),
                        ));
                        start = end;
                    }
                }
            }
        }
        cell_blocks
            .into_par_iter()
            .map(|(i, trial_cell_type, test_cell_type, cell_block)| {
                assemble_batch_singular::<Self::T, TestGrid, TrialGrid, Element>(
                    self,
                    Self::DERIV_SIZE,
                    shape,
                    trial_cell_type,
                    test_cell_type,
                    trial_space,
                    test_space,
                    &cell_block,
                    &trial_points[i],
                    &test_points[i],
                    &qweights[i],
                    &trial_tables[i],
                    &test_tables[i],
                )
            })
            .reduce(
                || SparseMatrixData::<Self::T>::new(shape),
                |mut a, b| {
                    a.add(b);
                    a
                },
            )
    }

    /// Assemble the singular correction
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        npts_test: usize,
        npts_trial: usize,
        shape: [usize; 2],
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> SparseMatrixData<Self::T> {
        if !equal_grids(test_space.grid(), trial_space.grid()) {
            // If the test and trial grids are different, there are no neighbouring triangles
            return SparseMatrixData::new(shape);
        }

        if !trial_space.is_serial() || !test_space.is_serial() {
            panic!("Dense assembly can only be used for function spaces stored in serial");
        }
        if shape[0] != test_space.global_size() || shape[1] != trial_space.global_size() {
            panic!("Matrix has wrong shape");
        }

        if npts_test != npts_trial {
            panic!("FMM with different test and trial quadrature rules not yet supported");
        }

        let grid = test_space.grid();

        let mut qweights_test = vec![];
        let mut qweights_trial = vec![];
        let mut qpoints_test = vec![];
        let mut qpoints_trial = vec![];
        let mut test_tables = vec![];
        let mut trial_tables = vec![];

        let mut cell_blocks = vec![];

        for test_cell_type in grid.cell_types() {
            for trial_cell_type in grid.cell_types() {
                let index = qweights_test.len();
                let qrule_test = simplex_rule(*test_cell_type, npts_test).unwrap();
                let mut test_pts =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_test, 2]);
                for i in 0..npts_test {
                    for j in 0..2 {
                        *test_pts.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_test.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                qweights_test.push(
                    qrule_test
                        .weights
                        .iter()
                        .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                        .collect::<Vec<_>>(),
                );
                let test_element = test_space.element(*test_cell_type);
                let mut test_table = rlst_dynamic_array4!(
                    Self::T,
                    test_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_test)
                );
                test_element.tabulate(&test_pts, Self::TABLE_DERIVS, &mut test_table);
                test_tables.push(test_table);
                qpoints_test.push(test_pts);

                let qrule_trial = simplex_rule(*trial_cell_type, npts_trial).unwrap();
                let mut trial_pts =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_trial, 2]);
                for i in 0..npts_trial {
                    for j in 0..2 {
                        *trial_pts.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_trial.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                qweights_trial.push(
                    qrule_trial
                        .weights
                        .iter()
                        .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                        .collect::<Vec<_>>(),
                );
                let trial_element = trial_space.element(*trial_cell_type);
                let mut trial_table = rlst_dynamic_array4!(
                    Self::T,
                    trial_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_trial)
                );
                trial_element.tabulate(&trial_pts, Self::TABLE_DERIVS, &mut trial_table);
                trial_tables.push(trial_table);
                qpoints_trial.push(trial_pts);

                let mut cell_pairs: Vec<(usize, usize)> = vec![];

                for vertex in 0..grid.number_of_vertices() {
                    let test_cells = grid
                        .vertex_to_cells(vertex)
                        .iter()
                        .map(|c| c.cell)
                        // TODO: store quadrature by cell type and remove this filter
                        .filter(|c| {
                            grid.cell_from_index(*c).topology().cell_type() == *test_cell_type
                        })
                        .collect::<Vec<_>>();
                    let trial_cells = grid
                        .vertex_to_cells(vertex)
                        .iter()
                        .map(|c| c.cell)
                        // TODO: store quadrature by cell type and remove this filter
                        .filter(|c| {
                            grid.cell_from_index(*c).topology().cell_type() == *trial_cell_type
                        })
                        .collect::<Vec<_>>();
                    for test_cell in &test_cells {
                        for trial_cell in &trial_cells {
                            let mut smallest = true;
                            for trial_v in grid
                                .cell_from_index(*trial_cell)
                                .topology()
                                .vertex_indices()
                            {
                                for test_v in
                                    grid.cell_from_index(*test_cell).topology().vertex_indices()
                                {
                                    if test_v == trial_v && test_v < vertex {
                                        smallest = false;
                                        break;
                                    }
                                }
                                if !smallest {
                                    break;
                                }
                            }
                            if smallest {
                                cell_pairs.push((*test_cell, *trial_cell));
                            }
                        }
                    }
                }

                let mut start = 0;
                while start < cell_pairs.len() {
                    let end = if start + Self::BATCHSIZE < cell_pairs.len() {
                        start + Self::BATCHSIZE
                    } else {
                        cell_pairs.len()
                    };
                    cell_blocks.push((
                        index,
                        *trial_cell_type,
                        *test_cell_type,
                        cell_pairs[start..end].to_vec(),
                    ));
                    start = end;
                }
            }
        }
        cell_blocks
            .into_par_iter()
            .map(|(i, trial_cell_type, test_cell_type, cell_block)| {
                assemble_batch_singular_correction::<Self::T, TestGrid, TrialGrid, Element>(
                    self,
                    Self::DERIV_SIZE,
                    shape,
                    trial_cell_type,
                    test_cell_type,
                    trial_space,
                    test_space,
                    &cell_block,
                    &qpoints_trial[i],
                    &qweights_trial[i],
                    &qpoints_test[i],
                    &qweights_test[i],
                    &trial_tables[i],
                    &test_tables[i],
                )
            })
            .reduce(
                || SparseMatrixData::<Self::T>::new(shape),
                |mut a, b| {
                    a.add(b);
                    a
                },
            )
    }

    /// Assemble the singular contributions into a dense matrix
    fn assemble_singular_into_dense<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        qdegree: usize,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) {
        let sparse_matrix = self.assemble_singular::<TestGrid, TrialGrid, Element>(
            qdegree,
            output.shape(),
            trial_space,
            test_space,
        );
        let data = sparse_matrix.data;
        let rows = sparse_matrix.rows;
        let cols = sparse_matrix.cols;
        for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            *output.get_mut([*i, *j]).unwrap() += *value;
        }
    }

    /// Assemble the singular contributions into a CSR sparse matrix
    fn assemble_singular_into_csr<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        qdegree: usize,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> CsrMatrix<Self::T> {
        let shape = [test_space.global_size(), trial_space.global_size()];
        let sparse_matrix = self.assemble_singular::<TestGrid, TrialGrid, Element>(
            qdegree,
            shape,
            trial_space,
            test_space,
        );

        CsrMatrix::<Self::T>::from_aij(
            sparse_matrix.shape,
            &sparse_matrix.rows,
            &sparse_matrix.cols,
            &sparse_matrix.data,
        )
        .unwrap()
    }

    /// Assemble the singular correction into a dense matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_dense<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        npts_test: usize,
        npts_trial: usize,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) {
        let sparse_matrix = self.assemble_singular_correction::<TestGrid, TrialGrid, Element>(
            npts_test,
            npts_trial,
            output.shape(),
            trial_space,
            test_space,
        );
        let data = sparse_matrix.data;
        let rows = sparse_matrix.rows;
        let cols = sparse_matrix.cols;
        for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
            *output.get_mut([*i, *j]).unwrap() += *value;
        }
    }

    /// Assemble the singular correction into a CSR matrix
    ///
    /// The singular correction is the contribution is the terms for adjacent cells are assembled using an (incorrect) non-singular quadrature rule
    fn assemble_singular_correction_into_csr<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        npts_test: usize,
        npts_trial: usize,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) -> CsrMatrix<Self::T> {
        let shape = [test_space.global_size(), trial_space.global_size()];
        let sparse_matrix = self.assemble_singular_correction::<TestGrid, TrialGrid, Element>(
            npts_test,
            npts_trial,
            shape,
            trial_space,
            test_space,
        );

        CsrMatrix::<Self::T>::from_aij(
            sparse_matrix.shape,
            &sparse_matrix.rows,
            &sparse_matrix.cols,
            &sparse_matrix.data,
        )
        .unwrap()
    }

    /// Assemble into a dense matrix
    fn assemble_into_dense<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
        trial_space: &(impl FunctionSpace<Grid = TrialGrid, FiniteElement = Element> + Sync),
        test_space: &(impl FunctionSpace<Grid = TestGrid, FiniteElement = Element> + Sync),
    ) {
        let test_colouring = test_space.cell_colouring();
        let trial_colouring = trial_space.cell_colouring();

        self.assemble_nonsingular_into_dense::<TestGrid, TrialGrid, Element>(
            output,
            37, // TODO: Allow user to set npts (note: 37 used as rule exists with 37 for both triangles and quads)
            37, // TODO: Allow user to set npts
            trial_space,
            test_space,
            &trial_colouring,
            &test_colouring,
        );
        self.assemble_singular_into_dense::<TestGrid, TrialGrid, Element>(
            output,
            4, // TODO: Allow user to set this
            trial_space,
            test_space,
        );
    }

    /// Assemble the non-singular contributions into a dense matrix
    #[allow(clippy::too_many_arguments)]
    fn assemble_nonsingular_into_dense<
        TestGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        TrialGrid: GridType<T = <Self::T as RlstScalar>::Real> + Sync,
        Element: FiniteElement<T = Self::T> + Sync,
    >(
        &self,
        output: &mut RlstArray<Self::T, 2>,
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

        for test_cell_type in test_space.grid().cell_types() {
            for trial_cell_type in trial_space.grid().cell_types() {
                let qrule_test = simplex_rule(*test_cell_type, npts_test).unwrap();
                let mut qpoints_test =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_test, 2]);
                for i in 0..npts_test {
                    for j in 0..2 {
                        *qpoints_test.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_test.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                let qweights_test = qrule_test
                    .weights
                    .iter()
                    .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                    .collect::<Vec<_>>();
                let qrule_trial = simplex_rule(*trial_cell_type, npts_trial).unwrap();
                let mut qpoints_trial =
                    rlst_dynamic_array2!(<Self::T as RlstScalar>::Real, [npts_trial, 2]);
                for i in 0..npts_trial {
                    for j in 0..2 {
                        *qpoints_trial.get_mut([i, j]).unwrap() =
                            num::cast::<f64, <Self::T as RlstScalar>::Real>(
                                qrule_trial.points[2 * i + j],
                            )
                            .unwrap();
                    }
                }
                let qweights_trial = qrule_trial
                    .weights
                    .iter()
                    .map(|w| num::cast::<f64, <Self::T as RlstScalar>::Real>(*w).unwrap())
                    .collect::<Vec<_>>();

                let test_element = test_space.element(*test_cell_type);
                let mut test_table = rlst_dynamic_array4!(
                    Self::T,
                    test_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_test)
                );
                test_element.tabulate(&qpoints_test, Self::TABLE_DERIVS, &mut test_table);

                let trial_element = trial_space.element(*trial_cell_type);
                let mut trial_table = rlst_dynamic_array4!(
                    Self::T,
                    trial_element.tabulate_array_shape(Self::TABLE_DERIVS, npts_trial)
                );
                trial_element.tabulate(&qpoints_test, Self::TABLE_DERIVS, &mut trial_table);

                let output_raw = RawData2D {
                    data: output.data_mut().as_mut_ptr(),
                    shape: output.shape(),
                };

                for test_c in &test_colouring[test_cell_type] {
                    for trial_c in &trial_colouring[trial_cell_type] {
                        let mut test_cells: Vec<&[usize]> = vec![];
                        let mut trial_cells: Vec<&[usize]> = vec![];

                        let mut test_start = 0;
                        while test_start < test_c.len() {
                            let test_end = if test_start + Self::BATCHSIZE < test_c.len() {
                                test_start + Self::BATCHSIZE
                            } else {
                                test_c.len()
                            };

                            let mut trial_start = 0;
                            while trial_start < trial_c.len() {
                                let trial_end = if trial_start + Self::BATCHSIZE < trial_c.len() {
                                    trial_start + Self::BATCHSIZE
                                } else {
                                    trial_c.len()
                                };
                                test_cells.push(&test_c[test_start..test_end]);
                                trial_cells.push(&trial_c[trial_start..trial_end]);
                                trial_start = trial_end;
                            }
                            test_start = test_end
                        }

                        let numtasks = test_cells.len();
                        let r: usize = (0..numtasks)
                            .into_par_iter()
                            .map(&|t| {
                                assemble_batch_nonadjacent::<Self::T, TestGrid, TrialGrid, Element>(
                                    self,
                                    Self::DERIV_SIZE,
                                    &output_raw,
                                    trial_space,
                                    trial_cells[t],
                                    test_space,
                                    test_cells[t],
                                    &qpoints_trial,
                                    &qweights_trial,
                                    &qpoints_test,
                                    &qweights_test,
                                    &trial_table,
                                    &test_table,
                                )
                            })
                            .sum();
                        assert_eq!(r, numtasks);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::function_space::SerialFunctionSpace;
    use approx::*;
    use bempp_element::element::LagrangeElementFamily;
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::element::Continuity;
    use rlst::RandomAccessByRef;

    #[test]
    fn test_singular_dp0() {
        let grid = regular_sphere::<f64>(0);
        let element = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        let assembler = LaplaceSingleLayerAssembler::<128, f64>::default();
        assembler.assemble_singular_into_dense(&mut matrix, 4, &space, &space);
        let csr = assembler.assemble_singular_into_csr(4, &space, &space);

        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_singular_p1() {
        let grid = regular_sphere::<f64>(0);
        let element = LagrangeElementFamily::<f64>::new(1, Continuity::Continuous);
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        let assembler = LaplaceSingleLayerAssembler::<128, f64>::default();
        assembler.assemble_singular_into_dense(&mut matrix, 4, &space, &space);
        let csr = assembler.assemble_singular_into_csr(4, &space, &space);

        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_singular_dp0_p1() {
        let grid = regular_sphere::<f64>(0);
        let element0 = LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous);
        let element1 = LagrangeElementFamily::<f64>::new(1, Continuity::Continuous);
        let space0 = SerialFunctionSpace::new(&grid, &element0);
        let space1 = SerialFunctionSpace::new(&grid, &element1);

        let ndofs0 = space0.global_size();
        let ndofs1 = space1.global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs1, ndofs0]);
        let assembler = LaplaceSingleLayerAssembler::<128, f64>::default();
        assembler.assemble_singular_into_dense(&mut matrix, 4, &space0, &space1);
        let csr = assembler.assemble_singular_into_csr(4, &space0, &space1);
        let indptr = csr.indptr();
        let indices = csr.indices();
        let data = csr.data();

        let mut row = 0;
        for (i, j) in indices.iter().enumerate() {
            while i >= indptr[row + 1] {
                row += 1;
            }
            assert_relative_eq!(*matrix.get([row, *j]).unwrap(), data[i], epsilon = 1e-8);
        }
    }
}
