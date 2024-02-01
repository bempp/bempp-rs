use crate::assembly::common::{RawData2D, SparseMatrixData};
use crate::function_space::SerialFunctionSpace;
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_traits::arrays::AdjacencyListAccess;
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};
use bempp_traits::kernel::Kernel;
use bempp_traits::types::EvalType;
use rayon::prelude::*;
use rlst_dense::{
    array::Array,
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array2, rlst_dynamic_array3, rlst_dynamic_array4,
    traits::{RandomAccessMut, RawAccess, RawAccessMut, Shape, UnsafeRandomAccessByRef},
};
use rlst_sparse::sparse::csr_mat::CsrMatrix;

fn get_quadrature_rule(
    test_celltype: ReferenceCellType,
    trial_celltype: ReferenceCellType,
    pairs: &[(usize, usize)],
    npoints: usize,
) -> TestTrialNumericalQuadratureDefinition {
    if pairs.is_empty() {
        panic!("Non-singular rule");
    } else {
        // Singular rules
        if test_celltype == ReferenceCellType::Triangle {
            if trial_celltype != ReferenceCellType::Triangle {
                unimplemented!("Mixed meshes not yet supported");
            }
            triangle_duffy(
                &CellToCellConnectivity {
                    connectivity_dimension: if pairs.len() == 1 {
                        0
                    } else if pairs.len() == 2 {
                        1
                    } else {
                        2
                    },
                    local_indices: pairs.to_vec(),
                },
                npoints,
            )
            .unwrap()
        } else {
            if test_celltype != ReferenceCellType::Quadrilateral {
                unimplemented!("Only triangles and quadrilaterals are currently supported");
            }
            if trial_celltype != ReferenceCellType::Quadrilateral {
                unimplemented!("Mixed meshes not yet supported");
            }
            quadrilateral_duffy(
                &CellToCellConnectivity {
                    connectivity_dimension: if pairs.len() == 1 {
                        0
                    } else if pairs.len() == 2 {
                        1
                    } else {
                        2
                    },
                    local_indices: pairs.to_vec(),
                },
                npoints,
            )
            .unwrap()
        }
    }
}

// TODO: use T not f64
#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular<'a>(
    shape: [usize; 2],
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
    cell_pairs: &[(usize, usize)],
    trial_points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    test_points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    weights: &[f64],
    trial_table: &Array<f64, BaseArray<f64, VectorContainer<f64>, 4>, 4>,
    test_table: &Array<f64, BaseArray<f64, VectorContainer<f64>, 4>, 4>,
) -> SparseMatrixData<f64> {
    let mut output = SparseMatrixData::<f64>::new_known_size(
        shape,
        cell_pairs.len() * trial_space.element().dim() * test_space.element().dim(),
    );
    let npts = weights.len();
    debug_assert!(weights.len() == npts);
    debug_assert!(test_points.shape()[0] == npts);
    debug_assert!(trial_points.shape()[0] == npts);

    let grid = test_space.grid();

    // Memory assignment to be moved elsewhere as passed into here mutable?
    let mut k = rlst_dynamic_array2!(f64, [4, npts]);
    let mut k2 = rlst_dynamic_array2!(f64, [1, npts]);
    let mut test_jdet = vec![0.0; npts];
    let mut test_mapped_pts = rlst_dynamic_array2!(f64, [npts, 3]);
    let mut test_normals = rlst_dynamic_array2!(f64, [npts, 3]);

    let mut trial_jdet = vec![0.0; npts];
    let mut trial_mapped_pts = rlst_dynamic_array2!(f64, [npts, 3]);
    let mut trial_normals = rlst_dynamic_array2!(f64, [npts, 3]);

    let trial_element = grid.geometry().element(cell_pairs[0].0);
    let test_element = grid.geometry().element(cell_pairs[0].1);

    let test_evaluator = grid.geometry().get_evaluator(test_element, test_points);
    let trial_evaluator = grid.geometry().get_evaluator(trial_element, trial_points);

    for (test_cell, trial_cell) in cell_pairs {
        let test_cell_tindex = grid.topology().index_map()[*test_cell];
        let test_cell_gindex = grid.geometry().index_map()[*test_cell];
        let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
        let trial_cell_gindex = grid.geometry().index_map()[*trial_cell];

        test_evaluator.compute_normals_and_jacobian_determinants(
            test_cell_gindex,
            &mut test_normals,
            &mut test_jdet,
        );
        test_evaluator.compute_points(test_cell_gindex, &mut test_mapped_pts);

        trial_evaluator.compute_normals_and_jacobian_determinants(
            trial_cell_gindex,
            &mut trial_normals,
            &mut trial_jdet,
        );
        trial_evaluator.compute_points(trial_cell_gindex, &mut trial_mapped_pts);

        kernel.assemble_diagonal_st(
            EvalType::Value,
            test_mapped_pts.data(),
            trial_mapped_pts.data(),
            k2.data_mut(),
        );

        kernel.assemble_diagonal_st(
            EvalType::ValueDeriv,
            test_mapped_pts.data(),
            trial_mapped_pts.data(),
            k.data_mut(),
        );

        for (test_i, test_dof) in test_space
            .dofmap()
            .cell_dofs(test_cell_tindex)
            .unwrap()
            .iter()
            .enumerate()
        {
            for (trial_i, trial_dof) in trial_space
                .dofmap()
                .cell_dofs(trial_cell_tindex)
                .unwrap()
                .iter()
                .enumerate()
            {
                let mut sum = 0.0;

                for (index, wt) in weights.iter().enumerate() {
                    unsafe {
                        sum += (
                            k.get_unchecked([1, index]) * trial_normals.get_unchecked([index, 0])
                            + k.get_unchecked([2, index]) * trial_normals.get_unchecked([index, 1])
                            + k.get_unchecked([3, index]) * trial_normals.get_unchecked([index, 2])
                        ) * wt
                            * test_table.get_unchecked([0, index, test_i, 0])
                            * test_jdet.get_unchecked(index)
                            * trial_table.get_unchecked([0, index, trial_i, 0])
                            * trial_jdet.get_unchecked(index);
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

#[allow(clippy::too_many_arguments)]
fn assemble_batch_nonadjacent<'a, const NPTS_TEST: usize, const NPTS_TRIAL: usize>(
    output: &RawData2D<f64>,
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    trial_cells: &[usize],
    test_space: &SerialFunctionSpace<'a>,
    test_cells: &[usize],
    trial_points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    trial_weights: &[f64],
    test_points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    test_weights: &[f64],
    trial_table: &Array<f64, BaseArray<f64, VectorContainer<f64>, 4>, 4>,
    test_table: &Array<f64, BaseArray<f64, VectorContainer<f64>, 4>, 4>,
) -> usize {
    debug_assert!(test_weights.len() == NPTS_TEST);
    debug_assert!(test_points.shape()[0] == NPTS_TEST);
    debug_assert!(trial_weights.len() == NPTS_TRIAL);
    debug_assert!(trial_points.shape()[0] == NPTS_TRIAL);

    let test_grid = test_space.grid();
    let test_c20 = test_grid.topology().connectivity(2, 0);
    let trial_grid = trial_space.grid();
    let trial_c20 = trial_grid.topology().connectivity(2, 0);

    let mut k = rlst_dynamic_array3!(f64, [NPTS_TEST, 4, NPTS_TRIAL]);
    let mut test_jdet = [0.0; NPTS_TEST];
    let mut test_mapped_pts = rlst_dynamic_array2!(f64, [NPTS_TEST, 3]);
    let mut test_normals = rlst_dynamic_array2!(f64, [NPTS_TEST, 3]);

    let test_element = test_grid.geometry().element(test_cells[0]);
    let trial_element = trial_grid.geometry().element(trial_cells[0]);

    let test_evaluator = test_grid
        .geometry()
        .get_evaluator(test_element, test_points);
    let trial_evaluator = trial_grid
        .geometry()
        .get_evaluator(trial_element, trial_points);

    let mut trial_jdet = vec![[0.0; NPTS_TRIAL]; trial_cells.len()];
    let mut trial_mapped_pts = vec![];
    let mut trial_normals = vec![];
    for _i in 0..trial_cells.len() {
        trial_mapped_pts.push(rlst_dynamic_array2!(f64, [NPTS_TRIAL, 3]));
        trial_normals.push(rlst_dynamic_array2!(f64, [NPTS_TRIAL, 3]));
    }

    for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
        let trial_cell_gindex = trial_grid.geometry().index_map()[*trial_cell];

        trial_evaluator.compute_normals_and_jacobian_determinants(
            trial_cell_gindex,
            &mut trial_normals[trial_cell_i],
            &mut trial_jdet[trial_cell_i],
        );
        trial_evaluator.compute_points(trial_cell_gindex, &mut trial_mapped_pts[trial_cell_i]);
    }

    let mut sum: f64;
    let mut trial_integrands = [0.0; NPTS_TRIAL];

    for test_cell in test_cells {
        let test_cell_tindex = test_grid.topology().index_map()[*test_cell];
        let test_cell_gindex = test_grid.geometry().index_map()[*test_cell];
        let test_vertices = unsafe { test_c20.row_unchecked(test_cell_tindex) };

        test_evaluator.compute_normals_and_jacobian_determinants(
            test_cell_gindex,
            &mut test_normals,
            &mut test_jdet,
        );
        test_evaluator.compute_points(test_cell_gindex, &mut test_mapped_pts);

        for (trial_cell_i, trial_cell) in trial_cells.iter().enumerate() {
            let trial_cell_tindex = trial_grid.topology().index_map()[*trial_cell];
            let trial_vertices = unsafe { trial_c20.row_unchecked(trial_cell_tindex) };

            let mut neighbour = false;
            for v in test_vertices {
                if trial_vertices.contains(v) {
                    neighbour = true;
                    break;
                }
            }

            if neighbour {
                continue;
            }

            kernel.assemble_st(
                EvalType::ValueDeriv,
                test_mapped_pts.data(),
                trial_mapped_pts[trial_cell_i].data(),
                k.data_mut(),
            );

            for (test_i, test_dof) in test_space
                .dofmap()
                .cell_dofs(test_cell_tindex)
                .unwrap()
                .iter()
                .enumerate()
            {
                for (trial_i, trial_dof) in trial_space
                    .dofmap()
                    .cell_dofs(trial_cell_tindex)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                        unsafe {
                            trial_integrands[trial_index] = trial_wt
                                * trial_jdet[trial_cell_i][trial_index]
                                * trial_table.get_unchecked([0, trial_index, trial_i, 0]);
                        }
                    }
                    sum = 0.0;
                    for (test_index, test_wt) in test_weights.iter().enumerate() {
                        let test_integrand = unsafe {
                            test_wt
                                * test_jdet[test_index]
                                * test_table.get_unchecked([0, test_index, test_i, 0])
                        };
                        for trial_index in 0..NPTS_TRIAL {
                            unsafe {
                                sum += (
                                    k.get_unchecked([test_index, 1, trial_index]) * trial_normals[trial_cell_i].get_unchecked([trial_index, 0])
                                    + k.get_unchecked([test_index, 2, trial_index]) * trial_normals[trial_cell_i].get_unchecked([trial_index, 1])
                                    + k.get_unchecked([test_index, 3, trial_index]) * trial_normals[trial_cell_i].get_unchecked([trial_index, 2])
                                )
                                    * test_integrand
                                    * trial_integrands.get_unchecked(trial_index);
                            }
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

#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular_correction<'a, const NPTS_TEST: usize, const NPTS_TRIAL: usize>(
    shape: [usize; 2],
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
    cell_pairs: &[(usize, usize)],
    trial_points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    trial_weights: &[f64],
    test_points: &Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    test_weights: &[f64],
    trial_table: &Array<f64, BaseArray<f64, VectorContainer<f64>, 4>, 4>,
    test_table: &Array<f64, BaseArray<f64, VectorContainer<f64>, 4>, 4>,
) -> SparseMatrixData<f64> {
    let mut output = SparseMatrixData::<f64>::new_known_size(
        shape,
        cell_pairs.len() * trial_space.element().dim() * test_space.element().dim(),
    );
    debug_assert!(test_weights.len() == NPTS_TEST);
    debug_assert!(test_points.shape()[0] == NPTS_TEST);
    debug_assert!(trial_weights.len() == NPTS_TRIAL);
    debug_assert!(trial_points.shape()[0] == NPTS_TRIAL);

    let grid = test_space.grid();

    let mut k = rlst_dynamic_array3!(f64, [NPTS_TEST, 4, NPTS_TRIAL]);
    let mut test_jdet = [0.0; NPTS_TEST];
    let mut test_mapped_pts = rlst_dynamic_array2!(f64, [NPTS_TEST, 3]);
    let mut test_normals = rlst_dynamic_array2!(f64, [NPTS_TEST, 3]);

    let trial_element = grid.geometry().element(cell_pairs[0].0);
    let test_element = grid.geometry().element(cell_pairs[0].1);

    let test_evaluator = grid.geometry().get_evaluator(test_element, test_points);
    let trial_evaluator = grid.geometry().get_evaluator(trial_element, trial_points);

    let mut trial_jdet = [0.0; NPTS_TRIAL];
    let mut trial_mapped_pts = rlst_dynamic_array2!(f64, [NPTS_TRIAL, 3]);
    let mut trial_normals = rlst_dynamic_array2!(f64, [NPTS_TRIAL, 3]);

    let mut sum: f64;
    let mut trial_integrands = [0.0; NPTS_TRIAL];

    for (test_cell, trial_cell) in cell_pairs {
        let test_cell_tindex = grid.topology().index_map()[*test_cell];
        let test_cell_gindex = grid.geometry().index_map()[*test_cell];

        test_evaluator.compute_normals_and_jacobian_determinants(
            test_cell_gindex,
            &mut test_normals,
            &mut test_jdet,
        );
        test_evaluator.compute_points(test_cell_gindex, &mut test_mapped_pts);

        let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
        let trial_cell_gindex = grid.geometry().index_map()[*trial_cell];

        trial_evaluator.compute_normals_and_jacobian_determinants(
            trial_cell_gindex,
            &mut trial_normals,
            &mut trial_jdet,
        );
        trial_evaluator.compute_points(trial_cell_gindex, &mut trial_mapped_pts);

        kernel.assemble_st(
            EvalType::ValueDeriv,
            test_mapped_pts.data(),
            trial_mapped_pts.data(),
            k.data_mut(),
        );

        for (test_i, test_dof) in test_space
            .dofmap()
            .cell_dofs(test_cell_tindex)
            .unwrap()
            .iter()
            .enumerate()
        {
            for (trial_i, trial_dof) in trial_space
                .dofmap()
                .cell_dofs(trial_cell_tindex)
                .unwrap()
                .iter()
                .enumerate()
            {
                for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                    unsafe {
                        trial_integrands[trial_index] = trial_wt
                            * trial_jdet[trial_index]
                            * trial_table.get_unchecked([0, trial_index, trial_i, 0]);
                    }
                }
                sum = 0.0;
                for (test_index, test_wt) in test_weights.iter().enumerate() {
                    let test_integrand = unsafe {
                        test_wt
                            * test_jdet[test_index]
                            * test_table.get_unchecked([0, test_index, test_i, 0])
                    };
                    for trial_index in 0..NPTS_TRIAL {
                        unsafe {
                        sum += (
                            k.get_unchecked([test_index, 1, trial_index]) * trial_normals.get_unchecked([trial_index, 0])
                            + k.get_unchecked([test_index, 2, trial_index]) * trial_normals.get_unchecked([trial_index, 1])
                            + k.get_unchecked([test_index, 3, trial_index]) * trial_normals.get_unchecked([trial_index, 2])
                        )
                                * test_integrand
                                * trial_integrands.get_unchecked(trial_index);
                        }
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

pub fn assemble<'a, const BLOCKSIZE: usize>(
    output: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    let test_colouring = test_space.compute_cell_colouring();
    let trial_colouring = trial_space.compute_cell_colouring();

    assemble_nonsingular::<16, 16, BLOCKSIZE>(
        output,
        kernel,
        trial_space,
        test_space,
        &trial_colouring,
        &test_colouring,
    );
    assemble_singular_into_dense::<4, BLOCKSIZE>(output, kernel, trial_space, test_space);
    // assemble_singular_into_dense::<1, BLOCKSIZE>(output, kernel, trial_space, test_space);
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_nonsingular<
    'a,
    const NPTS_TEST: usize,
    const NPTS_TRIAL: usize,
    const BLOCKSIZE: usize,
>(
    output: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
    trial_colouring: &Vec<Vec<usize>>,
    test_colouring: &Vec<Vec<usize>>,
) {
    if !trial_space.is_serial() || !test_space.is_serial() {
        panic!("Dense assembly can only be used for function spaces stored in serial");
    }
    if output.shape()[0] != test_space.dofmap().global_size()
        || output.shape()[1] != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    // TODO: pass cell types into this function
    let qrule_test = simplex_rule(ReferenceCellType::Triangle, NPTS_TEST).unwrap();
    let mut qpoints_test = rlst_dynamic_array2!(f64, [NPTS_TEST, 2]);
    for i in 0..NPTS_TEST {
        for j in 0..2 {
            *qpoints_test.get_mut([i, j]).unwrap() = qrule_test.points[2 * i + j];
        }
    }
    let qweights_test = qrule_test.weights;
    let qrule_trial = simplex_rule(ReferenceCellType::Triangle, NPTS_TRIAL).unwrap();
    let mut qpoints_trial = rlst_dynamic_array2!(f64, [NPTS_TRIAL, 2]);
    for i in 0..NPTS_TRIAL {
        for j in 0..2 {
            *qpoints_trial.get_mut([i, j]).unwrap() = qrule_trial.points[2 * i + j];
        }
    }
    let qweights_trial = qrule_trial.weights;

    let mut test_table =
        rlst_dynamic_array4!(f64, test_space.element().tabulate_array_shape(0, NPTS_TEST));
    test_space
        .element()
        .tabulate(&qpoints_test, 0, &mut test_table);

    let mut trial_table = rlst_dynamic_array4!(
        f64,
        trial_space.element().tabulate_array_shape(0, NPTS_TRIAL)
    );
    trial_space
        .element()
        .tabulate(&qpoints_test, 0, &mut trial_table);

    let output_raw = RawData2D {
        data: output.data_mut().as_mut_ptr(),
        shape: output.shape(),
    };

    for test_c in test_colouring {
        for trial_c in trial_colouring {
            let mut test_cells: Vec<&[usize]> = vec![];
            let mut trial_cells: Vec<&[usize]> = vec![];

            let mut test_start = 0;
            while test_start < test_c.len() {
                let test_end = if test_start + BLOCKSIZE < test_c.len() {
                    test_start + BLOCKSIZE
                } else {
                    test_c.len()
                };

                let mut trial_start = 0;
                while trial_start < trial_c.len() {
                    let trial_end = if trial_start + BLOCKSIZE < trial_c.len() {
                        trial_start + BLOCKSIZE
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
                    assemble_batch_nonadjacent::<NPTS_TEST, NPTS_TRIAL>(
                        &output_raw,
                        kernel,
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

pub fn assemble_singular_into_dense<'a, const QDEGREE: usize, const BLOCKSIZE: usize>(
    output: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    let sparse_matrix =
        assemble_singular::<QDEGREE, BLOCKSIZE>(output.shape(), kernel, trial_space, test_space);
    let data = sparse_matrix.data;
    let rows = sparse_matrix.rows;
    let cols = sparse_matrix.cols;
    for ((i, j), value) in rows.iter().zip(cols.iter()).zip(data.iter()) {
        *output.get_mut([*i, *j]).unwrap() += *value;
    }
}

pub fn assemble_singular_into_csr<'a, const QDEGREE: usize, const BLOCKSIZE: usize>(
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) -> CsrMatrix<f64> {
    let shape = [
        test_space.dofmap().global_size(),
        trial_space.dofmap().global_size(),
    ];
    let sparse_matrix =
        assemble_singular::<QDEGREE, BLOCKSIZE>(shape, kernel, trial_space, test_space);

    CsrMatrix::<f64>::from_aij(
        sparse_matrix.shape,
        &sparse_matrix.rows,
        &sparse_matrix.cols,
        &sparse_matrix.data,
    )
    .unwrap()
}

fn assemble_singular<'a, const QDEGREE: usize, const BLOCKSIZE: usize>(
    shape: [usize; 2],
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) -> SparseMatrixData<f64> {
    let mut output = SparseMatrixData::new(shape);

    if test_space.grid() != trial_space.grid() {
        // If the test and trial grids are different, there are no neighbouring triangles
        return output;
    }

    if !trial_space.is_serial() || !test_space.is_serial() {
        panic!("Dense assembly can only be used for function spaces stored in serial");
    }
    if shape[0] != test_space.dofmap().global_size()
        || shape[1] != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    let grid = test_space.grid();
    let c20 = grid.topology().connectivity(2, 0);

    let mut possible_pairs = vec![];
    // Vertex-adjacent
    for i in 0..3 {
        for j in 0..3 {
            possible_pairs.push(vec![(i, j)]);
        }
    }
    // edge-adjacent
    for i in 0..3 {
        for j in i + 1..3 {
            for k in 0..3 {
                for l in 0..3 {
                    if k != l {
                        possible_pairs.push(vec![(k, i), (l, j)]);
                    }
                }
            }
        }
    }
    // Same cell
    possible_pairs.push(vec![(0, 0), (1, 1), (2, 2)]);

    let mut qweights = vec![];
    let mut trial_points = vec![];
    let mut test_points = vec![];
    let mut trial_tables = vec![];
    let mut test_tables = vec![];
    for pairs in &possible_pairs {
        let qrule = get_quadrature_rule(
            ReferenceCellType::Triangle,
            ReferenceCellType::Triangle,
            pairs,
            QDEGREE,
        );
        let npts = qrule.weights.len();

        let mut points = rlst_dynamic_array2!(f64, [npts, 2]);
        for i in 0..npts {
            for j in 0..2 {
                *points.get_mut([i, j]).unwrap() = qrule.trial_points[2 * i + j];
            }
        }
        let mut table = rlst_dynamic_array4!(
            f64,
            trial_space
                .element()
                .tabulate_array_shape(0, points.shape()[0])
        );
        trial_space.element().tabulate(&points, 0, &mut table);
        trial_points.push(points);
        trial_tables.push(table);

        let mut points = rlst_dynamic_array2!(f64, [npts, 2]);
        for i in 0..npts {
            for j in 0..2 {
                *points.get_mut([i, j]).unwrap() = qrule.test_points[2 * i + j];
            }
        }
        let mut table = rlst_dynamic_array4!(
            f64,
            test_space
                .element()
                .tabulate_array_shape(0, points.shape()[0])
        );
        test_space.element().tabulate(&points, 0, &mut table);
        test_points.push(points);
        test_tables.push(table);
        qweights.push(qrule.weights);
    }
    let mut cell_pairs: Vec<Vec<(usize, usize)>> = vec![vec![]; possible_pairs.len()];
    for test_cell in 0..grid.topology().entity_count(grid.topology().dim()) {
        let test_cell_tindex = grid.topology().index_map()[test_cell];
        let test_vertices = c20.row(test_cell_tindex).unwrap();
        for trial_cell in 0..grid.topology().entity_count(grid.topology().dim()) {
            let trial_cell_tindex = grid.topology().index_map()[trial_cell];
            let trial_vertices = c20.row(trial_cell_tindex).unwrap();

            let mut pairs = vec![];
            for (trial_i, trial_v) in trial_vertices.iter().enumerate() {
                for (test_i, test_v) in test_vertices.iter().enumerate() {
                    if test_v == trial_v {
                        pairs.push((test_i, trial_i));
                    }
                }
            }
            if !pairs.is_empty() {
                cell_pairs[possible_pairs.iter().position(|r| *r == pairs).unwrap()]
                    .push((test_cell, trial_cell))
            }
        }
    }
    for (i, cells) in cell_pairs.iter().enumerate() {
        let mut start = 0;
        let mut cell_blocks = vec![];
        while start < cells.len() {
            let end = if start + BLOCKSIZE < cells.len() {
                start + BLOCKSIZE
            } else {
                cells.len()
            };
            cell_blocks.push(&cells[start..end]);
            start = end;
        }

        let numtasks = cell_blocks.len();
        let r: SparseMatrixData<f64> = (0..numtasks)
            .into_par_iter()
            .map(&|t| {
                assemble_batch_singular(
                    shape,
                    kernel,
                    trial_space,
                    test_space,
                    cell_blocks[t],
                    &trial_points[i],
                    &test_points[i],
                    &qweights[i],
                    &trial_tables[i],
                    &test_tables[i],
                )
            })
            .reduce(|| SparseMatrixData::<f64>::new(shape), |a, b| a.sum(b));

        output.add(r);
    }
    output
}

pub fn assemble_singular_correction_into_dense<
    'a,
    const NPTS_TEST: usize,
    const NPTS_TRIAL: usize,
    const BLOCKSIZE: usize,
>(
    output: &mut Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    let sparse_matrix = assemble_singular_correction::<NPTS_TEST, NPTS_TRIAL, BLOCKSIZE>(
        output.shape(),
        kernel,
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

pub fn assemble_singular_correction_into_csr<
    'a,
    const NPTS_TEST: usize,
    const NPTS_TRIAL: usize,
    const BLOCKSIZE: usize,
>(
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) -> CsrMatrix<f64> {
    let shape = [
        test_space.dofmap().global_size(),
        trial_space.dofmap().global_size(),
    ];
    let sparse_matrix = assemble_singular_correction::<NPTS_TEST, NPTS_TRIAL, BLOCKSIZE>(
        shape,
        kernel,
        trial_space,
        test_space,
    );

    CsrMatrix::<f64>::from_aij(
        sparse_matrix.shape,
        &sparse_matrix.rows,
        &sparse_matrix.cols,
        &sparse_matrix.data,
    )
    .unwrap()
}

fn assemble_singular_correction<
    'a,
    const NPTS_TEST: usize,
    const NPTS_TRIAL: usize,
    const BLOCKSIZE: usize,
>(
    shape: [usize; 2],
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) -> SparseMatrixData<f64> {
    if !trial_space.is_serial() || !test_space.is_serial() {
        panic!("Dense assembly can only be used for function spaces stored in serial");
    }
    if shape[0] != test_space.dofmap().global_size()
        || shape[1] != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    if NPTS_TEST != NPTS_TRIAL {
        panic!("FMM with different test and trial quadrature rules not yet supported");
    }

    let grid = test_space.grid();
    let c20 = grid.topology().connectivity(2, 0);

    // TODO: pass cell types into this function
    let qrule_test = simplex_rule(ReferenceCellType::Triangle, NPTS_TEST).unwrap();
    let mut qpoints_test = rlst_dynamic_array2!(f64, [NPTS_TEST, 2]);
    for i in 0..NPTS_TEST {
        for j in 0..2 {
            *qpoints_test.get_mut([i, j]).unwrap() = qrule_test.points[2 * i + j];
        }
    }
    let qweights_test = qrule_test.weights;
    let qrule_trial = simplex_rule(ReferenceCellType::Triangle, NPTS_TRIAL).unwrap();
    let mut qpoints_trial = rlst_dynamic_array2!(f64, [NPTS_TRIAL, 2]);
    for i in 0..NPTS_TRIAL {
        for j in 0..2 {
            *qpoints_trial.get_mut([i, j]).unwrap() = qrule_trial.points[2 * i + j];
        }
    }
    let qweights_trial = qrule_trial.weights;

    let mut test_table =
        rlst_dynamic_array4!(f64, test_space.element().tabulate_array_shape(0, NPTS_TEST));
    test_space
        .element()
        .tabulate(&qpoints_test, 0, &mut test_table);

    let mut trial_table = rlst_dynamic_array4!(
        f64,
        trial_space.element().tabulate_array_shape(0, NPTS_TRIAL)
    );
    trial_space
        .element()
        .tabulate(&qpoints_test, 0, &mut trial_table);

    let mut cell_pairs: Vec<(usize, usize)> = vec![];
    for test_cell in 0..grid.topology().entity_count(grid.topology().dim()) {
        let test_cell_tindex = grid.topology().index_map()[test_cell];
        let test_vertices = c20.row(test_cell_tindex).unwrap();
        for trial_cell in 0..grid.topology().entity_count(grid.topology().dim()) {
            let trial_cell_tindex = grid.topology().index_map()[trial_cell];
            let trial_vertices = c20.row(trial_cell_tindex).unwrap();

            let mut pairs = vec![];
            for (trial_i, trial_v) in trial_vertices.iter().enumerate() {
                for (test_i, test_v) in test_vertices.iter().enumerate() {
                    if test_v == trial_v {
                        pairs.push((test_i, trial_i));
                    }
                }
            }
            if !pairs.is_empty() {
                cell_pairs.push((test_cell, trial_cell))
            }
        }
    }

    let mut start = 0;
    let mut cell_blocks = vec![];
    while start < cell_pairs.len() {
        let end = if start + BLOCKSIZE < cell_pairs.len() {
            start + BLOCKSIZE
        } else {
            cell_pairs.len()
        };
        cell_blocks.push(&cell_pairs[start..end]);
        start = end;
    }

    let numtasks = cell_blocks.len();
    (0..numtasks)
        .into_par_iter()
        .map(&|t| {
            assemble_batch_singular_correction::<NPTS_TEST, NPTS_TRIAL>(
                shape,
                kernel,
                trial_space,
                test_space,
                cell_blocks[t],
                &qpoints_trial,
                &qweights_trial,
                &qpoints_test,
                &qweights_test,
                &trial_table,
                &test_table,
            )
        })
        .reduce(|| SparseMatrixData::<f64>::new(shape), |a, b| a.sum(b))
}

#[cfg(test)]
mod test {
    use crate::assembly::batched_double_layer::*;
    use crate::function_space::SerialFunctionSpace;
    use approx::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::{Continuity, ElementFamily};
    use rlst_dense::traits::RandomAccessByRef;

    #[test]
    fn test_singular_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        assemble_singular_into_dense::<4, 128>(
            &mut matrix,
            &Laplace3dKernel::new(),
            &space,
            &space,
        );
        let csr = assemble_singular_into_csr::<4, 128>(&Laplace3dKernel::new(), &space, &space);

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
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs, ndofs]);
        assemble_singular_into_dense::<4, 128>(
            &mut matrix,
            &Laplace3dKernel::new(),
            &space,
            &space,
        );
        let csr = assemble_singular_into_csr::<4, 128>(&Laplace3dKernel::new(), &space, &space);

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
        let grid = regular_sphere(0);
        let element0 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let element1 = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space0 = SerialFunctionSpace::new(&grid, &element0);
        let space1 = SerialFunctionSpace::new(&grid, &element1);

        let ndofs0 = space0.dofmap().global_size();
        let ndofs1 = space1.dofmap().global_size();

        let mut matrix = rlst_dynamic_array2!(f64, [ndofs1, ndofs0]);
        assemble_singular_into_dense::<4, 128>(
            &mut matrix,
            &Laplace3dKernel::new(),
            &space0,
            &space1,
        );
        let csr = assemble_singular_into_csr::<4, 128>(&Laplace3dKernel::new(), &space0, &space1);
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
