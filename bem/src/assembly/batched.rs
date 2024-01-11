use crate::function_space::SerialFunctionSpace;
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_tools::arrays::{transpose_to_matrix, zero_matrix, Array4D, Mat};
use bempp_traits::arrays::AdjacencyListAccess;
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};
use bempp_traits::kernel::Kernel;
use bempp_traits::types::EvalType;
use bempp_traits::types::Scalar;
use rayon::prelude::*;
use rlst_dense::rlst_dynamic_array4;
use rlst_dense::traits::{
    RandomAccessByRef, RawAccess, RawAccessMut, Shape, UnsafeRandomAccessByRef,
};

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

pub struct RawData2D<T: Scalar> {
    pub data: *mut T,
    pub shape: [usize; 2],
}

unsafe impl<T: Scalar> Sync for RawData2D<T> {}

// TODO: use T not f64
#[allow(clippy::too_many_arguments)]
fn assemble_batch_singular<'a>(
    output: &RawData2D<f64>,
    kernel: &impl Kernel<T = f64>,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
    cell_pairs: &[(usize, usize)],
    trial_points: &Mat<f64>,
    test_points: &Mat<f64>,
    weights: &[f64],
    trial_table: &Array4D<f64>,
    test_table: &Array4D<f64>,
) -> usize {
    let grid = test_space.grid();
    let mut k = vec![0.0];

    // Memory assignment to be moved elsewhere as passed into here mutable?
    let mut test_jdet = vec![0.0; test_points.shape()[0]];
    let mut trial_jdet = vec![0.0; trial_points.shape()[0]];
    let mut test_mapped_pts = zero_matrix([test_points.shape()[0], 3]);
    let mut trial_mapped_pts = zero_matrix([trial_points.shape()[0], 3]);
    let mut test_normals = zero_matrix([test_points.shape()[0], 3]);
    let mut trial_normals = zero_matrix([trial_points.shape()[0], 3]);

    for (test_cell, trial_cell) in cell_pairs {
        let test_cell_tindex = grid.topology().index_map()[*test_cell];
        let test_cell_gindex = grid.geometry().index_map()[*test_cell];
        let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
        let trial_cell_gindex = grid.geometry().index_map()[*trial_cell];

        grid.geometry().compute_jacobian_determinants(
            test_points,
            test_cell_gindex,
            &mut test_jdet,
        );
        grid.geometry()
            .compute_points(test_points, test_cell_gindex, &mut test_mapped_pts);
        if needs_test_normal {
            grid.geometry()
                .compute_normals(test_points, test_cell_gindex, &mut test_normals);
        }

        grid.geometry().compute_jacobian_determinants(
            trial_points,
            trial_cell_gindex,
            &mut trial_jdet,
        );
        grid.geometry()
            .compute_points(trial_points, trial_cell_gindex, &mut trial_mapped_pts);
        if needs_trial_normal {
            grid.geometry()
                .compute_normals(trial_points, trial_cell_gindex, &mut trial_normals);
        }

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
                    let mut test_row = vec![0.0; test_mapped_pts.shape()[1]];
                    for (i, ti) in test_row.iter_mut().enumerate() {
                        *ti = *test_mapped_pts.get([index, i]).unwrap();
                    }
                    let mut trial_row = vec![0.0; trial_mapped_pts.shape()[1]];
                    for (i, ti) in trial_row.iter_mut().enumerate() {
                        *ti = *trial_mapped_pts.get([index, i]).unwrap();
                    }

                    kernel.assemble_st(EvalType::Value, &test_row, &trial_row, &mut k);
                    sum += k[0]
                        * (wt
                            * test_table.get([0, index, test_i, 0]).unwrap()
                            * test_jdet[index]
                            * trial_table.get([0, index, trial_i, 0]).unwrap()
                            * trial_jdet[index]);
                }
                unsafe {
                    *output.data.add(*test_dof + output.shape[0] * *trial_dof) += sum;
                }
            }
        }
    }
    1
}

#[allow(clippy::too_many_arguments)]
fn assemble_batch_nonadjacent<'a, const NPTS_TEST: usize, const NPTS_TRIAL: usize>(
    output: &RawData2D<f64>,
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    trial_cells: &[usize],
    test_space: &SerialFunctionSpace<'a>,
    test_cells: &[usize],
    trial_points: &Mat<f64>,
    trial_weights: &[f64],
    test_points: &Mat<f64>,
    test_weights: &[f64],
    trial_table: &Array4D<f64>,
    test_table: &Array4D<f64>,
) -> usize {
    debug_assert!(test_weights.len() == NPTS_TEST);
    debug_assert!(test_points.shape()[0] == NPTS_TEST);
    debug_assert!(trial_weights.len() == NPTS_TRIAL);
    debug_assert!(trial_points.shape()[0] == NPTS_TRIAL);

    let test_grid = test_space.grid();
    let test_c20 = test_grid.topology().connectivity(2, 0);
    let trial_grid = trial_space.grid();
    let trial_c20 = trial_grid.topology().connectivity(2, 0);

    let mut k = vec![0.0; NPTS_TEST * NPTS_TRIAL];
    let mut test_jdet = [0.0; NPTS_TEST];
    let mut test_normals = zero_matrix([NPTS_TEST, 3]);

    let mut test_mapped_pts = rlst_dense::rlst_dynamic_array2![f64, [NPTS_TEST, 3]];

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
        trial_mapped_pts.push(zero_matrix([NPTS_TRIAL, 3]));
        trial_normals.push(zero_matrix([NPTS_TRIAL, 3]));
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
                EvalType::Value,
                test_mapped_pts.data(),
                trial_mapped_pts[trial_cell_i].data(),
                &mut k,
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
                            sum += k[test_index * trial_weights.len() + trial_index]
                                * test_integrand
                                * trial_integrands[trial_index];
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

pub fn assemble<'a>(
    output: &mut Mat<f64>,
    kernel: &impl Kernel<T = f64>,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    let test_colouring = test_space.compute_cell_colouring();
    let trial_colouring = trial_space.compute_cell_colouring();
    // TODO: make these configurable
    let blocksize = 128;

    assemble_nonsingular::<16, 16>(
        output,
        kernel,
        trial_space,
        test_space,
        &trial_colouring,
        &test_colouring,
        blocksize,
    );
    assemble_singular(
        output,
        kernel,
        needs_trial_normal,
        needs_test_normal,
        trial_space,
        test_space,
        &trial_colouring,
        &test_colouring,
        blocksize,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_nonsingular<'a, const NPTS_TEST: usize, const NPTS_TRIAL: usize>(
    output: &mut Mat<f64>,
    kernel: &impl Kernel<T = f64>,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
    trial_colouring: &Vec<Vec<usize>>,
    test_colouring: &Vec<Vec<usize>>,
    blocksize: usize,
) {
    if !trial_space.is_serial() || !test_space.is_serial() {
        panic!("Dense assemble can only be used for function spaces stored in serial");
    }
    if output.shape()[0] != test_space.dofmap().global_size()
        || output.shape()[1] != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    // Size of this might not be known at compile time
    // let test_dofs_per_cell = 1;
    // let trial_dofs_per_cell = 1;

    // TODO: pass cell types into this function
    let qrule_test = simplex_rule(ReferenceCellType::Triangle, NPTS_TEST).unwrap();
    let qpoints_test = transpose_to_matrix(&qrule_test.points, [NPTS_TEST, 2]);
    let qweights_test = qrule_test.weights;
    let qrule_trial = simplex_rule(ReferenceCellType::Triangle, NPTS_TRIAL).unwrap();
    let qpoints_trial = transpose_to_matrix(&qrule_trial.points, [NPTS_TRIAL, 2]);
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
                let test_end = if test_start + blocksize < test_c.len() {
                    test_start + blocksize
                } else {
                    test_c.len()
                };

                let mut trial_start = 0;
                while trial_start < trial_c.len() {
                    let trial_end = if trial_start + blocksize < trial_c.len() {
                        trial_start + blocksize
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

#[allow(clippy::too_many_arguments)]
pub fn assemble_singular<'a>(
    output: &mut Mat<f64>,
    kernel: &impl Kernel<T = f64>,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
    trial_colouring: &Vec<Vec<usize>>,
    test_colouring: &Vec<Vec<usize>>,
    blocksize: usize,
) {
    if test_space.grid() != trial_space.grid() {
        // If the test and trial grids are different, there are no neighbouring triangles
        return;
    }

    if !trial_space.is_serial() || !test_space.is_serial() {
        panic!("Dense assemble can only be used for function spaces stored in serial");
    }
    if output.shape()[0] != test_space.dofmap().global_size()
        || output.shape()[1] != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    // Size of this might not be known at compile time
    // let test_dofs_per_cell = 1;
    // let trial_dofs_per_cell = 1;

    // TODO: allow user to configure this
    let npoints = 4;

    let output_raw = RawData2D {
        data: output.data_mut().as_mut_ptr(),
        shape: output.shape(),
    };

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
                        possible_pairs.push(vec![(i, k), (j, l)]);
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
            npoints,
        );

        let points = transpose_to_matrix(&qrule.trial_points, [qrule.npoints, 2]);
        let mut table = rlst_dynamic_array4!(
            f64,
            trial_space
                .element()
                .tabulate_array_shape(0, points.shape()[0])
        );
        trial_space.element().tabulate(&points, 0, &mut table);
        trial_points.push(points);
        trial_tables.push(table);

        let points = transpose_to_matrix(&qrule.test_points, [qrule.npoints, 2]);
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

    for test_c in test_colouring {
        for trial_c in trial_colouring {
            let mut cell_pairs: Vec<Vec<(usize, usize)>> = vec![vec![]; possible_pairs.len()];
            for test_cell in test_c {
                let test_cell_tindex = grid.topology().index_map()[*test_cell];
                let test_vertices = c20.row(test_cell_tindex).unwrap();
                for trial_cell in trial_c {
                    let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
                    let trial_vertices = c20.row(trial_cell_tindex).unwrap();

                    let mut pairs = vec![];
                    for (test_i, test_v) in test_vertices.iter().enumerate() {
                        for (trial_i, trial_v) in trial_vertices.iter().enumerate() {
                            if test_v == trial_v {
                                pairs.push((test_i, trial_i));
                            }
                        }
                    }
                    if !pairs.is_empty() {
                        cell_pairs[possible_pairs.iter().position(|r| *r == pairs).unwrap()]
                            .push((*trial_cell, *test_cell))
                    }
                }
            }
            for (i, cells) in cell_pairs.iter().enumerate() {
                let mut start = 0;
                let mut cell_blocks = vec![];
                while start < cells.len() {
                    let end = if start + blocksize < cells.len() {
                        start + blocksize
                    } else {
                        cells.len()
                    };
                    cell_blocks.push(&cells[start..end]);
                    start = end;
                }

                let numtasks = cell_blocks.len();
                let r: usize = (0..numtasks)
                    .into_par_iter()
                    .map(&|t| {
                        assemble_batch_singular(
                            &output_raw,
                            kernel,
                            needs_trial_normal,
                            needs_test_normal,
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
                    .sum();
                assert_eq!(r, numtasks);
            }
        }
    }
}
#[cfg(test)]
mod test {
    use crate::assembly::batched::*;
    use crate::function_space::SerialFunctionSpace;
    use approx::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::{Continuity, ElementFamily};

    #[test]
    fn test_laplace_single_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = zero_matrix::<f64>([ndofs, ndofs]);
        assemble(
            &mut matrix,
            &Laplace3dKernel::new(),
            false,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472], vec![0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548], vec![0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473], vec![0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074], vec![0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074], vec![0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472], vec![0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074], vec![0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487]];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(*matrix.get([i, j]).unwrap(), entry, epsilon = 1e-3);
            }
        }
    }
    /*

    #[test]
    fn test_laplace_double_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new([ndofs, ndofs]);
        assemble(
            &mut matrix,
            &green::LaplaceGreenDyKernel {},
            true,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![-1.9658941517361406e-33, -0.08477786720045567, -0.048343860959178774, -0.08477786720045567, -0.08477786720045566, -0.048343860959178774, -0.033625570841778946, -0.04834386095917877], vec![-0.08477786720045567, -1.9658941517361406e-33, -0.08477786720045567, -0.048343860959178774, -0.04834386095917877, -0.08477786720045566, -0.048343860959178774, -0.033625570841778946], vec![-0.048343860959178774, -0.08477786720045567, -1.9658941517361406e-33, -0.08477786720045567, -0.033625570841778946, -0.04834386095917877, -0.08477786720045566, -0.048343860959178774], vec![-0.08477786720045567, -0.048343860959178774, -0.08477786720045567, -1.9658941517361406e-33, -0.048343860959178774, -0.033625570841778946, -0.04834386095917877, -0.08477786720045566], vec![-0.08477786720045566, -0.04834386095917877, -0.033625570841778946, -0.04834386095917877, 4.910045345075783e-33, -0.08477786720045566, -0.048343860959178774, -0.08477786720045566], vec![-0.04834386095917877, -0.08477786720045566, -0.04834386095917877, -0.033625570841778946, -0.08477786720045566, 4.910045345075783e-33, -0.08477786720045566, -0.048343860959178774], vec![-0.033625570841778946, -0.04834386095917877, -0.08477786720045566, -0.04834386095917877, -0.048343860959178774, -0.08477786720045566, 4.910045345075783e-33, -0.08477786720045566], vec![-0.04834386095917877, -0.033625570841778946, -0.04834386095917877, -0.08477786720045566, -0.08477786720045566, -0.048343860959178774, -0.08477786720045566, 4.910045345075783e-33]];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), entry, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_laplace_adjoint_double_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new([ndofs, ndofs]);
        assemble(
            &mut matrix,
            &green::LaplaceGreenDxKernel {},
            false,
            true,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![1.9658941517361406e-33, -0.08478435261011981, -0.048343860959178774, -0.0847843526101198, -0.08478435261011981, -0.04834386095917877, -0.033625570841778946, -0.048343860959178774], vec![-0.0847843526101198, 1.9658941517361406e-33, -0.08478435261011981, -0.048343860959178774, -0.048343860959178774, -0.08478435261011981, -0.04834386095917877, -0.033625570841778946], vec![-0.048343860959178774, -0.0847843526101198, 1.9658941517361406e-33, -0.08478435261011981, -0.033625570841778946, -0.048343860959178774, -0.08478435261011981, -0.04834386095917877], vec![-0.08478435261011981, -0.048343860959178774, -0.0847843526101198, 1.9658941517361406e-33, -0.04834386095917877, -0.033625570841778946, -0.048343860959178774, -0.08478435261011981], vec![-0.0847843526101198, -0.04834386095917877, -0.033625570841778946, -0.04834386095917877, -4.910045345075783e-33, -0.0847843526101198, -0.048343860959178774, -0.08478435261011981], vec![-0.04834386095917877, -0.0847843526101198, -0.04834386095917877, -0.033625570841778946, -0.08478435261011981, -4.910045345075783e-33, -0.0847843526101198, -0.048343860959178774], vec![-0.033625570841778946, -0.04834386095917877, -0.0847843526101198, -0.04834386095917877, -0.048343860959178774, -0.08478435261011981, -4.910045345075783e-33, -0.0847843526101198], vec![-0.04834386095917877, -0.033625570841778946, -0.04834386095917877, -0.0847843526101198, -0.0847843526101198, -0.048343860959178774, -0.08478435261011981, -4.910045345075783e-33]];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), entry, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_laplace_hypersingular_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new([ndofs, ndofs]);
        laplace_hypersingular_assemble(&mut matrix, &space, &space);

        for i in 0..ndofs {
            for j in 0..ndofs {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), 0.0, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_laplace_hypersingular_p1_p1() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new([ndofs, ndofs]);

        laplace_hypersingular_assemble(&mut matrix, &space, &space);

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![0.33550642155494004, -0.10892459915262698, -0.05664545560057827, -0.05664545560057828, -0.0566454556005783, -0.05664545560057828], vec![-0.10892459915262698, 0.33550642155494004, -0.05664545560057828, -0.05664545560057827, -0.05664545560057828, -0.05664545560057829], vec![-0.05664545560057828, -0.05664545560057827, 0.33550642155494004, -0.10892459915262698, -0.056645455600578286, -0.05664545560057829], vec![-0.05664545560057827, -0.05664545560057828, -0.10892459915262698, 0.33550642155494004, -0.05664545560057828, -0.056645455600578286], vec![-0.05664545560057829, -0.0566454556005783, -0.05664545560057829, -0.05664545560057829, 0.33550642155494004, -0.10892459915262698], vec![-0.05664545560057829, -0.05664545560057831, -0.05664545560057829, -0.05664545560057829, -0.10892459915262698, 0.33550642155494004]];

        let perm = [0, 5, 2, 4, 3, 1];

        for (i, pi) in perm.iter().enumerate() {
            for (j, pj) in perm.iter().enumerate() {
                assert_relative_eq!(
                    *matrix.get(i, j).unwrap(),
                    from_cl[*pi][*pj],
                    epsilon = 1e-4
                );
            }
        }
    }

    #[test]
    fn test_helmholtz_single_layer_real_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new([ndofs, ndofs]);
        assemble(
            &mut matrix,
            &green::HelmholtzGreenKernel { k: 3.0 },
            false,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![0.08742460357596939, -0.02332791148192136, -0.04211947809894265, -0.02332791148192136, -0.023327911481921364, -0.042119478098942634, -0.03447046598405515, -0.04211947809894265], vec![-0.023327911481921364, 0.08742460357596939, -0.02332791148192136, -0.04211947809894265, -0.04211947809894265, -0.02332791148192136, -0.042119478098942634, -0.03447046598405515], vec![-0.04211947809894265, -0.02332791148192136, 0.08742460357596939, -0.02332791148192136, -0.03447046598405515, -0.04211947809894265, -0.023327911481921364, -0.042119478098942634], vec![-0.02332791148192136, -0.04211947809894265, -0.023327911481921364, 0.08742460357596939, -0.042119478098942634, -0.03447046598405515, -0.04211947809894265, -0.02332791148192136], vec![-0.023327911481921364, -0.04211947809894265, -0.03447046598405515, -0.042119478098942634, 0.08742460357596939, -0.02332791148192136, -0.04211947809894265, -0.023327911481921364], vec![-0.042119478098942634, -0.02332791148192136, -0.04211947809894265, -0.034470465984055156, -0.02332791148192136, 0.08742460357596939, -0.023327911481921364, -0.04211947809894265], vec![-0.03447046598405515, -0.042119478098942634, -0.023327911481921364, -0.04211947809894265, -0.04211947809894265, -0.023327911481921364, 0.08742460357596939, -0.02332791148192136], vec![-0.04211947809894265, -0.034470465984055156, -0.042119478098942634, -0.02332791148192136, -0.023327911481921364, -0.04211947809894265, -0.02332791148192136, 0.08742460357596939]];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), entry, epsilon = 1e-4);
            }
        }
    }
    #[test]
    fn test_helmholtz_single_layer_complex_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new([ndofs, ndofs]);
        assemble(
            &mut matrix,
            &green::HelmholtzGreenKernel { k: 3.0 },
            false,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![Complex::new(0.08742460357596939, 0.11004203436820102), Complex::new(-0.02332791148192136, 0.04919102584271124), Complex::new(-0.04211947809894265, 0.003720159902487029), Complex::new(-0.02332791148192136, 0.04919102584271125), Complex::new(-0.023327911481921364, 0.04919102584271124), Complex::new(-0.042119478098942634, 0.003720159902487025), Complex::new(-0.03447046598405515, -0.02816544680626108), Complex::new(-0.04211947809894265, 0.0037201599024870254)], vec![Complex::new(-0.023327911481921364, 0.04919102584271125), Complex::new(0.08742460357596939, 0.11004203436820104), Complex::new(-0.02332791148192136, 0.04919102584271124), Complex::new(-0.04211947809894265, 0.0037201599024870263), Complex::new(-0.04211947809894265, 0.0037201599024870254), Complex::new(-0.02332791148192136, 0.04919102584271125), Complex::new(-0.042119478098942634, 0.003720159902487025), Complex::new(-0.03447046598405515, -0.028165446806261072)], vec![Complex::new(-0.04211947809894265, 0.003720159902487029), Complex::new(-0.02332791148192136, 0.04919102584271125), Complex::new(0.08742460357596939, 0.11004203436820102), Complex::new(-0.02332791148192136, 0.04919102584271124), Complex::new(-0.03447046598405515, -0.02816544680626108), Complex::new(-0.04211947809894265, 0.0037201599024870254), Complex::new(-0.023327911481921364, 0.04919102584271124), Complex::new(-0.042119478098942634, 0.003720159902487025)], vec![Complex::new(-0.02332791148192136, 0.04919102584271124), Complex::new(-0.04211947809894265, 0.0037201599024870263), Complex::new(-0.023327911481921364, 0.04919102584271125), Complex::new(0.08742460357596939, 0.11004203436820104), Complex::new(-0.042119478098942634, 0.003720159902487025), Complex::new(-0.03447046598405515, -0.028165446806261072), Complex::new(-0.04211947809894265, 0.0037201599024870254), Complex::new(-0.02332791148192136, 0.04919102584271125)], vec![Complex::new(-0.023327911481921364, 0.04919102584271125), Complex::new(-0.04211947809894265, 0.0037201599024870263), Complex::new(-0.03447046598405515, -0.02816544680626108), Complex::new(-0.042119478098942634, 0.003720159902487025), Complex::new(0.08742460357596939, 0.11004203436820104), Complex::new(-0.02332791148192136, 0.04919102584271124), Complex::new(-0.04211947809894265, 0.0037201599024870267), Complex::new(-0.023327911481921364, 0.04919102584271125)], vec![Complex::new(-0.042119478098942634, 0.003720159902487025), Complex::new(-0.02332791148192136, 0.04919102584271125), Complex::new(-0.04211947809894265, 0.0037201599024870263), Complex::new(-0.034470465984055156, -0.028165446806261075), Complex::new(-0.02332791148192136, 0.04919102584271124), Complex::new(0.08742460357596939, 0.11004203436820104), Complex::new(-0.023327911481921364, 0.04919102584271125), Complex::new(-0.04211947809894265, 0.0037201599024870237)], vec![Complex::new(-0.03447046598405515, -0.02816544680626108), Complex::new(-0.042119478098942634, 0.003720159902487025), Complex::new(-0.023327911481921364, 0.04919102584271125), Complex::new(-0.04211947809894265, 0.0037201599024870263), Complex::new(-0.04211947809894265, 0.0037201599024870267), Complex::new(-0.023327911481921364, 0.04919102584271125), Complex::new(0.08742460357596939, 0.11004203436820104), Complex::new(-0.02332791148192136, 0.04919102584271124)], vec![Complex::new(-0.04211947809894265, 0.0037201599024870263), Complex::new(-0.034470465984055156, -0.028165446806261075), Complex::new(-0.042119478098942634, 0.003720159902487025), Complex::new(-0.02332791148192136, 0.04919102584271125), Complex::new(-0.023327911481921364, 0.04919102584271125), Complex::new(-0.04211947809894265, 0.0037201599024870237), Complex::new(-0.02332791148192136, 0.04919102584271124), Complex::new(0.08742460357596939, 0.11004203436820104)]];
        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(matrix.get(i, j).unwrap().re, entry.re, epsilon = 1e-4);
                assert_relative_eq!(matrix.get(i, j).unwrap().im, entry.im, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_helmholtz_double_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new([ndofs, ndofs]);
        assemble(
            &mut matrix,
            &green::HelmholtzGreenDyKernel { k: 3.0 },
            true,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![Complex::new(-1.025266688854119e-33, -7.550086433767158e-36), Complex::new(-0.07902626473768169, -0.08184681047051735), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(-0.07902626473768172, -0.08184681047051737), Complex::new(-0.07902626473768169, -0.08184681047051737), Complex::new(0.01906923918000323, -0.10276858786959302), Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.019069239180003215, -0.10276858786959299)], vec![Complex::new(-0.07902626473768172, -0.08184681047051737), Complex::new(-1.025266688854119e-33, 1.0291684702482414e-35), Complex::new(-0.0790262647376817, -0.08184681047051737), Complex::new(0.019069239180003212, -0.10276858786959299), Complex::new(0.019069239180003212, -0.10276858786959298), Complex::new(-0.07902626473768168, -0.08184681047051737), Complex::new(0.01906923918000323, -0.10276858786959299), Complex::new(0.10089706509966115, -0.07681163409722506)], vec![Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(-0.07902626473768172, -0.08184681047051737), Complex::new(-1.025266688854119e-33, -7.550086433767158e-36), Complex::new(-0.07902626473768169, -0.08184681047051735), Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.019069239180003215, -0.10276858786959299), Complex::new(-0.07902626473768169, -0.08184681047051737), Complex::new(0.01906923918000323, -0.10276858786959302)], vec![Complex::new(-0.0790262647376817, -0.08184681047051737), Complex::new(0.019069239180003212, -0.10276858786959299), Complex::new(-0.07902626473768172, -0.08184681047051737), Complex::new(-1.025266688854119e-33, 1.0291684702482414e-35), Complex::new(0.01906923918000323, -0.10276858786959299), Complex::new(0.10089706509966115, -0.07681163409722506), Complex::new(0.019069239180003212, -0.10276858786959298), Complex::new(-0.07902626473768168, -0.08184681047051737)], vec![Complex::new(-0.07902626473768172, -0.08184681047051737), Complex::new(0.019069239180003215, -0.10276858786959298), Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.01906923918000323, -0.10276858786959299), Complex::new(5.00373588753262e-33, -1.8116810507789718e-36), Complex::new(-0.07902626473768169, -0.08184681047051735), Complex::new(0.019069239180003212, -0.10276858786959299), Complex::new(-0.07902626473768169, -0.08184681047051737)], vec![Complex::new(0.019069239180003222, -0.10276858786959299), Complex::new(-0.07902626473768173, -0.08184681047051737), Complex::new(0.01906923918000322, -0.10276858786959299), Complex::new(0.10089706509966115, -0.07681163409722506), Complex::new(-0.07902626473768169, -0.08184681047051735), Complex::new(7.314851820797302e-33, -1.088140415641433e-35), Complex::new(-0.07902626473768169, -0.08184681047051737), Complex::new(0.01906923918000322, -0.10276858786959299)], vec![Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.01906923918000323, -0.10276858786959299), Complex::new(-0.07902626473768172, -0.08184681047051737), Complex::new(0.019069239180003215, -0.10276858786959298), Complex::new(0.019069239180003212, -0.10276858786959299), Complex::new(-0.07902626473768169, -0.08184681047051737), Complex::new(5.00373588753262e-33, -1.8116810507789718e-36), Complex::new(-0.07902626473768169, -0.08184681047051735)], vec![Complex::new(0.01906923918000322, -0.10276858786959299), Complex::new(0.10089706509966115, -0.07681163409722506), Complex::new(0.019069239180003222, -0.10276858786959299), Complex::new(-0.07902626473768173, -0.08184681047051737), Complex::new(-0.07902626473768169, -0.08184681047051737), Complex::new(0.01906923918000322, -0.10276858786959299), Complex::new(-0.07902626473768169, -0.08184681047051735), Complex::new(7.314851820797302e-33, -1.088140415641433e-35)]];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(matrix.get(i, j).unwrap().re, entry.re, epsilon = 1e-4);
                assert_relative_eq!(matrix.get(i, j).unwrap().im, entry.im, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_helmholtz_adjoint_double_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new([ndofs, ndofs]);
        assemble(
            &mut matrix,
            &green::HelmholtzGreenDxKernel { k: 3.0 },
            false,
            true,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![Complex::new(1.025266688854119e-33, 7.550086433767158e-36), Complex::new(-0.079034545070751, -0.08184700030244885), Complex::new(0.019069239180003205, -0.10276858786959298), Complex::new(-0.07903454507075097, -0.08184700030244886), Complex::new(-0.07903454507075099, -0.08184700030244887), Complex::new(0.01906923918000323, -0.10276858786959299), Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.019069239180003212, -0.10276858786959298)], vec![Complex::new(-0.07903454507075097, -0.08184700030244885), Complex::new(1.025266688854119e-33, -1.0291684702482414e-35), Complex::new(-0.079034545070751, -0.08184700030244887), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(-0.07903454507075099, -0.08184700030244887), Complex::new(0.019069239180003233, -0.10276858786959299), Complex::new(0.10089706509966115, -0.07681163409722506)], vec![Complex::new(0.019069239180003205, -0.10276858786959298), Complex::new(-0.07903454507075097, -0.08184700030244886), Complex::new(1.025266688854119e-33, 7.550086433767158e-36), Complex::new(-0.079034545070751, -0.08184700030244885), Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.019069239180003212, -0.10276858786959298), Complex::new(-0.07903454507075099, -0.08184700030244887), Complex::new(0.01906923918000323, -0.10276858786959299)], vec![Complex::new(-0.079034545070751, -0.08184700030244887), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(-0.07903454507075097, -0.08184700030244885), Complex::new(1.025266688854119e-33, -1.0291684702482414e-35), Complex::new(0.019069239180003233, -0.10276858786959299), Complex::new(0.10089706509966115, -0.07681163409722506), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(-0.07903454507075099, -0.08184700030244887)], vec![Complex::new(-0.07903454507075099, -0.08184700030244887), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.01906923918000323, -0.10276858786959302), Complex::new(-5.00373588753262e-33, 1.8116810507789718e-36), Complex::new(-0.07903454507075099, -0.08184700030244885), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(-0.07903454507075099, -0.08184700030244886)], vec![Complex::new(0.019069239180003233, -0.10276858786959302), Complex::new(-0.07903454507075099, -0.08184700030244886), Complex::new(0.019069239180003212, -0.10276858786959298), Complex::new(0.10089706509966115, -0.07681163409722506), Complex::new(-0.07903454507075099, -0.08184700030244885), Complex::new(-7.314851820797302e-33, 1.088140415641433e-35), Complex::new(-0.07903454507075099, -0.08184700030244886), Complex::new(0.019069239180003215, -0.10276858786959298)], vec![Complex::new(0.10089706509966115, -0.07681163409722505), Complex::new(0.01906923918000323, -0.10276858786959302), Complex::new(-0.07903454507075099, -0.08184700030244887), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(0.01906923918000321, -0.10276858786959298), Complex::new(-0.07903454507075099, -0.08184700030244886), Complex::new(-5.00373588753262e-33, 1.8116810507789718e-36), Complex::new(-0.07903454507075099, -0.08184700030244885)], vec![Complex::new(0.019069239180003212, -0.10276858786959298), Complex::new(0.10089706509966115, -0.07681163409722506), Complex::new(0.019069239180003233, -0.10276858786959302), Complex::new(-0.07903454507075099, -0.08184700030244886), Complex::new(-0.07903454507075099, -0.08184700030244886), Complex::new(0.019069239180003215, -0.10276858786959298), Complex::new(-0.07903454507075099, -0.08184700030244885), Complex::new(-7.314851820797302e-33, 1.088140415641433e-35)]];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(matrix.get(i, j).unwrap().re, entry.re, epsilon = 1e-4);
                assert_relative_eq!(matrix.get(i, j).unwrap().im, entry.im, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_helmholtz_hypersingular_p1_p1() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            1,
            Continuity::Continuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new([ndofs, ndofs]);

        helmholtz_hypersingular_assemble(&mut matrix, &space, &space, 3.0);

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![Complex::new(-0.24054975187128322, -0.37234907871793793), Complex::new(-0.2018803657726846, -0.3708486980714607), Complex::new(-0.31151549914430937, -0.36517694339435425), Complex::new(-0.31146604913280734, -0.3652407688678574), Complex::new(-0.3114620814217625, -0.36524076431695807), Complex::new(-0.311434147468966, -0.36530056813389983)], vec![Complex::new(-0.2018803657726846, -0.3708486980714607), Complex::new(-0.24054975187128322, -0.3723490787179379), Complex::new(-0.31146604913280734, -0.3652407688678574), Complex::new(-0.31151549914430937, -0.36517694339435425), Complex::new(-0.3114620814217625, -0.36524076431695807), Complex::new(-0.311434147468966, -0.36530056813389983)], vec![Complex::new(-0.31146604913280734, -0.3652407688678574), Complex::new(-0.31151549914430937, -0.36517694339435425), Complex::new(-0.24054975187128322, -0.3723490787179379), Complex::new(-0.2018803657726846, -0.3708486980714607), Complex::new(-0.31146208142176246, -0.36524076431695807), Complex::new(-0.31143414746896597, -0.36530056813389983)], vec![Complex::new(-0.31151549914430937, -0.36517694339435425), Complex::new(-0.31146604913280734, -0.3652407688678574), Complex::new(-0.2018803657726846, -0.3708486980714607), Complex::new(-0.24054975187128322, -0.3723490787179379), Complex::new(-0.3114620814217625, -0.36524076431695807), Complex::new(-0.311434147468966, -0.36530056813389983)], vec![Complex::new(-0.31146208142176257, -0.36524076431695807), Complex::new(-0.3114620814217625, -0.3652407643169581), Complex::new(-0.3114620814217625, -0.3652407643169581), Complex::new(-0.3114620814217625, -0.3652407643169581), Complex::new(-0.24056452443903534, -0.37231826606213236), Complex::new(-0.20188036577268464, -0.37084869807146076)], vec![Complex::new(-0.3114335658086867, -0.36530052927274986), Complex::new(-0.31143356580868675, -0.36530052927274986), Complex::new(-0.3114335658086867, -0.36530052927274986), Complex::new(-0.3114335658086867, -0.36530052927274986), Complex::new(-0.2018803657726846, -0.37084869807146076), Complex::new(-0.2402983805938184, -0.37203286968364935)]];

        let perm = [0, 5, 2, 4, 3, 1];

        for (i, pi) in perm.iter().enumerate() {
            for (j, pj) in perm.iter().enumerate() {
                assert_relative_eq!(
                    matrix.get(i, j).unwrap().re,
                    from_cl[*pi][*pj].re,
                    epsilon = 1e-3
                );
                assert_relative_eq!(
                    matrix.get(i, j).unwrap().im,
                    from_cl[*pi][*pj].im,
                    epsilon = 1e-3
                );
            }
        }
    }
    */
}
