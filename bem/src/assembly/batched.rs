use crate::function_space::SerialFunctionSpace;
use bempp_kernel::traits::Kernel;
use bempp_kernel::types::EvalType;
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::simplex_rule;
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_tools::arrays::{Array2D, Array4D};
use bempp_traits::arrays::{AdjacencyListAccess, Array2DAccess, Array4DAccess};
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};
use bempp_traits::types::Scalar;
use rayon::prelude::*;
use rlst_dense::{RawAccess, UnsafeRandomAccessMut};

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
    pub shape: (usize, usize),
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
    trial_points: &Array2D<f64>,
    test_points: &Array2D<f64>,
    weights: &[f64],
    trial_table: &Array4D<f64>,
    test_table: &Array4D<f64>,
) -> usize {
    let grid = test_space.grid();
    let mut k = vec![0.0];

    // Memory assignment to be moved elsewhere as passed into here mutable?
    let mut test_jdet = vec![0.0; test_points.shape().0];
    let mut trial_jdet = vec![0.0; trial_points.shape().0];
    let mut test_mapped_pts = Array2D::<f64>::new((test_points.shape().0, 3));
    let mut trial_mapped_pts = Array2D::<f64>::new((trial_points.shape().0, 3));
    let mut test_normals = Array2D::<f64>::new((test_points.shape().0, 3));
    let mut trial_normals = Array2D::<f64>::new((trial_points.shape().0, 3));

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
                    kernel.assemble_st(
                        EvalType::Value,
                        test_mapped_pts.row(index).unwrap(),
                        trial_mapped_pts.row(index).unwrap(),
                        &mut k,
                    );
                    sum += k[0]
                        * (wt
                            * unsafe { test_table.get_unchecked(0, index, test_i, 0) }
                            * test_jdet[index]
                            * unsafe { trial_table.get_unchecked(0, index, trial_i, 0) }
                            * trial_jdet[index]);
                }
                unsafe {
                    *output.data.offset(
                        (*test_dof + output.shape.0 * *trial_dof)
                            .try_into()
                            .unwrap(),
                    ) += sum;
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
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    trial_cells: &[usize],
    test_space: &SerialFunctionSpace<'a>,
    test_cells: &[usize],
    trial_points: &Array2D<f64>,
    trial_weights: &[f64],
    test_points: &Array2D<f64>,
    test_weights: &[f64],
    trial_table: &Array4D<f64>,
    test_table: &Array4D<f64>,
) -> usize {
    let test_grid = test_space.grid();
    let test_c20 = test_grid.topology().connectivity(2, 0);
    let trial_grid = trial_space.grid();
    let trial_c20 = trial_grid.topology().connectivity(2, 0);

    let mut k = vec![0.0; NPTS_TEST * NPTS_TRIAL];
    let mut test_jdet = vec![0.0; NPTS_TEST];
    let mut trial_jdet = vec![0.0; NPTS_TRIAL];
    let mut test_normals = Array2D::<f64>::new((NPTS_TEST, 3));
    let mut trial_normals = Array2D::<f64>::new((NPTS_TRIAL, 3));

    // let mut rlst_test_normals = rlst_dense::rlst_dynamic_mat![f64, (NPTS_TEST, 3)];
    // let mut rlst_trial_normals = rlst_dense::rlst_dynamic_mat![f64, (NPTS_TEST, 3)];
    let mut rlst_test_mapped_pts = rlst_dense::rlst_dynamic_mat![f64, (NPTS_TEST, 3)];
    let mut rlst_trial_mapped_pts = rlst_dense::rlst_dynamic_mat![f64, (NPTS_TRIAL, 3)];

    let test_element = test_grid.geometry().element(test_cells[0]);

    let mut test_data = Array4D::<f64>::new(test_element.tabulate_array_shape(0, NPTS_TEST));
    test_element.tabulate(test_points, 0, &mut test_data);
    let gdim = test_grid.geometry().dim();

    // TODO: move this to grid.get_compute_points_function(test_points)
    let test_compute_points = |cell: usize, pts: &mut rlst_dense::Matrix<f64, rlst_dense::base_matrix::BaseMatrix<f64, rlst_dense::VectorContainer<f64>, rlst_dense::Dynamic>, rlst_dense::Dynamic>| {
        for p in 0..NPTS_TEST {
            for i in 0..gdim {
                unsafe {
                    *pts.get_unchecked_mut(p, i) = 0.0;
                }
            }
        }
        let vertices = test_grid.geometry().cell_vertices(cell).unwrap();
        for (i, n) in vertices.iter().enumerate() {
            let pt = test_grid.geometry().point(*n).unwrap();
            for p in 0..NPTS_TEST {
                for (j, pt_j) in pt.iter().enumerate() {
                    unsafe {
                        *pts.get_unchecked_mut(p, j) +=
                            *pt_j * *test_data.get_unchecked(0, p, i, 0);
                    }
                }
            }
        }
    };

    for test_cell in test_cells {
        let test_cell_tindex = test_grid.topology().index_map()[*test_cell];
        let test_cell_gindex = test_grid.geometry().index_map()[*test_cell];
        let test_vertices = unsafe { test_c20.row_unchecked(test_cell_tindex) };

        test_grid.geometry().compute_jacobian_determinants(
            test_points,
            test_cell_gindex,
            &mut test_jdet,
        );
        test_compute_points(test_cell_gindex, &mut rlst_test_mapped_pts);

        if needs_test_normal {
            test_grid
                .geometry()
                .compute_normals(test_points, test_cell_gindex, &mut test_normals);
        }

        for trial_cell in trial_cells {
            let trial_cell_tindex = trial_grid.topology().index_map()[*trial_cell];
            let trial_cell_gindex = trial_grid.geometry().index_map()[*trial_cell];
            let trial_vertices = unsafe { trial_c20.row_unchecked(trial_cell_tindex) };

            trial_grid.geometry().compute_jacobian_determinants(
                trial_points,
                trial_cell_gindex,
                &mut trial_jdet,
            );
            trial_grid.geometry().compute_points_rlst(
                trial_points,
                trial_cell_gindex,
                &mut rlst_trial_mapped_pts,
            );
            if needs_trial_normal {
                trial_grid.geometry().compute_normals(
                    trial_points,
                    trial_cell_gindex,
                    &mut trial_normals,
                );
            }

            kernel.assemble_st(
                EvalType::Value,
                rlst_test_mapped_pts.data(),
                rlst_trial_mapped_pts.data(),
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
                    let mut sum = 0.0;

                    for (test_index, test_wt) in test_weights.iter().enumerate() {
                        for (trial_index, trial_wt) in trial_weights.iter().enumerate() {
                            sum += k[test_index * trial_weights.len() + trial_index]
                                * (test_wt
                                    * trial_wt
                                    * unsafe {
                                        test_table.get_unchecked(0, test_index, test_i, 0)
                                    }
                                    * test_jdet[test_index]
                                    * unsafe {
                                        trial_table.get_unchecked(0, trial_index, trial_i, 0)
                                    }
                                    * trial_jdet[test_index]);
                        }
                    }
                    // TODO: should we write into a result array, then copy into output after this loop?
                    let mut neighbour = false;
                    for v in test_vertices {
                        if trial_vertices.contains(v) {
                            neighbour = true;
                            break;
                        }
                    }
                    if !neighbour {
                        unsafe {
                            *output.data.offset(
                                (*test_dof + output.shape.0 * *trial_dof)
                                    .try_into()
                                    .unwrap(),
                            ) += sum;
                        }
                    }
                }
            }
        }
    }
    1
}

pub fn assemble<'a>(
    output: &mut Array2D<f64>,
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
        needs_trial_normal,
        needs_test_normal,
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
    output: &mut Array2D<f64>,
    kernel: &impl Kernel<T = f64>,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
    trial_colouring: &Vec<Vec<usize>>,
    test_colouring: &Vec<Vec<usize>>,
    blocksize: usize,
) {
    if !trial_space.is_serial() || !test_space.is_serial() {
        panic!("Dense assemble can only be used for function spaces stored in serial");
    }
    if output.shape().0 != test_space.dofmap().global_size()
        || output.shape().1 != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    // Size of this might not be known at compile time
    // let test_dofs_per_cell = 1;
    // let trial_dofs_per_cell = 1;

    // TODO: pass cell types into this function
    let qrule_test = simplex_rule(ReferenceCellType::Triangle, NPTS_TEST).unwrap();
    let qpoints_test = Array2D::from_data(qrule_test.points, (NPTS_TEST, 2));
    let qweights_test = qrule_test.weights;
    let qrule_trial = simplex_rule(ReferenceCellType::Triangle, NPTS_TRIAL).unwrap();
    let qpoints_trial = Array2D::from_data(qrule_trial.points, (NPTS_TRIAL, 2));
    let qweights_trial = qrule_trial.weights;

    let mut test_table =
        Array4D::<f64>::new(test_space.element().tabulate_array_shape(0, NPTS_TEST));
    test_space
        .element()
        .tabulate(&qpoints_test, 0, &mut test_table);

    let mut trial_table =
        Array4D::<f64>::new(trial_space.element().tabulate_array_shape(0, NPTS_TRIAL));
    trial_space
        .element()
        .tabulate(&qpoints_test, 0, &mut trial_table);

    let output_raw = RawData2D {
        data: output.data.as_mut_ptr(),
        shape: *output.shape(),
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

            let numthreads = test_cells.len();
            let r: usize = (0..numthreads)
                .into_par_iter()
                .map(&|t| {
                    assemble_batch_nonadjacent::<NPTS_TEST, NPTS_TRIAL>(
                        &output_raw,
                        kernel,
                        needs_trial_normal,
                        needs_test_normal,
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
            assert_eq!(r, numthreads);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_singular<'a>(
    output: &mut Array2D<f64>,
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
    if output.shape().0 != test_space.dofmap().global_size()
        || output.shape().1 != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    // Size of this might not be known at compile time
    // let test_dofs_per_cell = 1;
    // let trial_dofs_per_cell = 1;

    // TODO: allow user to configure this
    let npoints = 4;

    let output_raw = RawData2D {
        data: output.data.as_mut_ptr(),
        shape: *output.shape(),
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

        let points = Array2D::from_data(qrule.trial_points, (qrule.npoints, 2));
        let mut table = Array4D::<f64>::new(
            trial_space
                .element()
                .tabulate_array_shape(0, points.shape().0),
        );
        trial_space.element().tabulate(&points, 0, &mut table);
        trial_points.push(points);
        trial_tables.push(table);

        let points = Array2D::from_data(qrule.test_points, (qrule.npoints, 2));
        let mut table = Array4D::<f64>::new(
            test_space
                .element()
                .tabulate_array_shape(0, points.shape().0),
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

                let numthreads = cell_blocks.len();
                let r: usize = (0..numthreads)
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
                assert_eq!(r, numthreads);
            }
        }
    }
}
#[cfg(test)]
mod test {
    use crate::assembly::batched::*;
    use crate::assembly::dense;
    use crate::function_space::SerialFunctionSpace;
    use crate::green;
    use approx::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_kernel::laplace_3d::Laplace3dKernel;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::{Continuity, ElementFamily};

    #[cfg_attr(debug_assertions, ignore)]
    #[test]
    fn test_laplace_single_layer_dp0_dp0_larger() {
        let grid = regular_sphere(2);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &Laplace3dKernel::new(),
            false,
            false,
            &space,
            &space,
        );

        let mut dmat = Array2D::<f64>::new((ndofs, ndofs));
        dense::assemble(
            &mut dmat,
            &green::LaplaceGreenKernel {},
            false,
            false,
            &space,
            &space,
        );

        for i in 0..matrix.shape().0 {
            for j in 0..matrix.shape().1 {
                assert_relative_eq!(
                    *matrix.get(i, j).unwrap(),
                    *dmat.get(i, j).unwrap(),
                    epsilon = 1e-4
                );
            }
        }
    }

    #[test]
    fn test_laplace_single_layer_dp0_dp0_smaller() {
        let grid = regular_sphere(1);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            Continuity::Discontinuous,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &Laplace3dKernel::new(),
            false,
            false,
            &space,
            &space,
        );

        let mut dmat = Array2D::<f64>::new((ndofs, ndofs));
        dense::assemble(
            &mut dmat,
            &green::LaplaceGreenKernel {},
            false,
            false,
            &space,
            &space,
        );

        for i in 0..matrix.shape().0 {
            for j in 0..matrix.shape().1 {
                assert_relative_eq!(
                    *matrix.get(i, j).unwrap(),
                    *dmat.get(i, j).unwrap(),
                    epsilon = 1e-3
                );
            }
        }
    }
}
