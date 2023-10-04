use crate::function_space::SerialFunctionSpace;
use crate::green::{
    //HelmholtzGreenHypersingularTermKernel, HelmholtzGreenKernel, LaplaceGreenKernel,
    Scalar,
    SingularKernel,
};
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
use rayon::prelude::*;

fn get_quadrature_rule(
    test_celltype: ReferenceCellType,
    trial_celltype: ReferenceCellType,
    pairs: Vec<(usize, usize)>,
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
                    local_indices: pairs,
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
                    local_indices: pairs,
                },
                npoints,
            )
            .unwrap()
        }
    }
}

struct RawData2D<T: Scalar> {
    pub data: *mut T,
    pub shape: (usize, usize),    
}

unsafe impl<T: Scalar> Sync for RawData2D<T> {}

#[allow(clippy::too_many_arguments)]
fn assemble_batch_nonadjacent<'a, T: Scalar + Clone + Copy>(
    output: &RawData2D<T>,
    kernel: &impl SingularKernel,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    trial_cells: &[usize],
    test_space: &SerialFunctionSpace<'a>,
    test_cells: &[usize],
    trial_points: &Array2D<f64>,
    trial_weights: &Vec<f64>,
    test_points: &Array2D<f64>,
    test_weights: &Vec<f64>,
    trial_table: &Array4D<f64>,
    test_table: &Array4D<f64>,
) -> usize {
    let test_grid = test_space.grid();
    let test_c20 = test_grid.topology().connectivity(2, 0);
    let trial_grid = trial_space.grid();
    let trial_c20 = trial_grid.topology().connectivity(2, 0);

    // Memory assignment to be moved elsewhere as passed into here mutable?
    let mut test_jdet = vec![0.0; test_points.shape().0];
    let mut trial_jdet = vec![0.0; trial_points.shape().0];
    let mut test_mapped_pts = Array2D::<f64>::new((test_points.shape().0, 3));
    let mut trial_mapped_pts = Array2D::<f64>::new((trial_points.shape().0, 3));
    let mut test_normals = Array2D::<f64>::new((test_points.shape().0, 3));
    let mut trial_normals = Array2D::<f64>::new((trial_points.shape().0, 3));

    for test_cell in test_cells {
        let test_cell_tindex = test_grid.topology().index_map()[*test_cell];
        let test_cell_gindex = test_grid.geometry().index_map()[*test_cell];
        let test_vertices = test_c20.row(test_cell_tindex).unwrap();

        test_grid.geometry().compute_jacobian_determinants(
            test_points,
            test_cell_gindex,
            &mut test_jdet,
        );
        test_grid
            .geometry()
            .compute_points(test_points, test_cell_gindex, &mut test_mapped_pts);
        if needs_test_normal {
            test_grid
                .geometry()
                .compute_normals(test_points, test_cell_gindex, &mut test_normals);
        }

        for trial_cell in trial_cells {
            let trial_cell_tindex = trial_grid.topology().index_map()[*trial_cell];
            let trial_cell_gindex = trial_grid.geometry().index_map()[*trial_cell];
            let trial_vertices = trial_c20.row(trial_cell_tindex).unwrap();

            trial_grid.geometry().compute_jacobian_determinants(
                trial_points,
                trial_cell_gindex,
                &mut trial_jdet,
            );
            trial_grid.geometry().compute_points(
                trial_points,
                trial_cell_gindex,
                &mut trial_mapped_pts,
            );
            if needs_trial_normal {
                trial_grid.geometry().compute_normals(
                    trial_points,
                    trial_cell_gindex,
                    &mut trial_normals,
                );
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
                    let mut sum = T::zero();

                    for test_index in 0..test_points.shape().0 {
                        for trial_index in 0..trial_points.shape().0 {
                            sum += kernel.eval::<T>(
                                unsafe { test_mapped_pts.row_unchecked(test_index) },
                                unsafe { trial_mapped_pts.row_unchecked(trial_index) },
                                unsafe { test_normals.row_unchecked(test_index) },
                                unsafe { trial_normals.row_unchecked(trial_index) },
                            ) * T::from_f64(
                                test_weights[test_index]
                                    * trial_weights[trial_index]
                                    * unsafe { test_table.get_unchecked(0, test_index, test_i, 0) }
                                    * test_jdet[test_index]
                                    * unsafe {
                                        trial_table.get_unchecked(0, trial_index, trial_i, 0)
                                    }
                                    * trial_jdet[test_index],
                            );
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
                            *output.data.offset((*test_dof + output.shape.0 * *trial_dof).try_into().unwrap()) += sum;
                        }
                    }
                }
            }
        }
    }
    1
}

pub fn assemble<'a, T: Scalar + Clone + Copy + Sync>(
    output: &mut Array2D<T>,
    kernel: &impl SingularKernel,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a>,
    test_space: &SerialFunctionSpace<'a>,
) {
    // Note: currently assumes that the two grids are the same
    // TODO: implement == and != for grids, then add:
    // if *trial_space.grid() != *test_space.grid() {
    //    unimplemented!("Assembling operators with spaces on different grids not yet supported");
    // }
    if !trial_space.is_serial() || !test_space.is_serial() {
        panic!("Dense assemble can only be used for function spaces stored in serial");
    }
    if output.shape().0 != test_space.dofmap().global_size()
        || output.shape().1 != trial_space.dofmap().global_size()
    {
        panic!("Matrix has wrong shape");
    }

    // TODO: make these configurable
    let blocksize = 128;

    // Size of this might not be known at compile time
    // let test_dofs_per_cell = 1;
    // let trial_dofs_per_cell = 1;

    // TODO: allow user to configure this
    let npoints = 16;

    // TODO: pass cell types into this function
    let qrule = simplex_rule(ReferenceCellType::Triangle, npoints).unwrap();
    let qpoints = Array2D::from_data(qrule.points, (qrule.npoints, 2));
    let qweights = qrule.weights;

    let mut test_table = Array4D::<f64>::new(
        test_space
            .element()
            .tabulate_array_shape(0, qpoints.shape().0),
    );
    test_space
        .element()
        .tabulate(&qpoints, 0, &mut test_table);

    let mut trial_table = Array4D::<f64>::new(
        trial_space
            .element()
            .tabulate_array_shape(0, qpoints.shape().0),
    );
    trial_space
        .element()
        .tabulate(&qpoints, 0, &mut trial_table);

    let test_colouring = test_space.compute_cell_colouring();
    let trial_colouring = trial_space.compute_cell_colouring();
    for test_c in &test_colouring {
        for trial_c in &trial_colouring {
            let mut test_cells: Vec<&[usize]> = vec![&[]];
            let mut trial_cells: Vec<&[usize]> = vec![&[]];

            let mut test_start = 0;
            while test_start < test_c.len() {
                let test_end = if test_start + blocksize < test_c.len() { test_start + blocksize } else { test_c.len() };

                let mut trial_start = 0;
                while trial_start < trial_c.len() {
                    let trial_end = if trial_start + blocksize < trial_c.len() { trial_start + blocksize } else { trial_c.len() };
                    test_cells.push(&test_c[test_start..test_end]);
                    trial_cells.push(&trial_c[trial_start..trial_end]);
                    trial_start = trial_end;
                }
                test_start = test_end
            }

            let numthreads = test_cells.len();
            let output_data = RawData2D { data: output.data.as_mut_ptr(), shape: *output.shape() };
            let r: usize = (0..numthreads).into_par_iter().map(&|t| {
                assemble_batch_nonadjacent(
                    &output_data,
                    kernel,
                    needs_trial_normal,
                    needs_test_normal,
                    trial_space,
                    trial_cells[t],
                    test_space,
                    test_cells[t],
                    &qpoints, &qweights,
                    &qpoints, &qweights,
                    &trial_table,
                    &test_table,
                )
            }).sum();
            assert_eq!(r, numthreads);
        }
    }

    if test_space.grid() != trial_space.grid() {
        // If the test and trial grids are different, there are no neighbouring triangles
        return;
    }

    // TODO: allow user to configure this
    let npoints = 4;

    let grid = test_space.grid();
    let c20 = grid.topology().connectivity(2, 0);

    // Loop through colours
        // loop through cells
            // Find pairs
            // if pairs.len() > 0
                // Add to block for that pair
        // assemble singular blocks for each pair

    for (vertex, cells) in grid.topology().connectivity(0, 2).iter_rows().enumerate() {
        for test_cell in cells {
            let test_cell_tindex = grid.topology().index_map()[*test_cell];
            let test_cell_gindex = grid.geometry().index_map()[*test_cell];
            let test_vertices = c20.row(test_cell_tindex).unwrap();
            for trial_cell in cells {
                let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
                let trial_cell_gindex = grid.geometry().index_map()[*trial_cell];
                let trial_vertices = c20.row(trial_cell_tindex).unwrap();

                let mut ismin = true;
                let mut pairs = vec![];
                for (test_i, test_v) in test_vertices.iter().enumerate() {
                    for (trial_i, trial_v) in trial_vertices.iter().enumerate() {
                        if test_v == trial_v {
                            if *test_v < vertex {
                                ismin = false;
                                break;
                            }
                            pairs.push((test_i, trial_i));
                        }
                    }
                    if !ismin { break; }
                }
                if ismin {
                    let rule = get_quadrature_rule(
                        grid.topology().cell_type(test_cell_tindex).unwrap(),
                        grid.topology().cell_type(trial_cell_tindex).unwrap(),
                        pairs,
                        npoints,
                    );

                    let test_points = Array2D::from_data(rule.test_points, (rule.npoints, 2));
                    let trial_points = Array2D::from_data(rule.trial_points, (rule.npoints, 2));
                    let mut test_table =
                        Array4D::<f64>::new(test_space.element().tabulate_array_shape(0, rule.npoints));
                    let mut trial_table = Array4D::<f64>::new(
                        trial_space.element().tabulate_array_shape(0, rule.npoints),
                    );

                    test_space
                        .element()
                        .tabulate(&test_points, 0, &mut test_table);
                    trial_space
                        .element()
                        .tabulate(&trial_points, 0, &mut trial_table);

                    let mut test_jdet = vec![0.0; rule.npoints];
                    let mut trial_jdet = vec![0.0; rule.npoints];

                    grid.geometry().compute_jacobian_determinants(
                        &test_points,
                        test_cell_gindex,
                        &mut test_jdet,
                    );
                    grid.geometry().compute_jacobian_determinants(
                        &trial_points,
                        trial_cell_gindex,
                        &mut trial_jdet,
                    );

                    let mut test_mapped_pts = Array2D::<f64>::new((rule.npoints, 3));
                    let mut trial_mapped_pts = Array2D::<f64>::new((rule.npoints, 3));
                    let mut test_normals = Array2D::<f64>::new((rule.npoints, 3));
                    let mut trial_normals = Array2D::<f64>::new((rule.npoints, 3));

                    grid.geometry().compute_points(
                        &test_points,
                        test_cell_gindex,
                        &mut test_mapped_pts,
                    );
                    grid.geometry().compute_points(
                        &trial_points,
                        trial_cell_gindex,
                        &mut trial_mapped_pts,
                    );
                    if needs_test_normal {
                        grid.geometry().compute_normals(
                            &test_points,
                            test_cell_gindex,
                            &mut test_normals,
                        );
                    }
                    if needs_trial_normal {
                        grid.geometry().compute_normals(
                            &trial_points,
                            trial_cell_gindex,
                            &mut trial_normals,
                        );
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
                            let mut sum = T::zero();

                            for index in 0..rule.npoints {
                                sum += kernel.eval::<T>(
                                    unsafe { test_mapped_pts.row_unchecked(index) },
                                    unsafe { trial_mapped_pts.row_unchecked(index) },
                                    unsafe { test_normals.row_unchecked(index) },
                                    unsafe { trial_normals.row_unchecked(index) },
                                ) * T::from_f64(
                                    rule.weights[index]
                                        * unsafe { test_table.get_unchecked(0, index, test_i, 0) }
                                        * test_jdet[index]
                                        * unsafe { trial_table.get_unchecked(0, index, trial_i, 0) }
                                        * trial_jdet[index],
                                );
                            }
                            *output.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                        }
                    }
                }
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
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::{Continuity, ElementFamily};
    // use num::complex::Complex;

    #[test]
    fn test_laplace_single_layer_dp0_dp0() {
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
            &green::LaplaceGreenKernel {},
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

        /*for i in 0..matrix.shape().0 {
            for j in 0..matrix.shape().1 {
                println!("{} {}",
                    *matrix.get(i, j).unwrap(),
                    *dmat.get(i, j).unwrap(),
                );
            }
        }*/
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
}
