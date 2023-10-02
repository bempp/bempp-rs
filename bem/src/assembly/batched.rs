use crate::green::{
    HelmholtzGreenHypersingularTermKernel, HelmholtzGreenKernel, LaplaceGreenKernel, Scalar,
    SingularKernel,
};
use crate::function_space::SerialFunctionSpace;
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::{available_rules, simplex_rule};
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_tools::arrays::{Array2D, Array4D};
use bempp_traits::arrays::{AdjacencyListAccess, Array2DAccess, Array4DAccess};
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn get_quadrature_rule(
    test_celltype: ReferenceCellType,
    trial_celltype: ReferenceCellType,
    pairs: Vec<(usize, usize)>,
    npoints: usize,
) -> TestTrialNumericalQuadratureDefinition {
    if pairs.is_empty() {
        // Standard rules
        let mut npoints_test = 10 * npoints * npoints;
        for p in available_rules(test_celltype) {
            if p >= npoints * npoints && p < npoints_test {
                npoints_test = p;
            }
        }
        let mut npoints_trial = 10 * npoints * npoints;
        for p in available_rules(trial_celltype) {
            if p >= npoints * npoints && p < npoints_trial {
                npoints_trial = p;
            }
        }
        let test_rule = simplex_rule(test_celltype, npoints_test).unwrap();
        let trial_rule = simplex_rule(trial_celltype, npoints_trial).unwrap();
        if test_rule.dim != trial_rule.dim {
            unimplemented!("Quadrature with different dimension cells not supported");
        }
        if test_rule.order != trial_rule.order {
            unimplemented!("Quadrature with different trial and test orders not supported");
        }
        let dim = test_rule.dim;
        let npts = test_rule.npoints * trial_rule.npoints;
        let mut test_points = Vec::<f64>::with_capacity(dim * npts);
        let mut trial_points = Vec::<f64>::with_capacity(dim * npts);
        let mut weights = Vec::<f64>::with_capacity(npts);

        for test_i in 0..test_rule.npoints {
            for trial_i in 0..trial_rule.npoints {
                for d in 0..dim {
                    test_points.push(test_rule.points[dim * test_i + d]);
                    trial_points.push(trial_rule.points[dim * trial_i + d]);
                }
                weights.push(test_rule.weights[test_i] * trial_rule.weights[trial_i]);
            }
        }

        TestTrialNumericalQuadratureDefinition {
            dim,
            order: test_rule.order,
            npoints: npts,
            weights,
            test_points,
            trial_points,
        }
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

pub fn assemble<'a, T: Scalar, E: FiniteElement>(
    output: &mut Array2D<T>,
    kernel: &impl SingularKernel,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &SerialFunctionSpace<'a, E>,
    test_space: &SerialFunctionSpace<'a, E>,
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

    // TODO: allow user to configure this
    let npoints = 4;

    let grid = trial_space.grid();
    let geometry = grid.geometry();
    let topology = grid.topology();
    let c20 = topology.connectivity(2, 0);

    // TODO: make these configurable
    let numthreads = 4;
    let blocksize = 32;

    // TODO: make one of these in each Rayon thread?
    // TODO: these values can be known at compile time?
    let mut test_cells: Vec<&[usize]> = vec![&[]; numthreads];
    let mut test_vertices = vec![vec![0.0; 4 * 3 * blocksize]; numthreads];
    let mut test_num_vertices = vec![vec![0; blocksize]; numthreads];
    let mut trial_cells: Vec<&[usize]> = vec![&[]; numthreads];
    let mut trial_vertices = vec![vec![0.0; 4 * 3 * blocksize]; numthreads];
    let mut trial_num_vertices = vec![vec![0; blocksize]; numthreads];

    // Size of this might not be known at compile time
    let dofs_per_cell = 1;
    let mut results = vec![vec![0.0; dofs_per_cell * blocksize]; numthreads];

    let test_colouring = test_space.compute_cell_colouring();
    let trial_colouring = trial_space.compute_cell_colouring();
    let mut thread = 0;
    for test_c in &test_colouring {
        let mut test_start = 0;
        while test_start < test_c.len() {
            test_cells[thread] = &test_c[test_start..test_start + blocksize];

            let mut test_pt = 0;
            let mut test_cell = 0;
                for (i, test_c) in test_cells[thread].iter().enumerate() {
                for v in geometry.cell_vertices(*test_c).unwrap() { // TODO: index map?
                    for (d, coord) in geometry.point(*v).unwrap().iter().enumerate() {
                        test_vertices[thread][test_pt * 3 + d] = *coord;
                    }
                    test_pt += 1
                }
                test_num_vertices[thread][test_cell] = 3;
                test_cell += 1
            }
            for trial_c in &trial_colouring {
                let mut trial_start = 0;
                while trial_start < trial_c.len() {
                    trial_cells[thread] = &trial_c[trial_start..trial_start + blocksize];

                    let mut trial_pt = 0;
                    let mut trial_cell = 0;
                    for (i, trial_c) in trial_cells[thread].iter().enumerate() {
                        for v in geometry.cell_vertices(*trial_c).unwrap() { // TODO: index map?
                            for (d, coord) in geometry.point(*v).unwrap().iter().enumerate() {
                                trial_vertices[thread][trial_pt * 3 + d] = *coord;
                            }
                            trial_pt += 1
                        }
                        trial_num_vertices[thread][trial_cell] = 3;
                        trial_cell += 1
                    }

                    println!("{} -> {}:{} {}:{}", thread,
                             test_start, if test_start + blocksize <= test_c.len() { test_start + blocksize } else { test_c.len() },
                             trial_start, if trial_start + blocksize <= trial_c.len() { trial_start + blocksize } else { trial_c.len() });


                    thread += 1;
                    thread %= numthreads;
                    trial_start += blocksize;
                }
                test_start += blocksize
            }
            let n = test_c.len() / numthreads;
            // give test_c[:32] and trial_c[:32] to proc 0
            // give test_c[32:64] and trial_c[:32] to proc 1
            // etc
        }
    }

    /*
    for test_cell in 0..grid.geometry().cell_count() {
        let test_cell_tindex = grid.topology().index_map()[test_cell];
        let test_cell_gindex = grid.geometry().index_map()[test_cell];
        let test_vertices = c20.row(test_cell_tindex).unwrap();

        let mut npoints_test_cell = 10 * npoints * npoints;
        for p in available_rules(grid.topology().cell_type(test_cell_tindex).unwrap()) {
            if p >= npoints * npoints && p < npoints_test_cell {
                npoints_test_cell = p;
            }
        }
        for trial_cell in 0..grid.geometry().cell_count() {
            let trial_cell_tindex = grid.topology().index_map()[trial_cell];
            let trial_cell_gindex = grid.geometry().index_map()[trial_cell];
            let trial_vertices = c20.row(trial_cell_tindex).unwrap();

            let mut npoints_trial_cell = 10 * npoints * npoints;
            for p in available_rules(grid.topology().cell_type(trial_cell_tindex).unwrap()) {
                if p >= npoints * npoints && p < npoints_trial_cell {
                    npoints_trial_cell = p;
                }
            }

            let mut pairs = vec![];
            for (test_i, test_v) in test_vertices.iter().enumerate() {
                for (trial_i, trial_v) in trial_vertices.iter().enumerate() {
                    if test_v == trial_v {
                        pairs.push((test_i, trial_i));
                    }
                }
            }
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
            let mut trial_table =
                Array4D::<f64>::new(trial_space.element().tabulate_array_shape(0, rule.npoints));

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

            grid.geometry()
                .compute_points(&test_points, test_cell_gindex, &mut test_mapped_pts);
            grid.geometry()
                .compute_points(&trial_points, trial_cell_gindex, &mut trial_mapped_pts);
            if needs_test_normal {
                grid.geometry()
                    .compute_normals(&test_points, test_cell_gindex, &mut test_normals);
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
    */
}

pub fn curl_curl_assemble<'a, T: Scalar, E: FiniteElement>(
    output: &mut Array2D<T>,
    kernel: &impl SingularKernel,
    trial_space: &SerialFunctionSpace<'a, E>,
    test_space: &SerialFunctionSpace<'a, E>,
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

    let npoints = 4;

    let grid = trial_space.grid();
    let c20 = grid.topology().connectivity(2, 0);

    for test_cell in 0..grid.geometry().cell_count() {
        let test_cell_tindex = grid.topology().index_map()[test_cell];
        let test_cell_gindex = grid.geometry().index_map()[test_cell];
        let test_vertices = c20.row(test_cell_tindex).unwrap();

        let mut npoints_test_cell = 10 * npoints * npoints;
        for p in available_rules(grid.topology().cell_type(test_cell_tindex).unwrap()) {
            if p >= npoints * npoints && p < npoints_test_cell {
                npoints_test_cell = p;
            }
        }
        for trial_cell in 0..grid.geometry().cell_count() {
            let trial_cell_tindex = grid.topology().index_map()[trial_cell];
            let trial_cell_gindex = grid.geometry().index_map()[trial_cell];
            let trial_vertices = c20.row(trial_cell_tindex).unwrap();

            let mut npoints_trial_cell = 10 * npoints * npoints;
            for p in available_rules(grid.topology().cell_type(trial_cell_tindex).unwrap()) {
                if p >= npoints * npoints && p < npoints_trial_cell {
                    npoints_trial_cell = p;
                }
            }

            let mut pairs = vec![];
            for (test_i, test_v) in test_vertices.iter().enumerate() {
                for (trial_i, trial_v) in trial_vertices.iter().enumerate() {
                    if test_v == trial_v {
                        pairs.push((test_i, trial_i));
                    }
                }
            }
            let rule = get_quadrature_rule(
                grid.topology().cell_type(test_cell_tindex).unwrap(),
                grid.topology().cell_type(trial_cell_tindex).unwrap(),
                pairs,
                npoints,
            );
            let test_points = Array2D::from_data(rule.test_points, (rule.npoints, 2));
            let trial_points = Array2D::from_data(rule.trial_points, (rule.npoints, 2));
            let mut test_table =
                Array4D::<f64>::new(test_space.element().tabulate_array_shape(1, rule.npoints));
            let mut trial_table =
                Array4D::<f64>::new(trial_space.element().tabulate_array_shape(1, rule.npoints));

            test_space
                .element()
                .tabulate(&test_points, 1, &mut test_table);
            trial_space
                .element()
                .tabulate(&trial_points, 1, &mut trial_table);

            let mut test_jdet = vec![0.0; rule.npoints];
            let mut trial_jdet = vec![0.0; rule.npoints];
            let mut test_jinv = Array2D::<f64>::new((rule.npoints, 6));
            let mut trial_jinv = Array2D::<f64>::new((rule.npoints, 6));
            let mut test_mapped_pts = Array2D::<f64>::new((rule.npoints, 3));
            let mut trial_mapped_pts = Array2D::<f64>::new((rule.npoints, 3));
            let mut test_normals = Array2D::<f64>::new((rule.npoints, 3));
            let mut trial_normals = Array2D::<f64>::new((rule.npoints, 3));

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
            grid.geometry().compute_jacobian_inverses(
                &test_points,
                test_cell_gindex,
                &mut test_jinv,
            );
            grid.geometry().compute_jacobian_inverses(
                &trial_points,
                trial_cell_gindex,
                &mut trial_jinv,
            );
            grid.geometry()
                .compute_points(&test_points, test_cell_gindex, &mut test_mapped_pts);
            grid.geometry()
                .compute_points(&trial_points, trial_cell_gindex, &mut trial_mapped_pts);
            grid.geometry()
                .compute_normals(&test_points, test_cell_gindex, &mut test_normals);
            grid.geometry()
                .compute_normals(&trial_points, trial_cell_gindex, &mut trial_normals);

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
                        let g0 = (
                            unsafe {
                                *trial_jinv.get_unchecked(index, 0)
                                    * *trial_table.get_unchecked(1, index, trial_i, 0)
                                    + *trial_jinv.get_unchecked(index, 3)
                                        * *trial_table.get_unchecked(2, index, trial_i, 0)
                            },
                            unsafe {
                                *trial_jinv.get_unchecked(index, 1)
                                    * *trial_table.get_unchecked(1, index, trial_i, 0)
                                    + *trial_jinv.get_unchecked(index, 4)
                                        * *trial_table.get_unchecked(2, index, trial_i, 0)
                            },
                            unsafe {
                                *trial_jinv.get_unchecked(index, 2)
                                    * *trial_table.get_unchecked(1, index, trial_i, 0)
                                    + *trial_jinv.get_unchecked(index, 5)
                                        * *trial_table.get_unchecked(2, index, trial_i, 0)
                            },
                        );
                        let g1 = (
                            unsafe {
                                *test_jinv.get_unchecked(index, 0)
                                    * *test_table.get_unchecked(1, index, test_i, 0)
                                    + *test_jinv.get_unchecked(index, 3)
                                        * *test_table.get_unchecked(2, index, test_i, 0)
                            },
                            unsafe {
                                *test_jinv.get_unchecked(index, 1)
                                    * *test_table.get_unchecked(1, index, test_i, 0)
                                    + *test_jinv.get_unchecked(index, 4)
                                        * *test_table.get_unchecked(2, index, test_i, 0)
                            },
                            unsafe {
                                *test_jinv.get_unchecked(index, 2)
                                    * *test_table.get_unchecked(1, index, test_i, 0)
                                    + *test_jinv.get_unchecked(index, 5)
                                        * *test_table.get_unchecked(2, index, test_i, 0)
                            },
                        );
                        let n0 = (
                            unsafe { *trial_normals.get_unchecked(index, 0) },
                            unsafe { *trial_normals.get_unchecked(index, 1) },
                            unsafe { *trial_normals.get_unchecked(index, 2) },
                        );
                        let n1 = (
                            unsafe { *test_normals.get_unchecked(index, 0) },
                            unsafe { *test_normals.get_unchecked(index, 1) },
                            unsafe { *test_normals.get_unchecked(index, 2) },
                        );

                        let dot_curls = (g0.0 * g1.0 + g0.1 * g1.1 + g0.2 * g1.2)
                            * (n0.0 * n1.0 + n0.1 * n1.1 + n0.2 * n1.2)
                            - (g0.0 * n1.0 + g0.1 * n1.1 + g0.2 * n1.2)
                                * (n0.0 * g1.0 + n0.1 * g1.1 + n0.2 * g1.2);

                        sum += kernel.eval::<T>(
                            unsafe { test_mapped_pts.row_unchecked(index) },
                            unsafe { trial_mapped_pts.row_unchecked(index) },
                            unsafe { test_normals.row_unchecked(index) },
                            unsafe { trial_normals.row_unchecked(index) },
                        ) * T::from_f64(
                            rule.weights[index] * dot_curls * test_jdet[index] * trial_jdet[index],
                        );
                    }
                    *output.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                }
            }
        }
    }
}

pub fn laplace_hypersingular_assemble<'a, T: Scalar, E: FiniteElement>(
    output: &mut Array2D<T>,
    trial_space: &SerialFunctionSpace<'a, E>,
    test_space: &SerialFunctionSpace<'a, E>,
) {
    curl_curl_assemble(output, &LaplaceGreenKernel {}, trial_space, test_space);
}

pub fn helmholtz_hypersingular_assemble<'a, T: Scalar, E: FiniteElement>(
    output: &mut Array2D<T>,
    trial_space: &SerialFunctionSpace<'a, E>,
    test_space: &SerialFunctionSpace<'a, E>,
    k: f64,
) {
    curl_curl_assemble(output, &HelmholtzGreenKernel { k }, trial_space, test_space);
    assemble(
        output,
        &HelmholtzGreenHypersingularTermKernel { k },
        true,
        true,
        trial_space,
        test_space,
    );
}

#[cfg(test)]
mod test {
    use crate::assembly::batched::*;
    use crate::function_space::SerialFunctionSpace;
    use crate::green;
    use approx::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::{Continuity, ElementFamily};
    use num::complex::Complex;

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

        // Compare to result from bempp-cl
        #[rustfmt::skip]
        let from_cl = vec![vec![0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472], vec![0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548], vec![0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473], vec![0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074], vec![0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074], vec![0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472], vec![0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074], vec![0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487]];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), entry, epsilon = 1e-4);
            }
        }
    }
}
