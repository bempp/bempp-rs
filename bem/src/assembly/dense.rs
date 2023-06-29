use crate::green::{
    HelmholtzGreenHypersingularTermKernel, HelmholtzGreenKernel, LaplaceGreenKernel, Scalar,
    SingularKernel,
};
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

pub fn assemble<'a, T: Scalar>(
    output: &mut Array2D<T>,
    kernel: &impl SingularKernel,
    needs_trial_normal: bool,
    needs_test_normal: bool,
    trial_space: &impl FunctionSpace<'a>,
    test_space: &impl FunctionSpace<'a>,
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
}

pub fn curl_curl_assemble<'a, T: Scalar>(
    output: &mut Array2D<T>,
    kernel: &impl SingularKernel,
    trial_space: &impl FunctionSpace<'a>,
    test_space: &impl FunctionSpace<'a>,
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

pub fn laplace_hypersingular_assemble<'a, T: Scalar>(
    output: &mut Array2D<T>,
    trial_space: &impl FunctionSpace<'a>,
    test_space: &impl FunctionSpace<'a>,
) {
    curl_curl_assemble(output, &LaplaceGreenKernel {}, trial_space, test_space);
}

pub fn helmholtz_hypersingular_assemble<'a, T: Scalar>(
    output: &mut Array2D<T>,
    trial_space: &impl FunctionSpace<'a>,
    test_space: &impl FunctionSpace<'a>,
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
    use crate::assembly::dense::*;
    use crate::function_space::SerialFunctionSpace;
    use crate::green;
    use approx::*;
    use bempp_element::element::create_element;
    use bempp_grid::shapes::regular_sphere;
    use bempp_traits::cell::ReferenceCellType;
    use bempp_traits::element::ElementFamily;
    use num::complex::Complex;

    #[test]
    fn test_laplace_single_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            true,
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
        let from_cl = vec![
            vec![
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
                0.08755414595678074,
                0.05963897421514473,
                0.04670742127454548,
                0.05963897421514472,
            ],
            vec![
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
                0.05963897421514472,
                0.08755414595678074,
                0.05963897421514473,
                0.04670742127454548,
            ],
            vec![
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
                0.04670742127454548,
                0.05963897421514472,
                0.08755414595678074,
                0.05963897421514473,
            ],
            vec![
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
                0.05963897421514473,
                0.04670742127454548,
                0.05963897421514472,
                0.08755414595678074,
            ],
            vec![
                0.08755414595678074,
                0.05963897421514472,
                0.046707421274545476,
                0.05963897421514473,
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
            ],
            vec![
                0.05963897421514473,
                0.08755414595678074,
                0.05963897421514472,
                0.046707421274545476,
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
                0.05963897421514472,
            ],
            vec![
                0.046707421274545476,
                0.05963897421514473,
                0.08755414595678074,
                0.05963897421514472,
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
                0.08755414595678074,
            ],
            vec![
                0.05963897421514472,
                0.046707421274545476,
                0.05963897421514473,
                0.08755414595678074,
                0.08755414595678074,
                0.05963897421514472,
                0.08755414595678074,
                0.1854538822982487,
            ],
        ];

        for (i, row) in from_cl.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), entry, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_laplace_double_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = create_element(
            ElementFamily::Lagrange,
            ReferenceCellType::Triangle,
            0,
            true,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &green::LaplaceGreenDyKernel {},
            true,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                -1.9658941517361406e-33,
                -0.08477786720045567,
                -0.048343860959178774,
                -0.08477786720045567,
                -0.08477786720045566,
                -0.048343860959178774,
                -0.033625570841778946,
                -0.04834386095917877,
            ],
            vec![
                -0.08477786720045567,
                -1.9658941517361406e-33,
                -0.08477786720045567,
                -0.048343860959178774,
                -0.04834386095917877,
                -0.08477786720045566,
                -0.048343860959178774,
                -0.033625570841778946,
            ],
            vec![
                -0.048343860959178774,
                -0.08477786720045567,
                -1.9658941517361406e-33,
                -0.08477786720045567,
                -0.033625570841778946,
                -0.04834386095917877,
                -0.08477786720045566,
                -0.048343860959178774,
            ],
            vec![
                -0.08477786720045567,
                -0.048343860959178774,
                -0.08477786720045567,
                -1.9658941517361406e-33,
                -0.048343860959178774,
                -0.033625570841778946,
                -0.04834386095917877,
                -0.08477786720045566,
            ],
            vec![
                -0.08477786720045566,
                -0.04834386095917877,
                -0.033625570841778946,
                -0.04834386095917877,
                4.910045345075783e-33,
                -0.08477786720045566,
                -0.048343860959178774,
                -0.08477786720045566,
            ],
            vec![
                -0.04834386095917877,
                -0.08477786720045566,
                -0.04834386095917877,
                -0.033625570841778946,
                -0.08477786720045566,
                4.910045345075783e-33,
                -0.08477786720045566,
                -0.048343860959178774,
            ],
            vec![
                -0.033625570841778946,
                -0.04834386095917877,
                -0.08477786720045566,
                -0.04834386095917877,
                -0.048343860959178774,
                -0.08477786720045566,
                4.910045345075783e-33,
                -0.08477786720045566,
            ],
            vec![
                -0.04834386095917877,
                -0.033625570841778946,
                -0.04834386095917877,
                -0.08477786720045566,
                -0.08477786720045566,
                -0.048343860959178774,
                -0.08477786720045566,
                4.910045345075783e-33,
            ],
        ];

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
            true,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &green::LaplaceGreenDxKernel {},
            false,
            true,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                1.9658941517361406e-33,
                -0.08478435261011981,
                -0.048343860959178774,
                -0.0847843526101198,
                -0.08478435261011981,
                -0.04834386095917877,
                -0.033625570841778946,
                -0.048343860959178774,
            ],
            vec![
                -0.0847843526101198,
                1.9658941517361406e-33,
                -0.08478435261011981,
                -0.048343860959178774,
                -0.048343860959178774,
                -0.08478435261011981,
                -0.04834386095917877,
                -0.033625570841778946,
            ],
            vec![
                -0.048343860959178774,
                -0.0847843526101198,
                1.9658941517361406e-33,
                -0.08478435261011981,
                -0.033625570841778946,
                -0.048343860959178774,
                -0.08478435261011981,
                -0.04834386095917877,
            ],
            vec![
                -0.08478435261011981,
                -0.048343860959178774,
                -0.0847843526101198,
                1.9658941517361406e-33,
                -0.04834386095917877,
                -0.033625570841778946,
                -0.048343860959178774,
                -0.08478435261011981,
            ],
            vec![
                -0.0847843526101198,
                -0.04834386095917877,
                -0.033625570841778946,
                -0.04834386095917877,
                -4.910045345075783e-33,
                -0.0847843526101198,
                -0.048343860959178774,
                -0.08478435261011981,
            ],
            vec![
                -0.04834386095917877,
                -0.0847843526101198,
                -0.04834386095917877,
                -0.033625570841778946,
                -0.08478435261011981,
                -4.910045345075783e-33,
                -0.0847843526101198,
                -0.048343860959178774,
            ],
            vec![
                -0.033625570841778946,
                -0.04834386095917877,
                -0.0847843526101198,
                -0.04834386095917877,
                -0.048343860959178774,
                -0.08478435261011981,
                -4.910045345075783e-33,
                -0.0847843526101198,
            ],
            vec![
                -0.04834386095917877,
                -0.033625570841778946,
                -0.04834386095917877,
                -0.0847843526101198,
                -0.0847843526101198,
                -0.048343860959178774,
                -0.08478435261011981,
                -4.910045345075783e-33,
            ],
        ];

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
            true,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new((ndofs, ndofs));
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
            false,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new((ndofs, ndofs));

        laplace_hypersingular_assemble(&mut matrix, &space, &space);

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                0.33550642155494004,
                -0.10892459915262698,
                -0.05664545560057827,
                -0.05664545560057828,
                -0.0566454556005783,
                -0.05664545560057828,
            ],
            vec![
                -0.10892459915262698,
                0.33550642155494004,
                -0.05664545560057828,
                -0.05664545560057827,
                -0.05664545560057828,
                -0.05664545560057829,
            ],
            vec![
                -0.05664545560057828,
                -0.05664545560057827,
                0.33550642155494004,
                -0.10892459915262698,
                -0.056645455600578286,
                -0.05664545560057829,
            ],
            vec![
                -0.05664545560057827,
                -0.05664545560057828,
                -0.10892459915262698,
                0.33550642155494004,
                -0.05664545560057828,
                -0.056645455600578286,
            ],
            vec![
                -0.05664545560057829,
                -0.0566454556005783,
                -0.05664545560057829,
                -0.05664545560057829,
                0.33550642155494004,
                -0.10892459915262698,
            ],
            vec![
                -0.05664545560057829,
                -0.05664545560057831,
                -0.05664545560057829,
                -0.05664545560057829,
                -0.10892459915262698,
                0.33550642155494004,
            ],
        ];

        let perm = vec![0, 5, 2, 4, 3, 1];

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
            true,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<f64>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &green::HelmholtzGreenKernel { k: 3.0 },
            false,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                0.08742460357596939,
                -0.02332791148192136,
                -0.04211947809894265,
                -0.02332791148192136,
                -0.023327911481921364,
                -0.042119478098942634,
                -0.03447046598405515,
                -0.04211947809894265,
            ],
            vec![
                -0.023327911481921364,
                0.08742460357596939,
                -0.02332791148192136,
                -0.04211947809894265,
                -0.04211947809894265,
                -0.02332791148192136,
                -0.042119478098942634,
                -0.03447046598405515,
            ],
            vec![
                -0.04211947809894265,
                -0.02332791148192136,
                0.08742460357596939,
                -0.02332791148192136,
                -0.03447046598405515,
                -0.04211947809894265,
                -0.023327911481921364,
                -0.042119478098942634,
            ],
            vec![
                -0.02332791148192136,
                -0.04211947809894265,
                -0.023327911481921364,
                0.08742460357596939,
                -0.042119478098942634,
                -0.03447046598405515,
                -0.04211947809894265,
                -0.02332791148192136,
            ],
            vec![
                -0.023327911481921364,
                -0.04211947809894265,
                -0.03447046598405515,
                -0.042119478098942634,
                0.08742460357596939,
                -0.02332791148192136,
                -0.04211947809894265,
                -0.023327911481921364,
            ],
            vec![
                -0.042119478098942634,
                -0.02332791148192136,
                -0.04211947809894265,
                -0.034470465984055156,
                -0.02332791148192136,
                0.08742460357596939,
                -0.023327911481921364,
                -0.04211947809894265,
            ],
            vec![
                -0.03447046598405515,
                -0.042119478098942634,
                -0.023327911481921364,
                -0.04211947809894265,
                -0.04211947809894265,
                -0.023327911481921364,
                0.08742460357596939,
                -0.02332791148192136,
            ],
            vec![
                -0.04211947809894265,
                -0.034470465984055156,
                -0.042119478098942634,
                -0.02332791148192136,
                -0.023327911481921364,
                -0.04211947809894265,
                -0.02332791148192136,
                0.08742460357596939,
            ],
        ];

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
            true,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &green::HelmholtzGreenKernel { k: 3.0 },
            false,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                Complex::new(0.08742460357596939, 0.11004203436820102),
                Complex::new(-0.02332791148192136, 0.04919102584271124),
                Complex::new(-0.04211947809894265, 0.003720159902487029),
                Complex::new(-0.02332791148192136, 0.04919102584271125),
                Complex::new(-0.023327911481921364, 0.04919102584271124),
                Complex::new(-0.042119478098942634, 0.003720159902487025),
                Complex::new(-0.03447046598405515, -0.02816544680626108),
                Complex::new(-0.04211947809894265, 0.0037201599024870254),
            ],
            vec![
                Complex::new(-0.023327911481921364, 0.04919102584271125),
                Complex::new(0.08742460357596939, 0.11004203436820104),
                Complex::new(-0.02332791148192136, 0.04919102584271124),
                Complex::new(-0.04211947809894265, 0.0037201599024870263),
                Complex::new(-0.04211947809894265, 0.0037201599024870254),
                Complex::new(-0.02332791148192136, 0.04919102584271125),
                Complex::new(-0.042119478098942634, 0.003720159902487025),
                Complex::new(-0.03447046598405515, -0.028165446806261072),
            ],
            vec![
                Complex::new(-0.04211947809894265, 0.003720159902487029),
                Complex::new(-0.02332791148192136, 0.04919102584271125),
                Complex::new(0.08742460357596939, 0.11004203436820102),
                Complex::new(-0.02332791148192136, 0.04919102584271124),
                Complex::new(-0.03447046598405515, -0.02816544680626108),
                Complex::new(-0.04211947809894265, 0.0037201599024870254),
                Complex::new(-0.023327911481921364, 0.04919102584271124),
                Complex::new(-0.042119478098942634, 0.003720159902487025),
            ],
            vec![
                Complex::new(-0.02332791148192136, 0.04919102584271124),
                Complex::new(-0.04211947809894265, 0.0037201599024870263),
                Complex::new(-0.023327911481921364, 0.04919102584271125),
                Complex::new(0.08742460357596939, 0.11004203436820104),
                Complex::new(-0.042119478098942634, 0.003720159902487025),
                Complex::new(-0.03447046598405515, -0.028165446806261072),
                Complex::new(-0.04211947809894265, 0.0037201599024870254),
                Complex::new(-0.02332791148192136, 0.04919102584271125),
            ],
            vec![
                Complex::new(-0.023327911481921364, 0.04919102584271125),
                Complex::new(-0.04211947809894265, 0.0037201599024870263),
                Complex::new(-0.03447046598405515, -0.02816544680626108),
                Complex::new(-0.042119478098942634, 0.003720159902487025),
                Complex::new(0.08742460357596939, 0.11004203436820104),
                Complex::new(-0.02332791148192136, 0.04919102584271124),
                Complex::new(-0.04211947809894265, 0.0037201599024870267),
                Complex::new(-0.023327911481921364, 0.04919102584271125),
            ],
            vec![
                Complex::new(-0.042119478098942634, 0.003720159902487025),
                Complex::new(-0.02332791148192136, 0.04919102584271125),
                Complex::new(-0.04211947809894265, 0.0037201599024870263),
                Complex::new(-0.034470465984055156, -0.028165446806261075),
                Complex::new(-0.02332791148192136, 0.04919102584271124),
                Complex::new(0.08742460357596939, 0.11004203436820104),
                Complex::new(-0.023327911481921364, 0.04919102584271125),
                Complex::new(-0.04211947809894265, 0.0037201599024870237),
            ],
            vec![
                Complex::new(-0.03447046598405515, -0.02816544680626108),
                Complex::new(-0.042119478098942634, 0.003720159902487025),
                Complex::new(-0.023327911481921364, 0.04919102584271125),
                Complex::new(-0.04211947809894265, 0.0037201599024870263),
                Complex::new(-0.04211947809894265, 0.0037201599024870267),
                Complex::new(-0.023327911481921364, 0.04919102584271125),
                Complex::new(0.08742460357596939, 0.11004203436820104),
                Complex::new(-0.02332791148192136, 0.04919102584271124),
            ],
            vec![
                Complex::new(-0.04211947809894265, 0.0037201599024870263),
                Complex::new(-0.034470465984055156, -0.028165446806261075),
                Complex::new(-0.042119478098942634, 0.003720159902487025),
                Complex::new(-0.02332791148192136, 0.04919102584271125),
                Complex::new(-0.023327911481921364, 0.04919102584271125),
                Complex::new(-0.04211947809894265, 0.0037201599024870237),
                Complex::new(-0.02332791148192136, 0.04919102584271124),
                Complex::new(0.08742460357596939, 0.11004203436820104),
            ],
        ];
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
            true,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &green::HelmholtzGreenDyKernel { k: 3.0 },
            true,
            false,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                Complex::new(-1.025266688854119e-33, -7.550086433767158e-36),
                Complex::new(-0.07902626473768169, -0.08184681047051735),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(-0.07902626473768172, -0.08184681047051737),
                Complex::new(-0.07902626473768169, -0.08184681047051737),
                Complex::new(0.01906923918000323, -0.10276858786959302),
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.019069239180003215, -0.10276858786959299),
            ],
            vec![
                Complex::new(-0.07902626473768172, -0.08184681047051737),
                Complex::new(-1.025266688854119e-33, 1.0291684702482414e-35),
                Complex::new(-0.0790262647376817, -0.08184681047051737),
                Complex::new(0.019069239180003212, -0.10276858786959299),
                Complex::new(0.019069239180003212, -0.10276858786959298),
                Complex::new(-0.07902626473768168, -0.08184681047051737),
                Complex::new(0.01906923918000323, -0.10276858786959299),
                Complex::new(0.10089706509966115, -0.07681163409722506),
            ],
            vec![
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(-0.07902626473768172, -0.08184681047051737),
                Complex::new(-1.025266688854119e-33, -7.550086433767158e-36),
                Complex::new(-0.07902626473768169, -0.08184681047051735),
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.019069239180003215, -0.10276858786959299),
                Complex::new(-0.07902626473768169, -0.08184681047051737),
                Complex::new(0.01906923918000323, -0.10276858786959302),
            ],
            vec![
                Complex::new(-0.0790262647376817, -0.08184681047051737),
                Complex::new(0.019069239180003212, -0.10276858786959299),
                Complex::new(-0.07902626473768172, -0.08184681047051737),
                Complex::new(-1.025266688854119e-33, 1.0291684702482414e-35),
                Complex::new(0.01906923918000323, -0.10276858786959299),
                Complex::new(0.10089706509966115, -0.07681163409722506),
                Complex::new(0.019069239180003212, -0.10276858786959298),
                Complex::new(-0.07902626473768168, -0.08184681047051737),
            ],
            vec![
                Complex::new(-0.07902626473768172, -0.08184681047051737),
                Complex::new(0.019069239180003215, -0.10276858786959298),
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.01906923918000323, -0.10276858786959299),
                Complex::new(5.00373588753262e-33, -1.8116810507789718e-36),
                Complex::new(-0.07902626473768169, -0.08184681047051735),
                Complex::new(0.019069239180003212, -0.10276858786959299),
                Complex::new(-0.07902626473768169, -0.08184681047051737),
            ],
            vec![
                Complex::new(0.019069239180003222, -0.10276858786959299),
                Complex::new(-0.07902626473768173, -0.08184681047051737),
                Complex::new(0.01906923918000322, -0.10276858786959299),
                Complex::new(0.10089706509966115, -0.07681163409722506),
                Complex::new(-0.07902626473768169, -0.08184681047051735),
                Complex::new(7.314851820797302e-33, -1.088140415641433e-35),
                Complex::new(-0.07902626473768169, -0.08184681047051737),
                Complex::new(0.01906923918000322, -0.10276858786959299),
            ],
            vec![
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.01906923918000323, -0.10276858786959299),
                Complex::new(-0.07902626473768172, -0.08184681047051737),
                Complex::new(0.019069239180003215, -0.10276858786959298),
                Complex::new(0.019069239180003212, -0.10276858786959299),
                Complex::new(-0.07902626473768169, -0.08184681047051737),
                Complex::new(5.00373588753262e-33, -1.8116810507789718e-36),
                Complex::new(-0.07902626473768169, -0.08184681047051735),
            ],
            vec![
                Complex::new(0.01906923918000322, -0.10276858786959299),
                Complex::new(0.10089706509966115, -0.07681163409722506),
                Complex::new(0.019069239180003222, -0.10276858786959299),
                Complex::new(-0.07902626473768173, -0.08184681047051737),
                Complex::new(-0.07902626473768169, -0.08184681047051737),
                Complex::new(0.01906923918000322, -0.10276858786959299),
                Complex::new(-0.07902626473768169, -0.08184681047051735),
                Complex::new(7.314851820797302e-33, -1.088140415641433e-35),
            ],
        ];

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
            true,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new((ndofs, ndofs));
        assemble(
            &mut matrix,
            &green::HelmholtzGreenDxKernel { k: 3.0 },
            false,
            true,
            &space,
            &space,
        );

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                Complex::new(1.025266688854119e-33, 7.550086433767158e-36),
                Complex::new(-0.079034545070751, -0.08184700030244885),
                Complex::new(0.019069239180003205, -0.10276858786959298),
                Complex::new(-0.07903454507075097, -0.08184700030244886),
                Complex::new(-0.07903454507075099, -0.08184700030244887),
                Complex::new(0.01906923918000323, -0.10276858786959299),
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.019069239180003212, -0.10276858786959298),
            ],
            vec![
                Complex::new(-0.07903454507075097, -0.08184700030244885),
                Complex::new(1.025266688854119e-33, -1.0291684702482414e-35),
                Complex::new(-0.079034545070751, -0.08184700030244887),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(-0.07903454507075099, -0.08184700030244887),
                Complex::new(0.019069239180003233, -0.10276858786959299),
                Complex::new(0.10089706509966115, -0.07681163409722506),
            ],
            vec![
                Complex::new(0.019069239180003205, -0.10276858786959298),
                Complex::new(-0.07903454507075097, -0.08184700030244886),
                Complex::new(1.025266688854119e-33, 7.550086433767158e-36),
                Complex::new(-0.079034545070751, -0.08184700030244885),
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.019069239180003212, -0.10276858786959298),
                Complex::new(-0.07903454507075099, -0.08184700030244887),
                Complex::new(0.01906923918000323, -0.10276858786959299),
            ],
            vec![
                Complex::new(-0.079034545070751, -0.08184700030244887),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(-0.07903454507075097, -0.08184700030244885),
                Complex::new(1.025266688854119e-33, -1.0291684702482414e-35),
                Complex::new(0.019069239180003233, -0.10276858786959299),
                Complex::new(0.10089706509966115, -0.07681163409722506),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(-0.07903454507075099, -0.08184700030244887),
            ],
            vec![
                Complex::new(-0.07903454507075099, -0.08184700030244887),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.01906923918000323, -0.10276858786959302),
                Complex::new(-5.00373588753262e-33, 1.8116810507789718e-36),
                Complex::new(-0.07903454507075099, -0.08184700030244885),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(-0.07903454507075099, -0.08184700030244886),
            ],
            vec![
                Complex::new(0.019069239180003233, -0.10276858786959302),
                Complex::new(-0.07903454507075099, -0.08184700030244886),
                Complex::new(0.019069239180003212, -0.10276858786959298),
                Complex::new(0.10089706509966115, -0.07681163409722506),
                Complex::new(-0.07903454507075099, -0.08184700030244885),
                Complex::new(-7.314851820797302e-33, 1.088140415641433e-35),
                Complex::new(-0.07903454507075099, -0.08184700030244886),
                Complex::new(0.019069239180003215, -0.10276858786959298),
            ],
            vec![
                Complex::new(0.10089706509966115, -0.07681163409722505),
                Complex::new(0.01906923918000323, -0.10276858786959302),
                Complex::new(-0.07903454507075099, -0.08184700030244887),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(0.01906923918000321, -0.10276858786959298),
                Complex::new(-0.07903454507075099, -0.08184700030244886),
                Complex::new(-5.00373588753262e-33, 1.8116810507789718e-36),
                Complex::new(-0.07903454507075099, -0.08184700030244885),
            ],
            vec![
                Complex::new(0.019069239180003212, -0.10276858786959298),
                Complex::new(0.10089706509966115, -0.07681163409722506),
                Complex::new(0.019069239180003233, -0.10276858786959302),
                Complex::new(-0.07903454507075099, -0.08184700030244886),
                Complex::new(-0.07903454507075099, -0.08184700030244886),
                Complex::new(0.019069239180003215, -0.10276858786959298),
                Complex::new(-0.07903454507075099, -0.08184700030244885),
                Complex::new(-7.314851820797302e-33, 1.088140415641433e-35),
            ],
        ];

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
            false,
        );
        let space = SerialFunctionSpace::new(&grid, &element);

        let ndofs = space.dofmap().global_size();

        let mut matrix = Array2D::<Complex<f64>>::new((ndofs, ndofs));

        helmholtz_hypersingular_assemble(&mut matrix, &space, &space, 3.0);

        // Compare to result from bempp-cl
        let from_cl = vec![
            vec![
                Complex::new(-0.24054975187128322, -0.37234907871793793),
                Complex::new(-0.2018803657726846, -0.3708486980714607),
                Complex::new(-0.31151549914430937, -0.36517694339435425),
                Complex::new(-0.31146604913280734, -0.3652407688678574),
                Complex::new(-0.3114620814217625, -0.36524076431695807),
                Complex::new(-0.311434147468966, -0.36530056813389983),
            ],
            vec![
                Complex::new(-0.2018803657726846, -0.3708486980714607),
                Complex::new(-0.24054975187128322, -0.3723490787179379),
                Complex::new(-0.31146604913280734, -0.3652407688678574),
                Complex::new(-0.31151549914430937, -0.36517694339435425),
                Complex::new(-0.3114620814217625, -0.36524076431695807),
                Complex::new(-0.311434147468966, -0.36530056813389983),
            ],
            vec![
                Complex::new(-0.31146604913280734, -0.3652407688678574),
                Complex::new(-0.31151549914430937, -0.36517694339435425),
                Complex::new(-0.24054975187128322, -0.3723490787179379),
                Complex::new(-0.2018803657726846, -0.3708486980714607),
                Complex::new(-0.31146208142176246, -0.36524076431695807),
                Complex::new(-0.31143414746896597, -0.36530056813389983),
            ],
            vec![
                Complex::new(-0.31151549914430937, -0.36517694339435425),
                Complex::new(-0.31146604913280734, -0.3652407688678574),
                Complex::new(-0.2018803657726846, -0.3708486980714607),
                Complex::new(-0.24054975187128322, -0.3723490787179379),
                Complex::new(-0.3114620814217625, -0.36524076431695807),
                Complex::new(-0.311434147468966, -0.36530056813389983),
            ],
            vec![
                Complex::new(-0.31146208142176257, -0.36524076431695807),
                Complex::new(-0.3114620814217625, -0.3652407643169581),
                Complex::new(-0.3114620814217625, -0.3652407643169581),
                Complex::new(-0.3114620814217625, -0.3652407643169581),
                Complex::new(-0.24056452443903534, -0.37231826606213236),
                Complex::new(-0.20188036577268464, -0.37084869807146076),
            ],
            vec![
                Complex::new(-0.3114335658086867, -0.36530052927274986),
                Complex::new(-0.31143356580868675, -0.36530052927274986),
                Complex::new(-0.3114335658086867, -0.36530052927274986),
                Complex::new(-0.3114335658086867, -0.36530052927274986),
                Complex::new(-0.2018803657726846, -0.37084869807146076),
                Complex::new(-0.2402983805938184, -0.37203286968364935),
            ],
        ];

        let perm = vec![0, 5, 2, 4, 3, 1];

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
}
