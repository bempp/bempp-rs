use crate::function_space::SerialFunctionSpace;
use crate::green::{laplace_green, laplace_green_dx, laplace_green_dy};
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::{available_rules, simplex_rule};
use bempp_quadrature::types::{CellToCellConnectivity, TestTrialNumericalQuadratureDefinition};
use bempp_tools::arrays::Array2D;
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
    if pairs.len() == 0 {
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
        println!("{} {}", test_rule.order, trial_rule.order);
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

pub fn laplace_single_layer<E: FiniteElement, F: FiniteElement>(
    trial_space: &SerialFunctionSpace<E>,
    test_space: &SerialFunctionSpace<F>,
) -> Array2D<f64> {
    assemble(laplace_green, false, false, trial_space, test_space)
}

pub fn laplace_double_layer<E: FiniteElement, F: FiniteElement>(
    trial_space: &SerialFunctionSpace<E>,
    test_space: &SerialFunctionSpace<F>,
) -> Array2D<f64> {
    assemble(laplace_green_dy, false, true, trial_space, test_space)
}

pub fn laplace_adjoint_double_layer<E: FiniteElement, F: FiniteElement>(
    trial_space: &SerialFunctionSpace<E>,
    test_space: &SerialFunctionSpace<F>,
) -> Array2D<f64> {
    assemble(laplace_green_dx, true, false, trial_space, test_space)
}

pub fn laplace_hypersingular<E: FiniteElement, F: FiniteElement>(
    trial_space: &SerialFunctionSpace<E>,
    test_space: &SerialFunctionSpace<F>,
) -> Array2D<f64> {
    hypersingular_assemble(laplace_green, trial_space, test_space)
}

fn assemble<E: FiniteElement, F: FiniteElement>(
    kernel: fn(&[f64], &[f64], &[f64], &[f64]) -> f64,
    needs_test_normal: bool,
    needs_trial_normal: bool,
    trial_space: &SerialFunctionSpace<E>,
    test_space: &SerialFunctionSpace<F>,
) -> Array2D<f64> {
    // Note: currently assumes that the two grids are the same
    // TODO: implement == and != for grids, then add:
    // if *trial_space.grid() != *test_space.grid() {
    //    unimplemented!("Assembling operators with spaces on different grids not yet supported");
    // }
    let npoints = 4;

    let grid = trial_space.grid();

    let c20 = grid.topology().connectivity(2, 0);

    let mut matrix = Array2D::<f64>::new((
        test_space.dofmap().global_size(),
        trial_space.dofmap().global_size(),
    ));

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
            let mut test_table = test_space.element().create_tabulate_array(0, rule.npoints);
            let mut trial_table = trial_space.element().create_tabulate_array(0, rule.npoints);

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
                    let mut sum = 0.0;

                    for index in 0..rule.npoints {
                        sum += kernel(
                            unsafe { test_mapped_pts.row_unchecked(index) },
                            unsafe { trial_mapped_pts.row_unchecked(index) },
                            unsafe { test_normals.row_unchecked(index) },
                            unsafe { trial_normals.row_unchecked(index) },
                        ) * rule.weights[index]
                            * unsafe { test_table.get_unchecked(0, index, test_i, 0) }
                            * test_jdet[index]
                            * unsafe { trial_table.get_unchecked(0, index, trial_i, 0) }
                            * trial_jdet[index];
                    }
                    *matrix.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                }
            }
        }
    }

    matrix
}

fn hypersingular_assemble<E: FiniteElement, F: FiniteElement>(
    kernel: fn(&[f64], &[f64], &[f64], &[f64]) -> f64,
    trial_space: &SerialFunctionSpace<E>,
    test_space: &SerialFunctionSpace<F>,
) -> Array2D<f64> {
    // Note: currently assumes that the two grids are the same
    // TODO: implement == and != for grids, then add:
    // if *trial_space.grid() != *test_space.grid() {
    //    unimplemented!("Assembling operators with spaces on different grids not yet supported");
    // }
    let npoints = 4;

    let grid = trial_space.grid();

    let c20 = grid.topology().connectivity(2, 0);

    let mut matrix = Array2D::<f64>::new((
        test_space.dofmap().global_size(),
        trial_space.dofmap().global_size(),
    ));

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
            let mut test_table = test_space.element().create_tabulate_array(1, rule.npoints);
            let mut trial_table = trial_space.element().create_tabulate_array(1, rule.npoints);

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
                    let mut sum = 0.0;

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

                        sum += kernel(
                            unsafe { test_mapped_pts.row_unchecked(index) },
                            unsafe { trial_mapped_pts.row_unchecked(index) },
                            unsafe { test_normals.row_unchecked(index) },
                            unsafe { trial_normals.row_unchecked(index) },
                        ) * rule.weights[index]
                            * dot_curls
                            * test_jdet[index]
                            * trial_jdet[index];
                    }
                    *matrix.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                }
            }
        }
    }

    matrix
}

#[cfg(test)]
mod test {
    use crate::assembly::dense::*;
    use crate::function_space::SerialFunctionSpace;
    use approx::*;
    use bempp_element::element::{LagrangeElementTriangleDegree0, LagrangeElementTriangleDegree1};
    use bempp_grid::shapes::regular_sphere;

    #[test]
    fn test_laplace_single_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = LagrangeElementTriangleDegree0 {};
        let space = SerialFunctionSpace::new(&grid, &element);

        let matrix = laplace_single_layer(&space, &space);

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

        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), from_cl[i][j], epsilon = 0.0001);
            }
        }
    }

    #[test]
    fn test_laplace_double_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = LagrangeElementTriangleDegree0 {};
        let space = SerialFunctionSpace::new(&grid, &element);

        let matrix = laplace_double_layer(&space, &space);

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

        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), from_cl[i][j], epsilon = 0.0001);
            }
        }
    }

    #[test]
    fn test_laplace_adjoint_double_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = LagrangeElementTriangleDegree0 {};
        let space = SerialFunctionSpace::new(&grid, &element);

        let matrix = laplace_adjoint_double_layer(&space, &space);

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

        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), from_cl[i][j], epsilon = 0.0001);
            }
        }
    }

    #[test]
    fn test_laplace_hypersingular_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = LagrangeElementTriangleDegree0 {};
        let space = SerialFunctionSpace::new(&grid, &element);

        let matrix = laplace_hypersingular(&space, &space);

        for i in 0..8 {
            for j in 0..8 {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), 0.0, epsilon = 0.0001);
            }
        }
    }

    #[test]
    fn test_laplace_hypersingular_p1_p1() {
        let grid = regular_sphere(0);
        let element = LagrangeElementTriangleDegree1 {};
        let space = SerialFunctionSpace::new(&grid, &element);

        let matrix = laplace_hypersingular(&space, &space);

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

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(
                    *matrix.get(i, j).unwrap(),
                    from_cl[perm[i]][perm[j]],
                    epsilon = 0.0001
                );
            }
        }
    }
}
