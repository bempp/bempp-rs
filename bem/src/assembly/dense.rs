use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::{available_rules, simplex_rule};
use bempp_quadrature::types::CellToCellConnectivity;
use bempp_tools::arrays::Array2D;
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn laplace_green(x: &[f64], y: &[f64], _nx: &[f64], _ny: &[f64]) -> f64 {
    let inv_dist = 1.0
        / f64::sqrt(
            (x[0] - y[0]) * (x[0] - y[0])
                + (x[1] - y[1]) * (x[1] - y[1])
                + (x[2] - y[2]) * (x[2] - y[2]),
        );

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist
}

fn laplace_green_dx(x: &[f64], y: &[f64], nx: &[f64], _ny: &[f64]) -> f64 {
    let inv_dist = 1.0
        / f64::sqrt(
            (x[0] - y[0]) * (x[0] - y[0])
                + (x[1] - y[1]) * (x[1] - y[1])
                + (x[2] - y[2]) * (x[2] - y[2]),
        );
    let sum = (y[0] - x[0]) * nx[0] + (y[1] - x[1]) * nx[1] + (y[2] - x[2]) * nx[2];

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist * inv_dist * inv_dist * sum
}

fn laplace_green_dy(x: &[f64], y: &[f64], _nx: &[f64], ny: &[f64]) -> f64 {
    let inv_dist = 1.0
        / f64::sqrt(
            (x[0] - y[0]) * (x[0] - y[0])
                + (x[1] - y[1]) * (x[1] - y[1])
                + (x[2] - y[2]) * (x[2] - y[2]),
        );
    let sum = (x[0] - y[0]) * ny[0] + (x[1] - y[1]) * ny[1] + (x[2] - y[2]) * ny[2];

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist * inv_dist * inv_dist * sum
}

pub fn laplace_single_layer(
    trial_space: &impl FunctionSpace,
    test_space: &impl FunctionSpace,
) -> Array2D<f64> {
    assemble(laplace_green, false, false, trial_space, test_space)
}

pub fn laplace_double_layer(
    trial_space: &impl FunctionSpace,
    test_space: &impl FunctionSpace,
) -> Array2D<f64> {
    assemble(laplace_green_dy, false, true, trial_space, test_space)
}

pub fn laplace_adjoint_double_layer(
    trial_space: &impl FunctionSpace,
    test_space: &impl FunctionSpace,
) -> Array2D<f64> {
    assemble(laplace_green_dx, true, false, trial_space, test_space)
}

pub fn laplace_hypersingular(
    trial_space: &impl FunctionSpace,
    test_space: &impl FunctionSpace,
) -> Array2D<f64> {
    hypersingular_assemble(laplace_green, trial_space, test_space)
}

fn assemble(
    kernel: fn(&[f64], &[f64], &[f64], &[f64]) -> f64,
    needs_test_normal: bool,
    needs_trial_normal: bool,
    trial_space: &impl FunctionSpace,
    test_space: &impl FunctionSpace,
) -> Array2D<f64> {
    // Note: currently assumes that the two grids are the same
    // TODO: implement == and != for grids, then add:
    // if *trial_space.grid() != *test_space.grid() {
    //    unimplemented!("Assembling operators with spaces on different grids not yet supported");
    // }
    let npoints = 4;

    let grid = trial_space.grid();

    let c20 = grid.topology().connectivity(2, 0);

    // Assign working memory
    let mut test_pt = Array2D::<f64>::new((1, 2));
    let mut trial_pt = Array2D::<f64>::new((1, 2));
    let mut test_mapped_pt = Array2D::<f64>::new((1, 3));
    let mut trial_mapped_pt = Array2D::<f64>::new((1, 3));

    let mut test_normal = Array2D::<f64>::new((1, 3));
    let mut trial_normal = Array2D::<f64>::new((1, 3));

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
            if pairs.len() == 0 {
                // Standard quadrature
                let test_rule = simplex_rule(
                    grid.topology().cell_type(test_cell_tindex).unwrap(),
                    npoints_test_cell,
                )
                .unwrap();
                let trial_rule = simplex_rule(
                    grid.topology().cell_type(trial_cell_tindex).unwrap(),
                    npoints_trial_cell,
                )
                .unwrap();

                let test_points = Array2D::from_data(test_rule.points, (test_rule.npoints, 2));
                let trial_points = Array2D::from_data(trial_rule.points, (trial_rule.npoints, 2));
                let mut test_table = test_space
                    .element()
                    .create_tabulate_array(0, test_rule.npoints);
                let mut trial_table = trial_space
                    .element()
                    .create_tabulate_array(0, trial_rule.npoints);

                test_space
                    .element()
                    .tabulate(&test_points, 0, &mut test_table);
                trial_space
                    .element()
                    .tabulate(&trial_points, 0, &mut trial_table);

                let mut test_jdet = vec![0.0; test_rule.npoints];
                let mut trial_jdet = vec![0.0; trial_rule.npoints];

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

                        for test_index in 0..test_rule.npoints {
                            unsafe {
                                *test_pt.get_unchecked_mut(0, 0) =
                                    *test_points.get_unchecked(test_index, 0);
                                *test_pt.get_unchecked_mut(0, 1) =
                                    *test_points.get_unchecked(test_index, 1);
                            }
                            grid.geometry().compute_points(
                                &test_pt,
                                test_cell_gindex,
                                &mut test_mapped_pt,
                            );
                            if needs_test_normal {
                                grid.geometry().compute_normals(
                                    &test_pt,
                                    test_cell_gindex,
                                    &mut test_normal,
                                );
                            }
                            let test_weight = test_rule.weights[test_index];

                            for trial_index in 0..trial_rule.npoints {
                                unsafe {
                                    *trial_pt.get_unchecked_mut(0, 0) =
                                        *trial_points.get_unchecked(trial_index, 0);
                                    *trial_pt.get_unchecked_mut(0, 1) =
                                        *trial_points.get_unchecked(trial_index, 1);
                                }
                                grid.geometry().compute_points(
                                    &trial_pt,
                                    trial_cell_gindex,
                                    &mut trial_mapped_pt,
                                );
                                if needs_trial_normal {
                                    grid.geometry().compute_normals(
                                        &trial_pt,
                                        trial_cell_gindex,
                                        &mut trial_normal,
                                    );
                                }
                                let trial_weight = trial_rule.weights[trial_index];

                                sum += kernel(
                                    unsafe { test_mapped_pt.row_unchecked(0) },
                                    unsafe { trial_mapped_pt.row_unchecked(0) },
                                    unsafe { test_normal.row_unchecked(0) },
                                    unsafe { trial_normal.row_unchecked(0) },
                                ) * test_weight
                                    * trial_weight
                                    * unsafe { test_table.get_unchecked(0, test_index, test_i, 0) }
                                    * test_jdet[test_index]
                                    * unsafe {
                                        trial_table.get_unchecked(0, trial_index, trial_i, 0)
                                    }
                                    * trial_jdet[trial_index];
                            }
                        }
                        *matrix.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                    }
                }
            } else {
                // Singular quadrature
                let singular_rule = if grid.topology().cell_type(test_cell_tindex).unwrap()
                    == ReferenceCellType::Triangle
                {
                    if grid.topology().cell_type(trial_cell_tindex).unwrap()
                        != ReferenceCellType::Triangle
                    {
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
                    if grid.topology().cell_type(test_cell_tindex).unwrap()
                        != ReferenceCellType::Quadrilateral
                    {
                        unimplemented!("Only triangles and quadrilaterals are currently supported");
                    }
                    if grid.topology().cell_type(trial_cell_tindex).unwrap()
                        != ReferenceCellType::Quadrilateral
                    {
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
                };

                let test_points =
                    Array2D::from_data(singular_rule.test_points, (singular_rule.npoints, 2));
                let trial_points =
                    Array2D::from_data(singular_rule.trial_points, (singular_rule.npoints, 2));
                let mut test_table = test_space
                    .element()
                    .create_tabulate_array(0, singular_rule.npoints);
                let mut trial_table = trial_space
                    .element()
                    .create_tabulate_array(0, singular_rule.npoints);

                test_space
                    .element()
                    .tabulate(&test_points, 0, &mut test_table);
                trial_space
                    .element()
                    .tabulate(&trial_points, 0, &mut trial_table);

                let mut test_jdet = vec![0.0; singular_rule.npoints];
                let mut trial_jdet = vec![0.0; singular_rule.npoints];

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

                        for index in 0..singular_rule.npoints {
                            unsafe {
                                *test_pt.get_unchecked_mut(0, 0) =
                                    *test_points.get_unchecked(index, 0);
                                *test_pt.get_unchecked_mut(0, 1) =
                                    *test_points.get_unchecked(index, 1);
                                *trial_pt.get_unchecked_mut(0, 0) =
                                    *trial_points.get_unchecked(index, 0);
                                *trial_pt.get_unchecked_mut(0, 1) =
                                    *trial_points.get_unchecked(index, 1);
                            }
                            grid.geometry().compute_points(
                                &test_pt,
                                test_cell_gindex,
                                &mut test_mapped_pt,
                            );
                            grid.geometry().compute_points(
                                &trial_pt,
                                trial_cell_gindex,
                                &mut trial_mapped_pt,
                            );
                            if needs_test_normal {
                                grid.geometry().compute_normals(
                                    &test_pt,
                                    test_cell_gindex,
                                    &mut test_normal,
                                );
                            }
                            if needs_trial_normal {
                                grid.geometry().compute_normals(
                                    &trial_pt,
                                    trial_cell_gindex,
                                    &mut trial_normal,
                                );
                            }

                            let weight = singular_rule.weights[index];

                            sum += kernel(
                                unsafe { test_mapped_pt.row_unchecked(0) },
                                unsafe { trial_mapped_pt.row_unchecked(0) },
                                unsafe { test_normal.row_unchecked(0) },
                                unsafe { trial_normal.row_unchecked(0) },
                            ) * weight
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
    }

    matrix
}

fn hypersingular_assemble(
    kernel: fn(&[f64], &[f64], &[f64], &[f64]) -> f64,
    trial_space: &impl FunctionSpace,
    test_space: &impl FunctionSpace,
) -> Array2D<f64> {
    // Note: currently assumes that the two grids are the same
    // TODO: implement == and != for grids, then add:
    // if *trial_space.grid() != *test_space.grid() {
    //    unimplemented!("Assembling operators with spaces on different grids not yet supported");
    // }
    let npoints = 4;

    let grid = trial_space.grid();

    let c20 = grid.topology().connectivity(2, 0);

    // Assign working memory
    let mut test_pt = Array2D::<f64>::new((1, 2));
    let mut trial_pt = Array2D::<f64>::new((1, 2));
    let mut test_mapped_pt = Array2D::<f64>::new((1, 3));
    let mut trial_mapped_pt = Array2D::<f64>::new((1, 3));

    let mut test_normal = Array2D::<f64>::new((1, 3));
    let mut trial_normal = Array2D::<f64>::new((1, 3));

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
            if pairs.len() == 0 {
                // Standard quadrature
                let test_rule = simplex_rule(
                    grid.topology().cell_type(test_cell_tindex).unwrap(),
                    npoints_test_cell,
                )
                .unwrap();
                let trial_rule = simplex_rule(
                    grid.topology().cell_type(trial_cell_tindex).unwrap(),
                    npoints_trial_cell,
                )
                .unwrap();

                let test_points = Array2D::from_data(test_rule.points, (test_rule.npoints, 2));
                let trial_points = Array2D::from_data(trial_rule.points, (trial_rule.npoints, 2));
                let mut test_table = test_space
                    .element()
                    .create_tabulate_array(1, test_rule.npoints);
                let mut trial_table = trial_space
                    .element()
                    .create_tabulate_array(1, trial_rule.npoints);

                test_space
                    .element()
                    .tabulate(&test_points, 1, &mut test_table);
                trial_space
                    .element()
                    .tabulate(&trial_points, 1, &mut trial_table);

                let mut test_jdet = vec![0.0; test_rule.npoints];
                let mut trial_jdet = vec![0.0; trial_rule.npoints];
                let mut test_jinv = Array2D::<f64>::new((test_rule.npoints, 6));
                let mut trial_jinv = Array2D::<f64>::new((trial_rule.npoints, 6));

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

                        for test_index in 0..test_rule.npoints {
                            unsafe {
                                *test_pt.get_unchecked_mut(0, 0) =
                                    *test_points.get_unchecked(test_index, 0);
                                *test_pt.get_unchecked_mut(0, 1) =
                                    *test_points.get_unchecked(test_index, 1);
                            }
                            grid.geometry().compute_points(
                                &test_pt,
                                test_cell_gindex,
                                &mut test_mapped_pt,
                            );
                            grid.geometry().compute_normals(
                                &test_pt,
                                test_cell_gindex,
                                &mut test_normal,
                            );
                            let test_weight = test_rule.weights[test_index];

                            for trial_index in 0..trial_rule.npoints {
                                unsafe {
                                    *trial_pt.get_unchecked_mut(0, 0) =
                                        *trial_points.get_unchecked(trial_index, 0);
                                    *trial_pt.get_unchecked_mut(0, 1) =
                                        *trial_points.get_unchecked(trial_index, 1);
                                }
                                grid.geometry().compute_points(
                                    &trial_pt,
                                    trial_cell_gindex,
                                    &mut trial_mapped_pt,
                                );
                                grid.geometry().compute_normals(
                                    &trial_pt,
                                    trial_cell_gindex,
                                    &mut trial_normal,
                                );
                                let trial_weight = trial_rule.weights[trial_index];

                                let g0 = (
                                    unsafe {
                                        *trial_jinv.get_unchecked(trial_index, 0)
                                            * *trial_table.get_unchecked(1, trial_index, trial_i, 0)
                                            + *trial_jinv.get_unchecked(trial_index, 3)
                                                * *trial_table.get_unchecked(
                                                    2,
                                                    trial_index,
                                                    trial_i,
                                                    0,
                                                )
                                    },
                                    unsafe {
                                        *trial_jinv.get_unchecked(trial_index, 1)
                                            * *trial_table.get_unchecked(1, trial_index, trial_i, 0)
                                            + *trial_jinv.get_unchecked(trial_index, 4)
                                                * *trial_table.get_unchecked(
                                                    2,
                                                    trial_index,
                                                    trial_i,
                                                    0,
                                                )
                                    },
                                    unsafe {
                                        *trial_jinv.get_unchecked(trial_index, 2)
                                            * *trial_table.get_unchecked(1, trial_index, trial_i, 0)
                                            + *trial_jinv.get_unchecked(trial_index, 5)
                                                * *trial_table.get_unchecked(
                                                    2,
                                                    trial_index,
                                                    trial_i,
                                                    0,
                                                )
                                    },
                                );
                                let g1 = (
                                    unsafe {
                                        *test_jinv.get_unchecked(test_index, 0)
                                            * *test_table.get_unchecked(1, test_index, test_i, 0)
                                            + *test_jinv.get_unchecked(test_index, 3)
                                                * *test_table
                                                    .get_unchecked(2, test_index, test_i, 0)
                                    },
                                    unsafe {
                                        *test_jinv.get_unchecked(test_index, 1)
                                            * *test_table.get_unchecked(1, test_index, test_i, 0)
                                            + *test_jinv.get_unchecked(test_index, 4)
                                                * *test_table
                                                    .get_unchecked(2, test_index, test_i, 0)
                                    },
                                    unsafe {
                                        *test_jinv.get_unchecked(test_index, 2)
                                            * *test_table.get_unchecked(1, test_index, test_i, 0)
                                            + *test_jinv.get_unchecked(test_index, 5)
                                                * *test_table
                                                    .get_unchecked(2, test_index, test_i, 0)
                                    },
                                );
                                let n0 = (
                                    unsafe { *trial_normal.get_unchecked(0, 0) },
                                    unsafe { *trial_normal.get_unchecked(0, 1) },
                                    unsafe { *trial_normal.get_unchecked(0, 2) },
                                );
                                let n1 = (
                                    unsafe { *test_normal.get_unchecked(0, 0) },
                                    unsafe { *test_normal.get_unchecked(0, 1) },
                                    unsafe { *test_normal.get_unchecked(0, 2) },
                                );

                                let dot_curls = (g0.0 * g1.0 + g0.1 * g1.1 + g0.2 * g1.2)
                                    * (n0.0 * n1.0 + n0.1 * n1.1 + n0.2 * n1.2)
                                    - (g0.0 * n1.0 + g0.1 * n1.1 + g0.2 * n1.2)
                                        * (n0.0 * g1.0 + n0.1 * g1.1 + n0.2 * g1.2);

                                sum += kernel(
                                    unsafe { test_mapped_pt.row_unchecked(0) },
                                    unsafe { trial_mapped_pt.row_unchecked(0) },
                                    unsafe { test_normal.row_unchecked(0) },
                                    unsafe { trial_normal.row_unchecked(0) },
                                ) * test_weight
                                    * trial_weight
                                    * dot_curls
                                    * test_jdet[test_index]
                                    * trial_jdet[trial_index];
                            }
                        }
                        *matrix.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                    }
                }
            } else {
                // Singular quadrature
                let singular_rule = if grid.topology().cell_type(test_cell_tindex).unwrap()
                    == ReferenceCellType::Triangle
                {
                    if grid.topology().cell_type(trial_cell_tindex).unwrap()
                        != ReferenceCellType::Triangle
                    {
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
                    if grid.topology().cell_type(test_cell_tindex).unwrap()
                        != ReferenceCellType::Quadrilateral
                    {
                        unimplemented!("Only triangles and quadrilaterals are currently supported");
                    }
                    if grid.topology().cell_type(trial_cell_tindex).unwrap()
                        != ReferenceCellType::Quadrilateral
                    {
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
                };

                let test_points =
                    Array2D::from_data(singular_rule.test_points, (singular_rule.npoints, 2));
                let trial_points =
                    Array2D::from_data(singular_rule.trial_points, (singular_rule.npoints, 2));
                let mut test_table = test_space
                    .element()
                    .create_tabulate_array(1, singular_rule.npoints);
                let mut trial_table = trial_space
                    .element()
                    .create_tabulate_array(1, singular_rule.npoints);

                test_space
                    .element()
                    .tabulate(&test_points, 1, &mut test_table);
                trial_space
                    .element()
                    .tabulate(&trial_points, 1, &mut trial_table);

                let mut test_jdet = vec![0.0; singular_rule.npoints];
                let mut trial_jdet = vec![0.0; singular_rule.npoints];
                let mut test_jinv = Array2D::<f64>::new((singular_rule.npoints, 6));
                let mut trial_jinv = Array2D::<f64>::new((singular_rule.npoints, 6));

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

                        for index in 0..singular_rule.npoints {
                            unsafe {
                                *test_pt.get_unchecked_mut(0, 0) =
                                    *test_points.get_unchecked(index, 0);
                                *test_pt.get_unchecked_mut(0, 1) =
                                    *test_points.get_unchecked(index, 1);
                                *trial_pt.get_unchecked_mut(0, 0) =
                                    *trial_points.get_unchecked(index, 0);
                                *trial_pt.get_unchecked_mut(0, 1) =
                                    *trial_points.get_unchecked(index, 1);
                            }
                            grid.geometry().compute_points(
                                &test_pt,
                                test_cell_gindex,
                                &mut test_mapped_pt,
                            );
                            grid.geometry().compute_points(
                                &trial_pt,
                                trial_cell_gindex,
                                &mut trial_mapped_pt,
                            );
                            grid.geometry().compute_normals(
                                &test_pt,
                                test_cell_gindex,
                                &mut test_normal,
                            );
                            grid.geometry().compute_normals(
                                &trial_pt,
                                trial_cell_gindex,
                                &mut trial_normal,
                            );

                            let weight = singular_rule.weights[index];

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
                                unsafe { *trial_normal.get_unchecked(0, 0) },
                                unsafe { *trial_normal.get_unchecked(0, 1) },
                                unsafe { *trial_normal.get_unchecked(0, 2) },
                            );
                            let n1 = (
                                unsafe { *test_normal.get_unchecked(0, 0) },
                                unsafe { *test_normal.get_unchecked(0, 1) },
                                unsafe { *test_normal.get_unchecked(0, 2) },
                            );

                            let dot_curls = (g0.0 * g1.0 + g0.1 * g1.1 + g0.2 * g1.2)
                                * (n0.0 * n1.0 + n0.1 * n1.1 + n0.2 * n1.2)
                                - (g0.0 * n1.0 + g0.1 * n1.1 + g0.2 * n1.2)
                                    * (n0.0 * g1.0 + n0.1 * g1.1 + n0.2 * g1.2);

                            sum += kernel(
                                unsafe { test_mapped_pt.row_unchecked(0) },
                                unsafe { trial_mapped_pt.row_unchecked(0) },
                                unsafe { test_normal.row_unchecked(0) },
                                unsafe { trial_normal.row_unchecked(0) },
                            ) * weight
                                * dot_curls
                                * test_jdet[index]
                                * trial_jdet[index];
                        }
                        *matrix.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                    }
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
