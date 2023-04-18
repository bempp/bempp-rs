use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::simplex_rules::{available_rules, simplex_rule};
use bempp_quadrature::types::CellToCellConnectivity;
use bempp_tools::arrays::Array2D;
use bempp_traits::bem::DofMap;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn laplace_green(x1: f64, x2: f64, x3: f64, y1: f64, y2: f64, y3: f64) -> f64 {
    let inv_dist =
        1.0 / f64::sqrt((x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2) + (x3 - y3) * (x3 - y3));

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist
}

pub fn laplace_single_layer(
    grid: &impl Grid,
    trial_element: &impl FiniteElement,
    trial_dofmap: &impl DofMap,
    test_element: &impl FiniteElement,
    test_dofmap: &impl DofMap,
) -> Array2D<f64> {
    let npoints = 3;

    let c20 = grid.topology().connectivity(2, 0);

    // Assign working memory
    let mut test_pt = Array2D::<f64>::new((1, 2));
    let mut trial_pt = Array2D::<f64>::new((1, 2));
    let mut test_mapped_pt = Array2D::<f64>::new((1, 3));
    let mut trial_mapped_pt = Array2D::<f64>::new((1, 3));

    let mut matrix = Array2D::<f64>::new((test_dofmap.global_size(), trial_dofmap.global_size()));

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
                let mut test_table = test_element.create_tabulate_array(0, test_rule.npoints);
                let mut trial_table = trial_element.create_tabulate_array(0, trial_rule.npoints);

                test_element.tabulate(&test_points, 0, &mut test_table);
                trial_element.tabulate(&trial_points, 0, &mut trial_table);

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

                for (test_i, test_dof) in test_dofmap
                    .cell_dofs(test_cell_tindex)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for (trial_i, trial_dof) in trial_dofmap
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
                                let trial_weight = trial_rule.weights[trial_index];

                                sum += laplace_green(
                                    unsafe { *test_mapped_pt.get_unchecked(0, 0) },
                                    unsafe { *test_mapped_pt.get_unchecked(0, 1) },
                                    unsafe { *test_mapped_pt.get_unchecked(0, 2) },
                                    unsafe { *trial_mapped_pt.get_unchecked(0, 0) },
                                    unsafe { *trial_mapped_pt.get_unchecked(0, 1) },
                                    unsafe { *trial_mapped_pt.get_unchecked(0, 2) },
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
                let mut test_table = test_element.create_tabulate_array(0, singular_rule.npoints);
                let mut trial_table = trial_element.create_tabulate_array(0, singular_rule.npoints);

                test_element.tabulate(&test_points, 0, &mut test_table);
                trial_element.tabulate(&trial_points, 0, &mut trial_table);

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

                for (test_i, test_dof) in test_dofmap
                    .cell_dofs(test_cell_tindex)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for (trial_i, trial_dof) in trial_dofmap
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
                            let weight = singular_rule.weights[index];

                            sum += laplace_green(
                                unsafe { *test_mapped_pt.get_unchecked(0, 0) },
                                unsafe { *test_mapped_pt.get_unchecked(0, 1) },
                                unsafe { *test_mapped_pt.get_unchecked(0, 2) },
                                unsafe { *trial_mapped_pt.get_unchecked(0, 0) },
                                unsafe { *trial_mapped_pt.get_unchecked(0, 1) },
                                unsafe { *trial_mapped_pt.get_unchecked(0, 2) },
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

#[cfg(test)]
mod test {
    use crate::assembly::dense::*;
    use crate::dofmap::SerialDofMap;
    use approx::*;
    use bempp_element::element::LagrangeElementTriangleDegree0;
    use bempp_grid::shapes::regular_sphere;

    #[test]
    fn test_laplace_single_layer_dp0_dp0() {
        let grid = regular_sphere(0);
        let element = LagrangeElementTriangleDegree0 {};
        let dofmap = SerialDofMap::new(&grid, &element);

        let matrix = laplace_single_layer(&grid, &element, &dofmap, &element, &dofmap);

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
                if matrix.get(i, j).unwrap().abs() > 0.0001 {
                    assert_relative_eq!(
                        *matrix.get(i, j).unwrap(),
                        from_cl[i][j],
                        epsilon = 0.0001
                    );
                }
            }
        }
    }
}
