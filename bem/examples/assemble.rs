use approx::*;
use bempp_bem::dofmap::SerialDofMap;
use bempp_element::element::LagrangeElementTriangleDegree0;
use bempp_grid::shapes::regular_sphere;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::types::CellToCellConnectivity;
use bempp_tools::arrays::Array2D;
use bempp_traits::bem::DofMap;
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn laplace_green(x1: f64, x2: f64, x3: f64, y1: f64, y2: f64, y3: f64) -> f64 {
    let inv_dist =
        1.0 / f64::sqrt((x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2) + (x3 - y3) * (x3 - y3));

    0.25 * std::f64::consts::FRAC_1_PI * inv_dist
}

fn main() {
    let grid = regular_sphere(0);
    let element = LagrangeElementTriangleDegree0 {};
    let dofmap = SerialDofMap::new(&grid, &element);

    let npoints = 3;
    let same_triangle_rule = triangle_duffy(
        &CellToCellConnectivity {
            connectivity_dimension: 2,
            local_indices: Vec::new(),
        },
        npoints,
    )
    .unwrap();

    let test_points = Array2D::from_data(
        same_triangle_rule.test_points,
        (same_triangle_rule.npoints, 2),
    );
    let trial_points = Array2D::from_data(
        same_triangle_rule.trial_points,
        (same_triangle_rule.npoints, 2),
    );
    let mut test_table = element.create_tabulate_array(0, same_triangle_rule.npoints);
    let mut trial_table = element.create_tabulate_array(0, same_triangle_rule.npoints);

    element.tabulate(&test_points, 0, &mut test_table);
    element.tabulate(&trial_points, 0, &mut trial_table);

    let c20 = grid.topology().connectivity(2, 0);

    // Assign working memory
    let mut pts = Array2D::<f64>::new((2, 2));
    let mut mapped_pts = Array2D::<f64>::new((2, 3));
    let mut test_jdet = vec![0.0; same_triangle_rule.npoints];
    let mut trial_jdet = vec![0.0; same_triangle_rule.npoints];

    let mut test_pt = Array2D::<f64>::new((1, 2));
    let mut trial_pt = Array2D::<f64>::new((1, 2));
    let mut test_mapped_pt = Array2D::<f64>::new((1, 3));
    let mut trial_mapped_pt = Array2D::<f64>::new((1, 3));

    let mut matrix = Array2D::<f64>::new((dofmap.global_size(), dofmap.global_size()));

    // TODO: index map for geometry and topology

    for cell0 in 0..grid.geometry().cell_count() {
        let cell0_tindex = grid.topology().index_map()[cell0];
        let cell0_gindex = grid.geometry().index_map()[cell0];

        grid.geometry()
            .compute_jacobian_determinants(&test_points, cell0_gindex, &mut test_jdet);
        grid.geometry()
            .compute_jacobian_determinants(&trial_points, cell0_gindex, &mut trial_jdet);

        for (test_i, test_dof) in dofmap.cell_dofs(cell0_tindex).unwrap().iter().enumerate() {
            for (trial_i, trial_dof) in dofmap.cell_dofs(cell0_tindex).unwrap().iter().enumerate() {
                let mut sum = 0.0;

                for index in 0..same_triangle_rule.npoints {
                    unsafe {
                        *pts.get_unchecked_mut(0, 0) = *test_points.get_unchecked(index, 0);
                        *pts.get_unchecked_mut(0, 1) = *test_points.get_unchecked(index, 1);
                        *pts.get_unchecked_mut(1, 0) = *trial_points.get_unchecked(index, 0);
                        *pts.get_unchecked_mut(1, 1) = *trial_points.get_unchecked(index, 1);
                    }
                    grid.geometry()
                        .compute_points(&pts, cell0_gindex, &mut mapped_pts);
                    let weight = same_triangle_rule.weights[index];

                    sum += laplace_green(
                        unsafe { *mapped_pts.get_unchecked(0, 0) },
                        unsafe { *mapped_pts.get_unchecked(0, 1) },
                        unsafe { *mapped_pts.get_unchecked(0, 2) },
                        unsafe { *mapped_pts.get_unchecked(1, 0) },
                        unsafe { *mapped_pts.get_unchecked(1, 1) },
                        unsafe { *mapped_pts.get_unchecked(1, 2) },
                    ) * weight
                        * unsafe { test_table.get_unchecked(0, index, test_i, 0) }
                        * test_jdet[index]
                        * unsafe { trial_table.get_unchecked(0, index, trial_i, 0) }
                        * trial_jdet[index];
                }
                *matrix.get_mut(*test_dof, *trial_dof).unwrap() += sum;
            }
        }
        for cell1 in grid.topology().adjacent_cells(cell0_tindex).iter() {
            if cell1.1 < 3 {
                let test_cell_tindex = cell0_tindex;
                let test_cell_gindex = cell0_gindex;
                let trial_cell_tindex = grid.topology().index_map()[cell1.0];
                let trial_cell_gindex = grid.geometry().index_map()[cell1.0];
                let test_vertices = c20.row(test_cell_tindex).unwrap();
                let trial_vertices = c20.row(trial_cell_tindex).unwrap();
                let mut pairs = vec![];
                for (test_i, test_v) in test_vertices.iter().enumerate() {
                    for (trial_i, trial_v) in trial_vertices.iter().enumerate() {
                        if test_v == trial_v {
                            pairs.push((test_i, trial_i));
                        }
                    }
                }
                let edge_adjacent_rule = triangle_duffy(
                    &CellToCellConnectivity {
                        connectivity_dimension: cell1.1 - 1,
                        local_indices: pairs,
                    },
                    npoints,
                )
                .unwrap();

                let ea_test_points = Array2D::from_data(
                    edge_adjacent_rule.test_points,
                    (edge_adjacent_rule.npoints, 2),
                );
                let ea_trial_points = Array2D::from_data(
                    edge_adjacent_rule.trial_points,
                    (edge_adjacent_rule.npoints, 2),
                );
                let mut test_table = element.create_tabulate_array(0, edge_adjacent_rule.npoints);
                let mut trial_table = element.create_tabulate_array(0, edge_adjacent_rule.npoints);

                element.tabulate(&ea_test_points, 0, &mut test_table);
                element.tabulate(&ea_trial_points, 0, &mut trial_table);

                let mut ea_test_jdet = vec![0.0; edge_adjacent_rule.npoints];
                let mut ea_trial_jdet = vec![0.0; edge_adjacent_rule.npoints];

                grid.geometry().compute_jacobian_determinants(
                    &ea_test_points,
                    test_cell_gindex,
                    &mut ea_test_jdet,
                );
                grid.geometry().compute_jacobian_determinants(
                    &ea_trial_points,
                    trial_cell_gindex,
                    &mut ea_trial_jdet,
                );

                for (test_i, test_dof) in dofmap
                    .cell_dofs(test_cell_tindex)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for (trial_i, trial_dof) in dofmap
                        .cell_dofs(trial_cell_tindex)
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let mut sum = 0.0;

                        for index in 0..edge_adjacent_rule.npoints {
                            unsafe {
                                *test_pt.get_unchecked_mut(0, 0) =
                                    *ea_test_points.get_unchecked(index, 0);
                                *test_pt.get_unchecked_mut(0, 1) =
                                    *ea_test_points.get_unchecked(index, 1);
                                *trial_pt.get_unchecked_mut(0, 0) =
                                    *ea_trial_points.get_unchecked(index, 0);
                                *trial_pt.get_unchecked_mut(0, 1) =
                                    *ea_trial_points.get_unchecked(index, 1);
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
                            let weight = edge_adjacent_rule.weights[index];

                            sum += laplace_green(
                                unsafe { *test_mapped_pt.get_unchecked(0, 0) },
                                unsafe { *test_mapped_pt.get_unchecked(0, 1) },
                                unsafe { *test_mapped_pt.get_unchecked(0, 2) },
                                unsafe { *trial_mapped_pt.get_unchecked(0, 0) },
                                unsafe { *trial_mapped_pt.get_unchecked(0, 1) },
                                unsafe { *trial_mapped_pt.get_unchecked(0, 2) },
                            ) * weight
                                * unsafe { test_table.get_unchecked(0, index, test_i, 0) }
                                * ea_test_jdet[index]
                                * unsafe { trial_table.get_unchecked(0, index, trial_i, 0) }
                                * ea_trial_jdet[index];
                        }
                        *matrix.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                    }
                }
            }
        }
    }

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

    println!("Laplace single layer matrix (from Bempp-cl)");
    for i in 0..dofmap.global_size() {
        println!("{:?}", from_cl[i]);
    }

    println!("Laplace single layer matrix");
    for i in 0..dofmap.global_size() {
        println!("{:?}", matrix.row(i).unwrap());
    }

    for i in 0..8 {
        for j in 0..8 {
            if matrix.get(i, j).unwrap().abs() > 0.0001 {
                println!(
                    "entry ({},{})  cl: {}  rs: {}  error: {}",
                    i,
                    j,
                    from_cl[i][j],
                    matrix.get(i, j).unwrap(),
                    (matrix.get(i, j).unwrap() - from_cl[i][j]).abs() / from_cl[i][j]
                );
            }
        }
    }
    for i in 0..8 {
        for j in 0..8 {
            if matrix.get(i, j).unwrap().abs() > 0.0001 {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), from_cl[i][j], epsilon = 0.0001);
            }
        }
    }
}
