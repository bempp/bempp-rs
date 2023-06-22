use approx::*;
use bempp_bem::assembly::{assemble_dense, BoundaryOperator, PDEType};
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_element::element::create_element;
use bempp_grid::shapes::regular_sphere;
use bempp_tools::arrays::{Array2D, Array4D};
use bempp_traits::arrays::{Array2DAccess, Array4DAccess, AdjacencyListAccess};
use bempp_traits::bem::{DofMap, FunctionSpace};
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::element::{ElementFamily, FiniteElement};
use bempp_traits::grid::{Grid, Geometry, Topology};

use bempp_bem::green;
use bempp_bem::green::SingularKernel;
use bempp_quadrature::simplex_rules::available_rules;
use bempp_quadrature::types::CellToCellConnectivity;
use bempp_quadrature::types::TestTrialNumericalQuadratureDefinition;
use bempp_quadrature::duffy::triangle::triangle_duffy;
use bempp_quadrature::duffy::quadrilateral::quadrilateral_duffy;
use bempp_quadrature::simplex_rules::simplex_rule;

fn singular_kernel<'a>(result: &mut impl Array2DAccess<'a, f64>) {
    *result.get_mut(0, 0).unwrap() = 0.004747866583277947;
}

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

fn main() {
    let grid = regular_sphere(2);
    let element = create_element(
        ElementFamily::Lagrange,
        ReferenceCellType::Triangle,
        0,
        true,
    );

    let space = SerialFunctionSpace::new(&grid, &element);

    let mut matrix = Array2D::<f64>::new((
        space.dofmap().global_size(),
        space.dofmap().global_size(),
    ));

    assemble_dense(
        &mut matrix,
        BoundaryOperator::SingleLayer,
        PDEType::Laplace,
        &space,
        &space,
    );


    let mut matrix2 = Array2D::<f64>::new((
        space.dofmap().global_size(),
        space.dofmap().global_size(),
    ));


    // TODO: allow user to configure this
    let npoints = 4;
    let kernel = green::LaplaceGreenKernel {};


    let grid = space.grid();
    let c20 = grid.topology().connectivity(2, 0);

    let mut local_result = Array2D::<f64>::new((1, 1));

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

        singular_kernel(&mut local_result);

        let dof_i = space.dofmap().cell_dofs(test_cell_tindex).unwrap()[0];
        *matrix2.get_mut(dof_i, dof_i).unwrap() = *local_result.get(0,0).unwrap();

        for trial_cell in 0..grid.geometry().cell_count() {
            if test_cell != trial_cell {
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
                    Array4D::<f64>::new(space.element().tabulate_array_shape(0, rule.npoints));
                let mut trial_table =
                    Array4D::<f64>::new(space.element().tabulate_array_shape(0, rule.npoints));

                space
                    .element()
                    .tabulate(&test_points, 0, &mut test_table);
                space
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
                let test_normals = Array2D::<f64>::new((0, 0));
                let trial_normals = Array2D::<f64>::new((0, 0));

                grid.geometry()
                    .compute_points(&test_points, test_cell_gindex, &mut test_mapped_pts);
                grid.geometry()
                    .compute_points(&trial_points, trial_cell_gindex, &mut trial_mapped_pts);

                for (test_i, test_dof) in space
                    .dofmap()
                    .cell_dofs(test_cell_tindex)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for (trial_i, trial_dof) in space
                        .dofmap()
                        .cell_dofs(trial_cell_tindex)
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let mut sum = 0.0;

                        for index in 0..rule.npoints {
                            sum += kernel.eval::<f64>(
                                unsafe { test_mapped_pts.row_unchecked(index) },
                                unsafe { trial_mapped_pts.row_unchecked(index) },
                                unsafe { test_normals.row_unchecked(index) },
                                unsafe { trial_normals.row_unchecked(index) },
                            ) * 
                                rule.weights[index]
                                    * unsafe { test_table.get_unchecked(0, index, test_i, 0) }
                                    * test_jdet[index]
                                    * unsafe { trial_table.get_unchecked(0, index, trial_i, 0) }
                                    * trial_jdet[index];
                        }
                        *matrix2.get_mut(*test_dof, *trial_dof).unwrap() += sum;
                    }
                }
            }
        }
    }


    for i in 0..space.dofmap().global_size() {
        for j in 0..space.dofmap().global_size() {
            assert_relative_eq!(matrix.get(i, j).unwrap(), matrix2.get(i, j).unwrap());
        }
    }

}
