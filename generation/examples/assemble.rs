use approx::*;
use bempp_bem::function_space::SerialFunctionSpace;
use bempp_generation::generate_kernels;
use bempp_grid::shapes::regular_sphere;
use bempp_tools::arrays::Array2D;
use bempp_traits::arrays::Array2DAccess;
use bempp_traits::bem::{DofMap, FunctionSpace, Kernel};
use bempp_traits::element::FiniteElement;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn main() {
    generate_kernels!(
        bem_kernel, "Lagrange", "Triangle", 0, true, "Lagrange", "Triangle", 0, true, "Lagrange",
        "Triangle", 1, false, "Lagrange", "Triangle", 1, false
    );

    let grid = regular_sphere(0);
    let space = SerialFunctionSpace::new(&grid, &bem_kernel.test_element);

    let ndofs = space.dofmap().global_size();

    let mut matrix = Array2D::<f64>::new((ndofs, ndofs));

    assemble(&mut matrix, &space, &space, &bem_kernel);

    // Compare to result from bempp-cl
    #[rustfmt::skip]
    let from_cl = vec![vec![0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472], vec![0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473, 0.04670742127454548], vec![0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074, 0.05963897421514473], vec![0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.05963897421514473, 0.04670742127454548, 0.05963897421514472, 0.08755414595678074], vec![0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074], vec![0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.046707421274545476, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074, 0.05963897421514472], vec![0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.05963897421514472, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487, 0.08755414595678074], vec![0.05963897421514472, 0.046707421274545476, 0.05963897421514473, 0.08755414595678074, 0.08755414595678074, 0.05963897421514472, 0.08755414595678074, 0.1854538822982487]];

    for (i, row) in from_cl.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            if i == j {
                assert_relative_eq!(*matrix.get(i, j).unwrap(), entry, epsilon = 1e-4);
            }
        }
    }
}

fn assemble<'a, E: FiniteElement>(
    matrix: &mut impl Array2DAccess<'a, f64>,
    test_space: &SerialFunctionSpace<E>,
    trial_space: &SerialFunctionSpace<E>,
    bem_kernel: &impl Kernel<f64>,
) {
    let mut local_result = Array2D::<f64>::new((
        bem_kernel.test_element_dim(),
        bem_kernel.trial_element_dim(),
    ));
    let grid = test_space.grid();

    let mut test_vertices = Array2D::<f64>::new((
        bem_kernel.test_geometry_element_dim(),
        grid.geometry().dim(),
    ));
    let mut trial_vertices = Array2D::<f64>::new((
        bem_kernel.trial_geometry_element_dim(),
        grid.geometry().dim(),
    ));

    // Test and trial cells are equal
    for test_cell in 0..grid.geometry().cell_count() {
        let test_cell_tindex = grid.topology().index_map()[test_cell];
        let test_dofs = test_space.dofmap().cell_dofs(test_cell_tindex).unwrap();
        let trial_dofs = trial_space.dofmap().cell_dofs(test_cell_tindex).unwrap();
        let test_cell_dofs = grid.geometry().cell_vertices(test_cell).unwrap();

        for (i, dof) in test_cell_dofs.iter().enumerate() {
            for (j, coord) in grid.geometry().point(*dof).unwrap().iter().enumerate() {
                *test_vertices.get_mut(i, j).unwrap() = *coord;
            }
        }

        bem_kernel.same_cell_kernel(
            &mut local_result.data,
            &test_vertices.data,
            &test_vertices.data,
        );
        for (test_i, test_dof) in test_dofs.iter().enumerate() {
            for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                *matrix.get_mut(*test_dof, *trial_dof).unwrap() =
                    *local_result.get(test_i, trial_i).unwrap()
            }
        }
    }
    /*
        for (test_cell, trial_cell, edge_info) in grid.topology().facet_adjacent_cells().iter() {
            let test_cell_tindex = grid.topology().index_map()[*test_cell];
            let test_dofs = test_space.dofmap().cell_dofs(test_cell_tindex).unwrap();
            let test_cell_dofs = grid.geometry().cell_vertices(*test_cell).unwrap();

            for i in 0..3 {
                for j in 0..3 {
                    *test_vertices.get_mut(i, j).unwrap() =
                        grid.geometry().point(test_cell_dofs[i]).unwrap()[j];
                }
            }

            let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
            let trial_dofs = space.dofmap().cell_dofs(trial_cell_tindex).unwrap();
            let trial_cell_dofs = grid.geometry().cell_vertices(*trial_cell).unwrap();

            for i in 0..3 {
                for j in 0..3 {
                    *trial_vertices.get_mut(i, j).unwrap() =
                        grid.geometry().point(trial_cell_dofs[i]).unwrap()[j];
                }
            }

            shared_edge_kernel_dp0(
                &mut local_result,
                &test_vertices,
                &trial_vertices,
                *edge_info,
            );

            for (test_i, test_dof) in test_dofs.iter().enumerate() {
                for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                    *result.get_mut(*test_dof, *trial_dof).unwrap() =
                        *local_result.get(test_i, trial_i).unwrap()
                }
            }
        }

        for (test_cell, trial_cell, vertex_info) in grid.topology().ridge_adjacent_cells().iter() {
            let test_cell_tindex = grid.topology().index_map()[*test_cell];
            let test_dofs = space.dofmap().cell_dofs(test_cell_tindex).unwrap();
            let test_cell_dofs = grid.geometry().cell_vertices(*test_cell).unwrap();

            for i in 0..3 {
                for j in 0..3 {
                    *test_vertices.get_mut(i, j).unwrap() =
                        grid.geometry().point(test_cell_dofs[i]).unwrap()[j];
                }
            }

            let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
            let trial_dofs = space.dofmap().cell_dofs(trial_cell_tindex).unwrap();
            let trial_cell_dofs = grid.geometry().cell_vertices(*trial_cell).unwrap();

            for i in 0..3 {
                for j in 0..3 {
                    *trial_vertices.get_mut(i, j).unwrap() =
                        grid.geometry().point(trial_cell_dofs[i]).unwrap()[j];
                }
            }

            shared_vertex_kernel_dp0(
                &mut local_result,
                &test_vertices,
                &trial_vertices,
                *vertex_info,
            );

            for (test_i, test_dof) in test_dofs.iter().enumerate() {
                for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                    *result.get_mut(*test_dof, *trial_dof).unwrap() =
                        *local_result.get(test_i, trial_i).unwrap()
                }
            }
        }

        for (test_cell, trial_cell) in grid.topology().nonadjacent_cells().iter() {
            let test_cell_tindex = grid.topology().index_map()[*test_cell];
            let test_dofs = space.dofmap().cell_dofs(test_cell_tindex).unwrap();
            let test_cell_dofs = grid.geometry().cell_vertices(*test_cell).unwrap();

            for i in 0..3 {
                for j in 0..3 {
                    *test_vertices.get_mut(i, j).unwrap() =
                        grid.geometry().point(test_cell_dofs[i]).unwrap()[j];
                }
            }

            let trial_cell_tindex = grid.topology().index_map()[*trial_cell];
            let trial_dofs = space.dofmap().cell_dofs(trial_cell_tindex).unwrap();
            let trial_cell_dofs = grid.geometry().cell_vertices(*trial_cell).unwrap();

            for i in 0..3 {
                for j in 0..3 {
                    *trial_vertices.get_mut(i, j).unwrap() =
                        grid.geometry().point(trial_cell_dofs[i]).unwrap()[j];
                }
            }

            nonneighbour_kernel_dp0(&mut local_result, &test_vertices, &trial_vertices);
            for (test_i, test_dof) in test_dofs.iter().enumerate() {
                for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                    *result.get_mut(*test_dof, *trial_dof).unwrap() =
                        *local_result.get(test_i, trial_i).unwrap()
                }
            }
        }
    */
}

