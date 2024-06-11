use bempp::grid::flat_triangle_grid::FlatTriangleGridBuilder;
// use bempp::grid::mixed_grid::MixedGridBuilder;
// use bempp::grid::single_element_grid::SingleElementGridBuilder;
use bempp::traits::grid::{Builder, Cell, Geometry, Grid, Point, Topology};
use bempp::traits::types::ReferenceCellType;

extern crate blas_src;
extern crate lapack_src;

// #[test]
// fn test_grid_mixed_cell_type() {
//     //! Build a mixed grid using its builder
//     let mut b = MixedGridBuilder::<3, f64>::new(());
//     b.add_point(0, [-1.0, 0.0, 0.0]);
//     b.add_point(1, [-0.5, 0.0, 0.2]);
//     b.add_point(2, [0.0, 0.0, 0.0]);
//     b.add_point(3, [1.0, 0.0, 0.0]);
//     b.add_point(4, [2.0, 0.0, 0.0]);
//     b.add_point(
//         5,
//         [
//             -std::f64::consts::FRAC_1_SQRT_2,
//             std::f64::consts::FRAC_1_SQRT_2,
//             0.0,
//         ],
//     );
//     b.add_point(6, [0.0, 0.5, 0.0]);
//     b.add_point(7, [0.0, 1.0, 0.0]);
//     b.add_point(8, [1.0, 1.0, 0.0]);
//     b.add_cell(0, (vec![0, 2, 7, 6, 5, 1], ReferenceCellType::Triangle, 2));
//     b.add_cell(1, (vec![2, 3, 7, 8], ReferenceCellType::Quadrilateral, 1));
//     b.add_cell(2, (vec![3, 4, 8], ReferenceCellType::Triangle, 1));

//     let grid = b.create_grid();

//     assert_eq!(grid.number_of_vertices(), 6);
//     assert_eq!(grid.number_of_points(), 9);
//     assert_eq!(grid.number_of_cells(), 3);

//     let mut coords = vec![0.0; grid.physical_dimension()];
//     for point in grid.iter_all_points() {
//         point.coords(coords.as_mut_slice());
//         println!("{:#?}", coords);
//     }

//     for cell in grid.iter_all_cells() {
//         println!("{:?}", cell.index());
//     }
//     for cell in grid.iter_all_cells() {
//         for (local_index, (vertex_index, edge_index)) in cell
//             .topology()
//             .vertex_indices()
//             .zip(cell.topology().edge_indices())
//             .enumerate()
//         {
//             println!(
//                 "Cell: {}, Vertex: {}, {:?}, Edge: {}, {:?}, Volume: {}",
//                 cell.index(),
//                 local_index,
//                 vertex_index,
//                 local_index,
//                 edge_index,
//                 cell.geometry().volume(),
//             )
//         }
//     }
// }

// #[test]
// fn test_grid_single_element() {
//     //! Build a single element grid using its builder
//     let mut b = SingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Triangle, 2));
//     b.add_point(0, [0.0, 0.0, 0.0]);
//     b.add_point(1, [0.5, 0.0, 0.2]);
//     b.add_point(2, [1.0, 0.0, 0.0]);
//     b.add_point(3, [0.0, 0.5, 0.0]);
//     b.add_point(4, [0.5, 0.5, 0.0]);
//     b.add_point(5, [1.0, 0.5, 0.0]);
//     b.add_point(6, [0.0, 1.0, 0.0]);
//     b.add_point(7, [0.5, 1.0, 0.0]);
//     b.add_point(8, [1.0, 1.0, 0.0]);
//     b.add_cell(0, vec![0, 2, 6, 4, 3, 1]);
//     b.add_cell(1, vec![2, 8, 6, 7, 4, 5]);
//     let grid = b.create_grid();

//     assert_eq!(grid.number_of_vertices(), 4);
//     assert_eq!(grid.number_of_points(), 9);
//     assert_eq!(grid.number_of_cells(), 2);

//     let mut coords = vec![0.0; grid.physical_dimension()];
//     for point in grid.iter_all_points() {
//         point.coords(coords.as_mut_slice());
//         println!("{:#?}", coords);
//     }

//     for cell in grid.iter_all_cells() {
//         println!("{:?}", cell.index());
//     }
//     for cell in grid.iter_all_cells() {
//         for (local_index, (vertex_index, edge_index)) in cell
//             .topology()
//             .vertex_indices()
//             .zip(cell.topology().edge_indices())
//             .enumerate()
//         {
//             println!(
//                 "Cell: {}, Vertex: {}, {:?}, Edge: {}, {:?}, Volume: {}",
//                 cell.index(),
//                 local_index,
//                 vertex_index,
//                 local_index,
//                 edge_index,
//                 cell.geometry().volume(),
//             )
//         }
//     }
// }

#[test]
fn test_grid_flat_triangle() {
    //! Build a flat triangle grid using its builder
    let mut b = FlatTriangleGridBuilder::<f64>::new(());
    b.add_point(1, [0.0, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 1.0]);
    b.add_point(3, [1.0, 1.0, 0.0]);
    b.add_point(4, [0.0, 1.0, 0.0]);
    b.add_cell(0, [1, 2, 3]);
    b.add_cell(1, [2, 3, 4]);

    let grid = b.create_grid();

    assert_eq!(grid.number_of_vertices(), 4);
    assert_eq!(grid.number_of_points(), 4);
    assert_eq!(grid.number_of_cells(), 2);

    let mut coords = vec![0.0; grid.physical_dimension()];
    for point in grid.iter_all_points() {
        point.coords(coords.as_mut_slice());
        println!("{:#?}", coords);
    }

    for cell in grid.iter_all_cells() {
        println!("{:?}", cell.index());
    }
    for cell in grid.iter_all_cells() {
        for (local_index, (vertex_index, edge_index)) in cell
            .topology()
            .vertex_indices()
            .zip(cell.topology().edge_indices())
            .enumerate()
        {
            println!(
                "Cell: {}, Vertex: {}, {:?}, Edge: {}, {:?}, Volume: {}",
                cell.index(),
                local_index,
                vertex_index,
                local_index,
                edge_index,
                cell.geometry().volume(),
            )
        }
    }
}
