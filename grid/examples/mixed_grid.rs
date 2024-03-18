use bempp_grid::mixed_grid::SerialMixedGridBuilder;
use bempp_traits::{
    grid::{Builder, CellType, GeometryType, GridType, PointType},
    types::ReferenceCellType,
};

extern crate blas_src;
extern crate lapack_src;

/// Creating a mixed grid
///
/// In a mixed grid, the geometry of each cell can be represented by a different element. This allows
/// for a mixture of flat and curved cells, and/or a mixture of cell types
fn main() {
    // Create the grid builder, inputting the physical/geometric dimension (3)
    let mut b = SerialMixedGridBuilder::<3, f64>::new(());
    // Add ten points
    b.add_point(0, [0.0, 0.0, 0.0]);
    b.add_point(1, [1.0, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.5, 0.0]);
    b.add_point(3, [1.0, 1.0, 0.0]);
    b.add_point(4, [1.5, 0.0, 0.0]);
    b.add_point(5, [1.5, 0.5, -0.2]);
    b.add_point(6, [1.5, 1.0, -0.1]);
    b.add_point(7, [2.0, 0.0, 0.0]);
    b.add_point(8, [2.0, 0.5, -0.1]);
    b.add_point(9, [2.0, 1.0, 0.2]);
    // Add two cells
    // The first cell is a flat triangle, so three points are input alongside the cell type and degree
    b.add_cell(0, (vec![0, 1, 3], ReferenceCellType::Triangle, 1));
    // The second cell is a curved degree 2 quadrilateral
    // As the curved cell neighbours a flat cell, care must be taken to ensure that the
    // neighbouring edge is indeed straight. In this case, this is done by placing the point
    // with id 2 at the midpoint of the straight edge
    b.add_cell(
        1,
        (
            vec![1, 7, 3, 9, 4, 2, 8, 6, 5],
            ReferenceCellType::Quadrilateral,
            2,
        ),
    );
    // Create the grid
    let grid = b.create_grid();

    // Print the coordinates or each point in the mesh
    let mut coords = vec![0.0; grid.physical_dimension()];
    for point in grid.iter_all_points() {
        point.coords(coords.as_mut_slice());
        println!("point {}: {:#?}", point.index(), coords);
    }

    // Print the vertices of each cell
    for cell in grid.iter_all_cells() {
        println!(
            "cell {}: {:?} ",
            cell.index(),
            cell.geometry()
                .vertices()
                .map(|v| v.index())
                .collect::<Vec<_>>()
        );
    }
}
