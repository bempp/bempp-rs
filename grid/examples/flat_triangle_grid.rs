use bempp_grid::flat_triangle_grid::SerialFlatTriangleGridBuilder;
use bempp_traits::grid::{Builder, CellType, GeometryType, GridType, PointType};

/// Creating a flat triangle grid
///
/// In a flat triangle grid, all the cells are flat triangles in 3D space.
fn main() {
    // The grid will be created using the grid builder
    let mut b = SerialFlatTriangleGridBuilder::<f64>::new(());
    // Add four points, giving them the ids 1 to 4
    b.add_point(1, [0.0, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 1.0]);
    b.add_point(3, [1.0, 1.0, 0.0]);
    b.add_point(4, [0.0, 1.0, 0.0]);
    // Add two cells. The vertex ids used above are used to define the cells
    b.add_cell(0, [1, 2, 3]);
    b.add_cell(1, [2, 3, 4]);
    // Create the grid
    let grid = b.create_grid();

    // Print the coordinates or each point in the mesh. Note that that point indices
    // start from 0 and are not equal to the ids used when inputting the points
    let mut coords = vec![0.0; grid.physical_dimension()];
    for point in grid.iter_all_points() {
        point.coords(coords.as_mut_slice());
        println!("point {} (id {}): {:#?}", point.index(), point.id(), coords);
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
