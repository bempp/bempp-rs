use bempp_grid::flat_triangle_grid::SerialFlatTriangleGridBuilder;
use bempp_traits::grid::{Builder, CellType, GeometryType, GridType, PointType};

fn main() {
    // Build a flat triangle grid using its builder
    let mut b = SerialFlatTriangleGridBuilder::<f64>::new(());
    // Add four points
    b.add_point(1, [0.0, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 1.0]);
    b.add_point(3, [1.0, 1.0, 0.0]);
    b.add_point(4, [0.0, 1.0, 0.0]);
    // Add two cells
    b.add_cell(0, [1, 2, 3]);
    b.add_cell(1, [2, 3, 4]);
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
        println!("cell {}: {:?} ", cell.index(), cell.geometry().vertices().map(|v| v.index()).collect::<Vec<_>>());
    }
}
