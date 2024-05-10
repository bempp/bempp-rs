use bempp::grid::single_element_grid::SingleElementGridBuilder;
use bempp::traits::{
    grid::{Builder, CellType, GeometryType, GridType, PointType},
    types::ReferenceCellType,
};

extern crate blas_src;
extern crate lapack_src;

/// Creating a single element grid
///
/// In a single element grid, the same finite element will be used to represent the geometry
/// of each cell. For example, a grid of bilinear quadrilaterals can be created by using a degree 1
/// element on a quadrilateral
fn main() {
    // When creating the grid builder, we give the physical/geometric dimension (3) and the cell type
    // and degree of the element
    let mut b = SingleElementGridBuilder::<3, f64>::new((ReferenceCellType::Quadrilateral, 1));
    // Add six points with ids 0 to 5
    b.add_point(0, [0.0, 0.0, 0.0]);
    b.add_point(1, [1.0, 0.0, 0.0]);
    b.add_point(2, [2.0, 0.0, 0.2]);
    b.add_point(3, [0.0, 1.0, 0.0]);
    b.add_point(4, [1.0, 1.0, -0.2]);
    b.add_point(5, [2.0, 1.0, 0.0]);
    // Add two cells
    b.add_cell(0, vec![0, 1, 3, 4]);
    b.add_cell(1, vec![1, 2, 4, 5]);
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
