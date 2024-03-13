use bempp_grid::flat_triangle_grid::SerialFlatTriangleGridBuilder;
use bempp_traits::grid::{Builder, CellType, GeometryType, GridType, PointType, TopologyType};

fn main() {
    //! Build a flat triangle grid using its builder
    let mut b = SerialFlatTriangleGridBuilder::<f64>::new(());
    b.add_point(1, [0.0, 0.0, 0.0]);
    b.add_point(2, [1.0, 0.0, 1.0]);
    b.add_point(3, [1.0, 1.0, 0.0]);
    b.add_point(4, [0.0, 1.0, 0.0]);
    b.add_cell(0, [1, 2, 3]);
    b.add_cell(1, [2, 3, 4]);

    let grid = b.create_grid();

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
