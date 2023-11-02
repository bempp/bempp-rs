use bempp_grid::grid::SerialGrid;
use bempp_grid::io::export_as_gmsh;
use bempp_tools::arrays::{to_matrix, AdjacencyList};
use bempp_traits::arrays::AdjacencyListAccess;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn main() {
    // Create a grid of three cells. One cell is a curved triangle, one cell is a flat triangle, the other is a curved quadrilateral
    let grid = SerialGrid::new(
        to_matrix(
            &vec![
0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 2.0, 1.5, 0.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            (13, 3),
        ),
        AdjacencyList::from_data(
            vec![2, 7, 12, 0, 2, 9, 11, 1, 4, 6, 10, 5, 2, 7, 11, 8, 6, 3],
            vec![0, 3, 12, 18],
        ),
        vec![
            ReferenceCellType::Triangle,
            ReferenceCellType::Quadrilateral,
            ReferenceCellType::Triangle,
        ],
    );

    let c20 = grid.topology().connectivity(2, 0);

    // Print information about the cells
    for i in 0..3 {
        let ti = grid.topology().index_map()[i];
        let gi = grid.geometry().index_map()[i];
        let ct = grid.topology().cell_type(ti).unwrap();
        println!(
            "cell {} is a {}",
            i,
            if ct == ReferenceCellType::Triangle {
                "Triangle"
            } else {
                "Quadrilateral"
            }
        );
        println!(
            "cell {} is cell number {} in the toplogy and cell number {} in the geometry.",
            i, ti, gi
        );
        let vertices = c20.row(ti).unwrap();
        if ct == ReferenceCellType::Triangle {
            println!(
                "The (topological) vertices of cell {} are {}, {}, and {}",
                i, vertices[0], vertices[1], vertices[2]
            );
        } else {
            println!(
                "The (topological) vertices of cell {} are {}, {}, {}, and {}",
                i, vertices[0], vertices[1], vertices[2], vertices[3]
            );
        }
        println!("The geometric points for this cell are:");
        for pti in grid.geometry().cell_vertices(i).unwrap() {
            let pt = grid.geometry().point(*pti).unwrap();
            println!("  {} {} {}", pt[0], pt[1], pt[2]);
        }
        println!();
    }

    // Export the grid in gmsh format
    export_as_gmsh(&grid, String::from("_examples_curved_cells.msh"));
}
