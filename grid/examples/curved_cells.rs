use solvers_grid::grid::SerialGrid;
use solvers_grid::io::export_as_gmsh;
use solvers_tools::arrays::{AdjacencyList, Array2D};
use solvers_traits::cell::ReferenceCellType;
use solvers_traits::grid::{Geometry, Grid, Topology};

fn main() {
    // Create a grid of three cells. One cell is a curved triangle, one cell is a flat triangle, the other is a curved quadrilateral
    let mut grid = SerialGrid::new(
        Array2D::from_data(
            vec![
                0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.25, 0.0, 0.0, 0.5, 0.5, 0.5,
                0.5, 0.5, 1.0, 0.5, 0.5, 2.0, 0.5, 0.0, 1.5, 0.75, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0,
                0.0, 1.0, 1.0, 0.0, 2.0, -0.5, 0.0,
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
        let c20 = grid.topology_mut().connectivity(2, 0).row(ti).unwrap();
        if ct == ReferenceCellType::Triangle {
            println!(
                "The (topological) vertices of cell {} are {}, {}, and {}",
                i, c20[0], c20[1], c20[2]
            );
        } else {
            println!(
                "The (topological) vertices of cell {} are {}, {}, {}, and {}",
                i, c20[0], c20[1], c20[2], c20[3]
            );
        }
        println!("The geometric points for this cell are:");
        for pti in grid.geometry().cell_vertices(i).unwrap() {
            let pt = grid.geometry().point(*pti).unwrap();
            println!("  {} {} {}", pt[0], pt[1], pt[2]);
        }
        println!("");
    }

    // Export the grid in gmsh format
    export_as_gmsh(&mut grid, String::from("_examples_curved_cells.msh"));
}
