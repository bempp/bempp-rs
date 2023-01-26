use solvers_grid::io::export_as_gmsh;
use solvers_grid::shapes::regular_sphere;
use solvers_traits::grid::{Geometry, Grid, Topology};

fn main() {
    // Create a regular sphere
    let mut grid = regular_sphere(6);

    // Get the number of points in the geometry
    println!(
        "The grid has {} points in its geometry",
        grid.geometry().point_count()
    );

    // Print the number of points and cells in the topology
    grid.topology_mut().create_connectivity(0, 0);
    println!(
        "The grid has {} points in its topology",
        grid.topology().entity_count(0)
    );
    println!("The grid has {} cells", grid.topology().entity_count(2));

    grid.topology_mut().create_connectivity(2, 0);

    // Print information about the first four cells
    for i in 0..4 {
        println!("");

        // Print the topological vertices of a cell
        let t_vertices = grid.topology().connectivity(2, 0).row(i).unwrap();
        println!(
            "Triangle {} has vertices with topological numbers {}, {}, and {}",
            i, t_vertices[0], t_vertices[1], t_vertices[2]
        );

        // Use local2global to get the global number of the given cell
        let cell_n = grid.topology().local2global(i);
        println!("Triangle {}'s global id is {}", i, cell_n);
        let g_vertices = grid.geometry().cell_vertices(cell_n).unwrap();

        // Print the geometric vertices of a cell
        // NOTE: for curved cells, there will be more geometric vertices than topological vertices, and the numbering
        // will not necessarily match the numbering of the topological vertices.
        println!("The geometric vertices of triangle {} are:", i);
        for v in g_vertices {
            let coords = grid.geometry().point(*v).unwrap();
            println!(
                "  Vertex {} with coordinates ({}, {}, {})",
                v, coords[0], coords[1], coords[2]
            );
        }
    }

    // Save the mesh in gmsh format
    export_as_gmsh(&grid, String::from("examples_grid.msh"));
}
