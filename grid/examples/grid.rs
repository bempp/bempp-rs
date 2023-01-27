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
    println!(
        "The grid has {} points in its topology",
        grid.topology_mut().entity_count(0)
    );
    println!("The grid has {} cells", grid.topology_mut().entity_count(2));

    // Print information about the first four cells
    for i in 0..4 {
        println!("");

        // Print the topological vertices of a cell
        let tcell = grid.topology().index_map()[i];
        let t_vertices = grid.topology_mut().connectivity(2, 0).row(tcell).unwrap();
        println!(
            "Triangle {} has vertices with topological numbers {}, {}, and {}",
            i, t_vertices[0], t_vertices[1], t_vertices[2]
        );

        // Print the geometric vertices of a cell
        // NOTE: for curved cells, there will be more geometric vertices than topological vertices, and the numbering
        // will not necessarily match the numbering of the topological vertices.
        let g_vertices = grid.geometry().cell_vertices(i).unwrap();
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
    export_as_gmsh(&mut grid, String::from("examples_grid.msh"));
}
