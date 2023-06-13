use bempp_grid::io::export_as_gmsh;
use bempp_grid::shapes::regular_sphere;
use bempp_traits::arrays::AdjacencyListAccess;
use bempp_traits::grid::{Geometry, Grid, Topology};

fn main() {
    // Create a regular sphere
    let grid = regular_sphere(6);

    // Get the number of points in the geometry
    println!(
        "The grid has {} points in its geometry",
        grid.geometry().point_count()
    );

    // Print the number of points and cells in the topology
    println!(
        "The grid has {} points in its topology",
        grid.topology().entity_count(0)
    );
    println!("The grid has {} cells", grid.topology().entity_count(2));

    // Print information about the first four cells
    let c20 = grid.topology().connectivity(2, 0);
    for i in 0..4 {
        println!("");

        // Print the topological vertices of a cell
        let tcell = grid.topology().index_map()[i];
        let t_vertices = c20.row(tcell).unwrap();
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
    export_as_gmsh(&grid, String::from("_examples_grid.msh"));
}

#[test]
fn test_surface_area() {
    for levels in 1..7 {
        let grid = regular_sphere(levels);

        let mut area = 0.0;
        for i in 0..grid.topology().entity_count(2) {
            let v = grid.geometry().cell_vertices(i).unwrap();
            let e1 = [
                grid.geometry().point(v[1]).unwrap()[0] - grid.geometry().point(v[0]).unwrap()[0],
                grid.geometry().point(v[1]).unwrap()[1] - grid.geometry().point(v[0]).unwrap()[1],
                grid.geometry().point(v[1]).unwrap()[2] - grid.geometry().point(v[0]).unwrap()[2],
            ];
            let e2 = [
                grid.geometry().point(v[2]).unwrap()[0] - grid.geometry().point(v[0]).unwrap()[0],
                grid.geometry().point(v[2]).unwrap()[1] - grid.geometry().point(v[0]).unwrap()[1],
                grid.geometry().point(v[2]).unwrap()[2] - grid.geometry().point(v[0]).unwrap()[2],
            ];
            let c = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            area += (c[0].powi(2) + c[1].powi(2) + c[2].powi(2)).sqrt() / 2.0;
        }
        // Surface area should be lower than surface area of the sphere
        assert!(area <= 12.5663706144);
        if levels >= 4 {
            assert!(area > 12.52);
        } else if levels >= 2 {
            assert!(area > 11.9);
        } else {
            assert!(area > 10.4);
        }
    }
}
