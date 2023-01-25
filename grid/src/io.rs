//! Functions for inputting and outputting grids to/from file
use solvers_traits::grid::Geometry;
use solvers_traits::grid::Grid;
use solvers_traits::grid::Topology;
use std::fs;

/// Export a grid as a gmsh file
pub fn export_as_gmsh(grid: &impl Grid, fname: String) {
    let mut gmsh_s = String::from("");
    gmsh_s.push_str("$MeshFormat\n");
    gmsh_s.push_str("2.2 0 8\n");
    gmsh_s.push_str("$EndMeshFormat\n");
    gmsh_s.push_str("$Nodes\n");
    let node_count = grid.geometry().point_count();
    gmsh_s.push_str(&format!("{node_count}\n"));
    for i in 0..node_count {
        gmsh_s.push_str(&format!("{i}"));
        let pt = grid.geometry().point(i).unwrap();
        for j in pt {
            gmsh_s.push_str(&format!(" {j}"));
        }
        for _ in grid.geometry().dim()..3 {
            gmsh_s.push_str(&format!(" 0.0"));
        }
        gmsh_s.push_str("\n");
    }
    gmsh_s.push_str("$EndNodes\n");
    gmsh_s.push_str("$Elements\n");
    let cell_count = grid.topology().entity_count(grid.topology().dim());
    gmsh_s.push_str(&format!("{cell_count}\n"));
    for i in 0..cell_count {
        let cell = grid.topology().cell(i).unwrap();
        gmsh_s.push_str(&format!("{i} "));
        let vertex_order: Vec<usize>;
        if cell.len() == 3 {
            gmsh_s.push_str("2");
            vertex_order = vec![0, 1, 2];
        } else if cell.len() == 4 {
            gmsh_s.push_str("3");
            vertex_order = vec![0, 1, 3, 2];
        } else {
            panic!("Unsupported cell type.");
        }
        gmsh_s.push_str(" 2 0 0");
        for j in vertex_order {
            // currently assumes that Geometry and Topology use the same order
            gmsh_s.push_str(&format!(" {}", cell[j]))
        }
        gmsh_s.push_str("\n");
    }
    gmsh_s.push_str("$EndElements\n");

    fs::write(fname, gmsh_s).expect("Unable to write file");
}

#[cfg(test)]
mod test {
    use crate::grid::SerialGrid;
    use crate::io::*;
    use crate::shapes::regular_sphere;
    use solvers_tools::arrays::AdjacencyList;
    use solvers_tools::arrays::Array2D;

    #[test]
    fn test_gmsh_output_regular_sphere() {
        let g = regular_sphere(2);
        export_as_gmsh(&g, String::from("test_io_sphere.msh"));
    }

    #[test]
    fn test_gmsh_output_quads() {
        let g = SerialGrid::new(
            Array2D::from_data(
                vec![
                    0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0,
                    1.0, 1.0,
                ],
                (9, 2),
            ),
            AdjacencyList::from_data(
                vec![0, 1, 3, 4, 3, 4, 6, 7, 1, 2, 4, 5, 4, 5, 7, 8],
                vec![0, 4, 8, 12, 16],
            ),
        );
        export_as_gmsh(&g, String::from("test_io_screen.msh"));
    }

    #[test]
    fn test_gmsh_output_mixed_cell_type() {
        let g = SerialGrid::new(
            Array2D::from_data(
                vec![
                    0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0,
                    1.0, 1.0,
                ],
                (9, 2),
            ),
            AdjacencyList::from_data(
                vec![0, 1, 4, 0, 4, 3, 1, 2, 4, 5, 3, 4, 7, 3, 7, 6, 4, 5, 7, 8],
                vec![0, 3, 6, 10, 13, 16, 20],
            ),
        );
        export_as_gmsh(&g, String::from("test_io_screen_mixed.msh"));
    }
}
