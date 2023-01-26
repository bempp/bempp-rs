//! Functions for inputting and outputting grids to/from file
use crate::grid::SerialGrid;
use solvers_traits::cell::ReferenceCellType;
use solvers_traits::grid::{Geometry, Grid, Topology};
use std::fs;

/// Export a grid as a gmsh file
pub fn export_as_gmsh(grid: &SerialGrid, fname: String) {
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
    let cell_count = grid.geometry().cell_count();
    gmsh_s.push_str(&format!("{cell_count}\n"));
    let mut coordinate_element = &grid.geometry().coordinate_elements()[0];
    let mut element_i = 0;
    for i in 0..cell_count {
        while element_i + 1 < grid.geometry().element_changes().len()
            && grid.geometry().element_changes()[element_i + 1] == i
        {
            element_i += 1;
            coordinate_element = &grid.geometry().coordinate_elements()[element_i];
        }
        let cell = grid.geometry().cell_vertices(i).unwrap();
        gmsh_s.push_str(&format!("{i} "));
        let vertex_order: Vec<usize>;
        if coordinate_element.cell_type() == ReferenceCellType::Triangle {
            vertex_order = match coordinate_element.degree() {
                1 => {
                    gmsh_s.push_str("2");
                    vec![0, 1, 2]
                }
                2 => {
                    gmsh_s.push_str("9");
                    vec![0, 1, 2, 5, 3, 4]
                }
                3 => {
                    gmsh_s.push_str("21");
                    vec![0, 1, 2, 7, 8, 3, 4, 6, 5, 9]
                }
                4 => {
                    gmsh_s.push_str("23");
                    vec![/* TODO */]
                }
                5 => {
                    gmsh_s.push_str("25");
                    vec![/* TODO */]
                }
                _ => {
                    panic!("Unsupported degree");
                }
            };
        } else if coordinate_element.cell_type() == ReferenceCellType::Quadrilateral {
            vertex_order = match coordinate_element.degree() {
                1 => {
                    gmsh_s.push_str("3");
                    vec![0, 1, 3, 2]
                }
                2 => {
                    gmsh_s.push_str("10");
                    vec![0, 1, 3, 2, 4, 6, 7, 5, 8]
                }
                _ => {
                    panic!("Unsupported degree");
                }
            };
        } else {
            panic!("Unsupported cell type.");
        }
        gmsh_s.push_str(" 2 0 0");
        for j in vertex_order {
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
    use solvers_traits::cell::ReferenceCellType;

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
            vec![ReferenceCellType::Quadrilateral; 4],
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
            vec![
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Quadrilateral,
                ReferenceCellType::Triangle,
                ReferenceCellType::Triangle,
                ReferenceCellType::Quadrilateral,
            ],
        );
        export_as_gmsh(&g, String::from("test_io_screen_mixed.msh"));
    }
}
