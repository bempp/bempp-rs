//! Functions for inputting and outputting grids to/from file
use crate::grid::SerialGrid;
use bempp_traits::cell::ReferenceCellType;
use bempp_traits::grid::{Geometry, Grid, Topology};
use std::fs;

fn get_permutation_to_gmsh(cell_type: ReferenceCellType, degree: usize) -> Vec<usize> {
    match cell_type {
        ReferenceCellType::Triangle => match degree {
            1 => vec![0, 1, 2],
            2 => vec![0, 1, 2, 5, 3, 4],
            3 => vec![0, 1, 2, 7, 8, 3, 4, 6, 5, 9],
            4 => vec![0, 1, 2, 9, 10, 11, 3, 4, 5, 8, 7, 6, 12, 13, 14],
            5 => vec![
                0, 1, 2, 11, 12, 13, 14, 3, 4, 5, 6, 10, 9, 8, 7, 15, 16, 17, 18, 19, 20,
            ],
            _ => {
                panic!("Unsupported degree");
            }
        },
        ReferenceCellType::Quadrilateral => match degree {
            1 => vec![0, 1, 3, 2],
            2 => vec![0, 1, 3, 2, 4, 6, 7, 5, 8],
            _ => {
                panic!("Unsupported degree");
            }
        },
        _ => {
            panic!("Unsupported cell type.");
        }
    }
}

fn get_gmsh_cell(cell_type: ReferenceCellType, degree: usize) -> usize {
    match cell_type {
        ReferenceCellType::Triangle => match degree {
            1 => 2,
            2 => 9,
            3 => 21,
            4 => 23,
            5 => 25,
            _ => {
                panic!("Unsupported degree");
            }
        },
        ReferenceCellType::Quadrilateral => match degree {
            1 => 3,
            2 => 10,
            _ => {
                panic!("Unsupported degree");
            }
        },
        _ => {
            panic!("Unsupported cell type.");
        }
    }
}

/// Export a grid as a gmsh file
pub fn export_as_gmsh(grid: &SerialGrid, fname: String) {
    let mut gmsh_s = String::from("");
    gmsh_s.push_str("$MeshFormat\n");
    gmsh_s.push_str("4.1 0 8\n");
    gmsh_s.push_str("$EndMeshFormat\n");
    gmsh_s.push_str("$Nodes\n");
    let node_count = grid.geometry().point_count();
    gmsh_s.push_str(&format!("1 {node_count} 1 {node_count}\n"));
    gmsh_s.push_str(&format!("2 1 0 {node_count}\n"));
    for i in 0..node_count {
        gmsh_s.push_str(&format!("{}\n", i + 1));
    }
    for i in 0..node_count {
        let pt = grid.geometry().point(i).unwrap();
        for (n, j) in pt.iter().enumerate() {
            if n != 0 {
                gmsh_s.push_str(&format!(" "));
            }
            gmsh_s.push_str(&format!("{j}"));
        }
        for _ in grid.geometry().dim()..3 {
            gmsh_s.push_str(&format!(" 0.0"));
        }
        gmsh_s.push_str("\n");
    }
    gmsh_s.push_str("$EndNodes\n");
    gmsh_s.push_str("$Elements\n");

    let tdim = grid.topology().dim();
    let cell_count = grid.topology().entity_count(tdim);
    let ncoordelements = grid.geometry().coordinate_elements().len();
    gmsh_s.push_str(&format!("{ncoordelements} {cell_count} 1 {cell_count}\n"));
    for (i, element) in grid.geometry().coordinate_elements().iter().enumerate() {
        let start = grid.geometry().element_changes()[i];
        let end = {
            if i == ncoordelements - 1 {
                cell_count
            } else {
                grid.geometry().element_changes()[i + 1]
            }
        };
        gmsh_s.push_str(&format!(
            "2 1 {} {}\n",
            get_gmsh_cell(element.cell_type(), element.degree()),
            end - start
        ));
        for i in start..end {
            let cell = grid.geometry().cell_vertices(i).unwrap();
            gmsh_s.push_str(&format!("{i}"));
            for j in get_permutation_to_gmsh(element.cell_type(), element.degree()) {
                gmsh_s.push_str(&format!(" {}", cell[j] + 1))
            }
            gmsh_s.push_str("\n");
        }
    }
    gmsh_s.push_str("$EndElements\n");

    fs::write(fname, gmsh_s).expect("Unable to write file");
}

#[cfg(test)]
mod test {
    use crate::grid::SerialGrid;
    use crate::io::*;
    use crate::shapes::regular_sphere;
    use bempp_tools::arrays::AdjacencyList;
    use bempp_tools::arrays::Array2D;
    use bempp_traits::cell::ReferenceCellType;

    #[test]
    fn test_gmsh_output_regular_sphere() {
        let g = regular_sphere(2);
        export_as_gmsh(&g, String::from("_test_io_sphere.msh"));
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
        export_as_gmsh(&g, String::from("_test_io_screen.msh"));
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
        export_as_gmsh(&g, String::from("_test_io_screen_mixed.msh"));
    }
}
