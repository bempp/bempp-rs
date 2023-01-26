//! Functions for inputting and outputting grids to/from file
use solvers_traits::grid::Geometry;
use solvers_traits::grid::Grid;
use solvers_traits::grid::Topology;
use std::fs;

/// Export a grid as a gmsh file
pub fn export_as_gmsh(grid: &mut impl Grid, fname: String) {
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
    let cell_count = grid.topology_mut().entity_count(tdim);

    // Note: This can and will be tidied up once higher order geometry is supported
    let mut ntriangles = 0;
    let mut nquads = 0;
    for i in 0..cell_count {
        let cell = grid.topology().cell(i).unwrap();
        if cell.len() == 3 {
            ntriangles += 1;
        } else if cell.len() == 4 {
            nquads += 1;
        } else {
            panic!("Unsupported cell type.");
        }
    }

    let nblocks = {
        if ntriangles > 0 {
            1
        } else {
            0
        }
    } + {
        if nquads > 0 {
            1
        } else {
            0
        }
    };

    gmsh_s.push_str(&format!("{nblocks} {cell_count} 1 {cell_count}\n"));

    let mut cell_i = 1;

    if ntriangles > 0 {
        gmsh_s.push_str(&format!("2 1 2 {ntriangles}\n"));

        for i in 0..cell_count {
            let cell = grid.topology().cell(i).unwrap();
            if cell.len() == 3 {
                gmsh_s.push_str(&format!("{cell_i}"));
                for j in [0, 1, 2] {
                    // currently assumes that Geometry and Topology use the same order
                    gmsh_s.push_str(&format!(" {}", cell[j] + 1))
                }
                gmsh_s.push_str("\n");
                cell_i += 1;
            }
        }
    }
    if nquads > 0 {
        gmsh_s.push_str(&format!("2 1 3 {nquads}\n"));

        for i in 0..cell_count {
            let cell = grid.topology().cell(i).unwrap();
            if cell.len() == 4 {
                gmsh_s.push_str(&format!("{cell_i}"));
                for j in [0, 1, 3, 2] {
                    // currently assumes that Geometry and Topology use the same order
                    gmsh_s.push_str(&format!(" {}", cell[j] + 1))
                }
                gmsh_s.push_str("\n");
                cell_i += 1;
            }
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
    use solvers_tools::arrays::AdjacencyList;
    use solvers_tools::arrays::Array2D;

    #[test]
    fn test_gmsh_output_regular_sphere() {
        let mut g = regular_sphere(2);
        export_as_gmsh(&mut g, String::from("test_io_sphere.msh"));
    }

    #[test]
    fn test_gmsh_output_quads() {
        let mut g = SerialGrid::new(
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
        export_as_gmsh(&mut g, String::from("test_io_screen.msh"));
    }

    #[test]
    fn test_gmsh_output_mixed_cell_type() {
        let mut g = SerialGrid::new(
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
        export_as_gmsh(&mut g, String::from("test_io_screen_mixed.msh"));
    }
}
