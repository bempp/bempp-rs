//! Input/output
use super::SerialFlatTriangleGrid;
use bempp_traits::grid::GmshIO;
use num::Float;
use rlst::{RandomAccessByRef, RlstScalar, Shape};

impl<T: Float + RlstScalar<Real = T>> GmshIO for SerialFlatTriangleGrid<T> {
    fn to_gmsh_string(&self) -> String {
        let cell_count = self.cells_to_entities[0].len();
        let node_count = self.coordinates.shape()[0];

        let mut gmsh_s = String::from("");
        gmsh_s.push_str("$MeshFormat\n");
        gmsh_s.push_str("4.1 0 8\n");
        gmsh_s.push_str("$EndMeshFormat\n");
        gmsh_s.push_str("$Nodes\n");
        gmsh_s.push_str(&format!("1 {node_count} 1 {node_count}\n"));
        gmsh_s.push_str(&format!("2 1 0 {node_count}\n"));
        for i in 0..node_count {
            gmsh_s.push_str(&format!("{}\n", i + 1));
        }
        for i in 0..node_count {
            for n in 0..3 {
                if n != 0 {
                    gmsh_s.push(' ');
                }
                gmsh_s.push_str(&format!("{}", self.coordinates.get([i, n]).unwrap()));
            }
            gmsh_s.push('\n');
        }
        gmsh_s.push_str("$EndNodes\n");
        gmsh_s.push_str("$Elements\n");

        gmsh_s.push_str(&format!("1 {cell_count} 1 {cell_count}\n"));
        gmsh_s.push_str(&format!("2 1 2 {cell_count}\n"));
        for (i, vertices) in self.cells_to_entities[0].iter().enumerate() {
            gmsh_s.push_str(&format!("{}", i + 1));
            for v in vertices {
                gmsh_s.push_str(&format!(" {}", v + 1))
            }
            gmsh_s.push('\n');
        }
        gmsh_s.push_str("$EndElements\n");

        gmsh_s
    }
}
