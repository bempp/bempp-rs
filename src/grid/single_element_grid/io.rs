//! Input/output
use super::SerialSingleElementGrid;
use crate::grid::io::{get_gmsh_cell, get_permutation_to_gmsh};
use crate::grid::traits::Geometry;
use crate::traits::{element::FiniteElement, grid::GmshIO};
use num::Float;
use rlst::{RandomAccessByRef, RlstScalar, Shape};

impl<T: Float + RlstScalar<Real = T>> GmshIO for SerialSingleElementGrid<T> {
    fn to_gmsh_string(&self) -> String {
        let cell_count = self.geometry.cell_count();
        let node_count = self.geometry.coordinates.shape()[0];
        let edim = self.geometry.element.dim();
        let cell_type = self.geometry.element.cell_type();
        let degree = self.geometry.element.embedded_superdegree();

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
                gmsh_s.push_str(&format!(
                    "{}",
                    self.geometry.coordinates.get([i, n]).unwrap()
                ));
            }
            gmsh_s.push('\n');
        }
        gmsh_s.push_str("$EndNodes\n");
        gmsh_s.push_str("$Elements\n");

        gmsh_s.push_str(&format!("1 {cell_count} 1 {cell_count}\n"));
        gmsh_s.push_str(&format!(
            "2 1 {} {cell_count}\n",
            get_gmsh_cell(cell_type, degree)
        ));
        let gmsh_perm = get_permutation_to_gmsh(cell_type, degree);
        for i in 0..cell_count {
            gmsh_s.push_str(&format!("{}", i + 1));
            for j in &gmsh_perm {
                gmsh_s.push_str(&format!(" {}", self.geometry.cells[i * edim + *j] + 1))
            }
            gmsh_s.push('\n');
        }
        gmsh_s.push_str("$EndElements\n");

        gmsh_s
    }
}
