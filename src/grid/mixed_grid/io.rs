//! Input/output
use super::MixedGrid;
use crate::grid::io::{get_gmsh_cell, get_permutation_to_gmsh};
use crate::grid::traits::Geometry;
use crate::traits::grid::GmshIO;
use ndelement::traits::FiniteElement;
use num::Float;
use rlst::{RandomAccessByRef, RlstScalar, Shape};

impl<T: Float + RlstScalar<Real = T>> GmshIO for MixedGrid<T> {
    fn to_gmsh_string(&self) -> String {
        let cell_count = self.geometry.cell_count();
        let node_count = self.geometry.coordinates.shape()[0];

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

        gmsh_s.push_str(&format!(
            "{} {cell_count} 1 {cell_count}\n",
            self.geometry.elements.len()
        ));

        for (e_index, element) in self.geometry.elements.iter().enumerate() {
            let cell_type = element.cell_type();
            let degree = element.embedded_superdegree();
            let gmsh_perm = get_permutation_to_gmsh(cell_type, degree);

            gmsh_s.push_str(&format!(
                "2 1 {} {}\n",
                get_gmsh_cell(cell_type, degree),
                self.geometry.cell_indices[e_index].len()
            ));
            for (i, index) in self.geometry.cell_indices[e_index].iter().enumerate() {
                let points = self.geometry.cell_points(*index).unwrap();
                gmsh_s.push_str(&format!("{}", i + 1));
                for j in &gmsh_perm {
                    gmsh_s.push_str(&format!(" {}", points[*j] + 1))
                }
                gmsh_s.push('\n');
            }
        }
        gmsh_s.push_str("$EndElements\n");

        gmsh_s
    }
}
