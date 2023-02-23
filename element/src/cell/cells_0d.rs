//! One-dimensional reference cells

use crate::cell::*;

/// The reference interval
pub struct Point;

impl ReferenceCell for Point {
    fn dim(&self) -> usize {
        0
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Point
    }

    fn label(&self) -> &'static str {
        "point"
    }

    fn vertices(&self) -> &[f64] {
        static VERTICES: [f64; 0] = [];
        &VERTICES
    }

    fn edges(&self) -> &[usize] {
        static EDGES: [usize; 0] = [];
        &EDGES
    }

    fn faces(&self) -> &[usize] {
        static FACES: [usize; 0] = [];
        &FACES
    }
    fn faces_nvertices(&self) -> &[usize] {
        static FACES_NV: [usize; 0] = [];
        &FACES_NV
    }

    fn entity_types(&self, dim: usize) -> Result<Vec<ReferenceCellType>, ()> {
        match dim {
            0 => Ok(vec![ReferenceCellType::Point]),
            _ => Err(()),
        }
    }

    fn vertex_count(&self) -> usize {
        1
    }
    fn edge_count(&self) -> usize {
        0
    }
    fn face_count(&self) -> usize {
        0
    }
    fn volume_count(&self) -> usize {
        0
    }
    fn connectivity(
        &self,
        entity_dim: usize,
        entity_number: usize,
        connected_dim: usize,
    ) -> Result<Vec<usize>, ()> {
        match entity_dim {
            0 => {
                assert!(entity_number < 1);
                match connected_dim {
                    0 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            _ => Err(()),
        }
    }
}
