//! One-dimensional reference cells

use crate::cell::*;

/// The reference interval
pub struct Interval;

impl ReferenceCell for Interval {
    fn dim(&self) -> usize {
        1
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Interval
    }

    fn label(&self) -> &'static str {
        "interval"
    }

    fn vertices(&self) -> &[f64] {
        static VERTICES: [f64; 2] = [0.0, 1.0];
        &VERTICES
    }

    fn edges(&self) -> &[usize] {
        static EDGES: [usize; 2] = [0, 1];
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

    fn vertex_count(&self) -> usize {
        2
    }
    fn edge_count(&self) -> usize {
        1
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
                assert!(entity_number < 2);
                match connected_dim {
                    0 => Ok(vec![entity_number]),
                    1 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            1 => {
                assert!(entity_number == 0);
                match connected_dim {
                    0 => Ok(vec![0, 1]),
                    1 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            _ => Err(()),
        }
    }
}
