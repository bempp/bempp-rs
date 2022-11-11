//! Three-dimensional reference cells

use crate::cell::*;

/// The reference tetrahedron
pub struct Tetrahedron;

/// The reference tetrahedron
pub struct Hexahedron;

/// The reference prism
pub struct Prism;

/// The reference pyramid
pub struct Pyramid;

impl ReferenceCell for Tetrahedron {
    fn dim(&self) -> usize {
        3
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Tetrahedron
    }

    fn label(&self) -> &'static str {
        "tetrahedron"
    }

    fn vertices(&self) -> &[f64] {
        static VERTICES: [f64; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        &VERTICES
    }

    fn edges(&self) -> &[usize] {
        static EDGES: [usize; 12] = [2, 3, 1, 3, 1, 2, 0, 3, 0, 2, 0, 1];
        &EDGES
    }

    fn faces(&self) -> &[usize] {
        static FACES: [usize; 12] = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2];
        &FACES
    }
    fn faces_nvertices(&self) -> &[usize] {
        static FACES_NV: [usize; 4] = [3, 3, 3, 3];
        &FACES_NV
    }

    fn vertex_count(&self) -> usize {
        4
    }
    fn edge_count(&self) -> usize {
        6
    }
    fn face_count(&self) -> usize {
        4
    }
    fn volume_count(&self) -> usize {
        1
    }
    fn connectivity(
        &self,
        entity_dim: usize,
        entity_number: usize,
        connected_dim: usize,
    ) -> Result<Vec<usize>, ()> {
        match entity_dim {
            0 => {
                assert!(entity_number < 4);
                match connected_dim {
                    0 => Ok(vec![entity_number]),
                    1 => match entity_number {
                        0 => Ok(vec![3, 4, 5]),
                        1 => Ok(vec![1, 2, 5]),
                        2 => Ok(vec![0, 2, 4]),
                        3 => Ok(vec![0, 1, 3]),
                        _ => Err(()),
                    },
                    2 => match entity_number {
                        0 => Ok(vec![1, 2, 3]),
                        1 => Ok(vec![0, 2, 3]),
                        2 => Ok(vec![0, 1, 3]),
                        3 => Ok(vec![0, 1, 2]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            1 => {
                assert!(entity_number < 6);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![2, 3]),
                        1 => Ok(vec![1, 3]),
                        2 => Ok(vec![1, 2]),
                        3 => Ok(vec![0, 3]),
                        4 => Ok(vec![0, 2]),
                        5 => Ok(vec![0, 1]),
                        _ => Err(()),
                    },
                    1 => Ok(vec![entity_number]),
                    2 => match entity_number {
                        0 => Ok(vec![0, 1]),
                        1 => Ok(vec![0, 2]),
                        2 => Ok(vec![0, 3]),
                        3 => Ok(vec![1, 2]),
                        4 => Ok(vec![1, 3]),
                        5 => Ok(vec![2, 3]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            2 => {
                assert!(entity_number < 4);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![1, 2, 3]),
                        1 => Ok(vec![0, 2, 3]),
                        2 => Ok(vec![0, 1, 3]),
                        3 => Ok(vec![0, 1, 2]),
                        _ => Err(()),
                    },
                    1 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 3, 4]),
                        2 => Ok(vec![1, 3, 5]),
                        3 => Ok(vec![2, 4, 5]),
                        _ => Err(()),
                    },
                    2 => Ok(vec![entity_number]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            3 => {
                assert!(entity_number == 0);
                match connected_dim {
                    0 => Ok(vec![0, 1, 2, 3]),
                    1 => Ok(vec![0, 1, 2, 3, 4, 5]),
                    2 => Ok(vec![0, 1, 2, 3]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            _ => Err(()),
        }
    }
}

impl ReferenceCell for Hexahedron {
    fn dim(&self) -> usize {
        3
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Hexahedron
    }

    fn label(&self) -> &'static str {
        "hexahedron"
    }

    fn vertices(&self) -> &[f64] {
        static VERTICES: [f64; 24] = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        &VERTICES
    }

    fn edges(&self) -> &[usize] {
        static EDGES: [usize; 24] = [
            0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 3, 7, 4, 5, 4, 6, 5, 7, 6, 7,
        ];
        &EDGES
    }

    fn faces(&self) -> &[usize] {
        static FACES: [usize; 24] = [
            0, 1, 2, 3, 0, 1, 4, 5, 0, 2, 4, 6, 1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7,
        ];
        &FACES
    }
    fn faces_nvertices(&self) -> &[usize] {
        static FACES_NV: [usize; 6] = [4, 4, 4, 4, 4, 4];
        &FACES_NV
    }

    fn vertex_count(&self) -> usize {
        8
    }
    fn edge_count(&self) -> usize {
        12
    }
    fn face_count(&self) -> usize {
        6
    }
    fn volume_count(&self) -> usize {
        1
    }
    fn connectivity(
        &self,
        entity_dim: usize,
        entity_number: usize,
        connected_dim: usize,
    ) -> Result<Vec<usize>, ()> {
        match entity_dim {
            0 => {
                assert!(entity_number < 8);
                match connected_dim {
                    0 => Ok(vec![entity_number]),
                    1 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 3, 4]),
                        2 => Ok(vec![1, 5, 6]),
                        3 => Ok(vec![3, 5, 7]),
                        4 => Ok(vec![2, 8, 9]),
                        5 => Ok(vec![4, 8, 10]),
                        6 => Ok(vec![6, 9, 11]),
                        7 => Ok(vec![7, 10, 11]),
                        _ => Err(()),
                    },
                    2 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 1, 3]),
                        2 => Ok(vec![0, 2, 4]),
                        3 => Ok(vec![0, 3, 4]),
                        4 => Ok(vec![1, 2, 5]),
                        5 => Ok(vec![1, 3, 5]),
                        6 => Ok(vec![2, 4, 5]),
                        7 => Ok(vec![3, 4, 5]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            1 => {
                assert!(entity_number < 12);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![0, 1]),
                        1 => Ok(vec![0, 2]),
                        2 => Ok(vec![0, 4]),
                        3 => Ok(vec![1, 3]),
                        4 => Ok(vec![1, 5]),
                        5 => Ok(vec![2, 3]),
                        6 => Ok(vec![2, 6]),
                        7 => Ok(vec![3, 7]),
                        8 => Ok(vec![4, 5]),
                        9 => Ok(vec![4, 6]),
                        10 => Ok(vec![5, 7]),
                        11 => Ok(vec![6, 7]),
                        _ => Err(()),
                    },
                    1 => Ok(vec![entity_number]),
                    2 => match entity_number {
                        0 => Ok(vec![0, 1]),
                        1 => Ok(vec![0, 2]),
                        2 => Ok(vec![1, 2]),
                        3 => Ok(vec![0, 3]),
                        4 => Ok(vec![1, 3]),
                        5 => Ok(vec![0, 4]),
                        6 => Ok(vec![2, 4]),
                        7 => Ok(vec![3, 4]),
                        8 => Ok(vec![1, 5]),
                        9 => Ok(vec![2, 5]),
                        10 => Ok(vec![3, 5]),
                        11 => Ok(vec![4, 5]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            2 => {
                assert!(entity_number < 6);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![0, 1, 2, 3]),
                        1 => Ok(vec![0, 1, 4, 5]),
                        2 => Ok(vec![0, 2, 4, 6]),
                        3 => Ok(vec![1, 3, 5, 7]),
                        4 => Ok(vec![2, 3, 6, 7]),
                        5 => Ok(vec![4, 5, 6, 7]),
                        _ => Err(()),
                    },
                    1 => match entity_number {
                        0 => Ok(vec![0, 1, 3, 5]),
                        1 => Ok(vec![0, 2, 4, 8]),
                        2 => Ok(vec![1, 2, 6, 9]),
                        3 => Ok(vec![3, 4, 7, 10]),
                        4 => Ok(vec![5, 6, 7, 11]),
                        5 => Ok(vec![8, 9, 10, 11]),
                        _ => Err(()),
                    },
                    2 => Ok(vec![entity_number]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            3 => {
                assert!(entity_number == 0);
                match connected_dim {
                    0 => Ok(vec![0, 1, 2, 3, 4, 5, 6, 7]),
                    1 => Ok(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                    2 => Ok(vec![0, 1, 2, 3, 4, 5]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            _ => Err(()),
        }
    }
}

impl ReferenceCell for Prism {
    fn dim(&self) -> usize {
        3
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Prism
    }

    fn label(&self) -> &'static str {
        "prism"
    }

    fn vertices(&self) -> &[f64] {
        static VERTICES: [f64; 18] = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0,
        ];
        &VERTICES
    }

    fn edges(&self) -> &[usize] {
        static EDGES: [usize; 18] = [0, 1, 0, 2, 0, 3, 1, 2, 1, 4, 2, 5, 3, 4, 3, 5, 4, 5];
        &EDGES
    }
    fn faces(&self) -> &[usize] {
        static FACES: [usize; 18] = [0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 5, 1, 2, 4, 5, 3, 4, 5];
        &FACES
    }
    fn faces_nvertices(&self) -> &[usize] {
        static FACES_NV: [usize; 5] = [3, 4, 4, 4, 3];
        &FACES_NV
    }
    fn vertex_count(&self) -> usize {
        6
    }
    fn edge_count(&self) -> usize {
        9
    }
    fn face_count(&self) -> usize {
        5
    }
    fn volume_count(&self) -> usize {
        1
    }
    fn connectivity(
        &self,
        entity_dim: usize,
        entity_number: usize,
        connected_dim: usize,
    ) -> Result<Vec<usize>, ()> {
        match entity_dim {
            0 => {
                assert!(entity_number < 6);
                match connected_dim {
                    0 => Ok(vec![entity_number]),
                    1 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 3, 4]),
                        2 => Ok(vec![1, 3, 5]),
                        3 => Ok(vec![2, 6, 7]),
                        4 => Ok(vec![4, 6, 8]),
                        5 => Ok(vec![5, 7, 8]),
                        _ => Err(()),
                    },
                    2 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 1, 3]),
                        2 => Ok(vec![0, 2, 3]),
                        3 => Ok(vec![1, 2, 4]),
                        4 => Ok(vec![1, 3, 4]),
                        5 => Ok(vec![2, 3, 4]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            1 => {
                assert!(entity_number < 9);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![0, 1]),
                        1 => Ok(vec![0, 2]),
                        2 => Ok(vec![0, 3]),
                        3 => Ok(vec![1, 2]),
                        4 => Ok(vec![1, 4]),
                        5 => Ok(vec![2, 5]),
                        6 => Ok(vec![3, 4]),
                        7 => Ok(vec![3, 5]),
                        8 => Ok(vec![4, 5]),
                        _ => Err(()),
                    },
                    1 => Ok(vec![entity_number]),
                    2 => match entity_number {
                        0 => Ok(vec![0, 1]),
                        1 => Ok(vec![0, 2]),
                        2 => Ok(vec![1, 2]),
                        3 => Ok(vec![0, 3]),
                        4 => Ok(vec![1, 3]),
                        5 => Ok(vec![2, 3]),
                        6 => Ok(vec![1, 4]),
                        7 => Ok(vec![2, 4]),
                        8 => Ok(vec![3, 4]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            2 => {
                assert!(entity_number < 5);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 1, 3, 4]),
                        2 => Ok(vec![0, 2, 3, 5]),
                        3 => Ok(vec![1, 2, 4, 5]),
                        4 => Ok(vec![3, 4, 5]),
                        _ => Err(()),
                    },
                    1 => match entity_number {
                        0 => Ok(vec![0, 1, 3]),
                        1 => Ok(vec![0, 2, 4, 6]),
                        2 => Ok(vec![1, 2, 5, 7]),
                        3 => Ok(vec![3, 4, 5, 8]),
                        4 => Ok(vec![6, 7, 8]),
                        _ => Err(()),
                    },
                    2 => Ok(vec![entity_number]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            3 => {
                assert!(entity_number == 0);
                match connected_dim {
                    0 => Ok(vec![0, 1, 2, 3, 4, 5]),
                    1 => Ok(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]),
                    2 => Ok(vec![0, 1, 2, 3, 4]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            _ => Err(()),
        }
    }
}

impl ReferenceCell for Pyramid {
    fn dim(&self) -> usize {
        3
    }

    fn cell_type(&self) -> ReferenceCellType {
        ReferenceCellType::Pyramid
    }

    fn label(&self) -> &'static str {
        "pyramid"
    }

    fn vertices(&self) -> &[f64] {
        static VERTICES: [f64; 15] = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        &VERTICES
    }

    fn edges(&self) -> &[usize] {
        static EDGES: [usize; 16] = [0, 1, 0, 2, 0, 4, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4];
        &EDGES
    }
    fn faces(&self) -> &[usize] {
        static FACES: [usize; 16] = [0, 1, 2, 3, 0, 1, 4, 0, 2, 4, 1, 3, 4, 2, 3, 4];
        &FACES
    }
    fn faces_nvertices(&self) -> &[usize] {
        static FACES_NV: [usize; 5] = [4, 3, 3, 3, 3];
        &FACES_NV
    }
    fn vertex_count(&self) -> usize {
        5
    }
    fn edge_count(&self) -> usize {
        8
    }
    fn face_count(&self) -> usize {
        5
    }
    fn volume_count(&self) -> usize {
        1
    }
    fn connectivity(
        &self,
        entity_dim: usize,
        entity_number: usize,
        connected_dim: usize,
    ) -> Result<Vec<usize>, ()> {
        match entity_dim {
            0 => {
                assert!(entity_number < 5);
                match connected_dim {
                    0 => Ok(vec![entity_number]),
                    1 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 3, 4]),
                        2 => Ok(vec![1, 5, 6]),
                        3 => Ok(vec![3, 5, 7]),
                        4 => Ok(vec![2, 4, 6, 7]),
                        _ => Err(()),
                    },
                    2 => match entity_number {
                        0 => Ok(vec![0, 1, 2]),
                        1 => Ok(vec![0, 1, 3]),
                        2 => Ok(vec![0, 2, 4]),
                        3 => Ok(vec![0, 3, 4]),
                        4 => Ok(vec![1, 2, 3, 4]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            1 => {
                assert!(entity_number < 8);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![0, 1]),
                        1 => Ok(vec![0, 2]),
                        2 => Ok(vec![0, 4]),
                        3 => Ok(vec![1, 3]),
                        4 => Ok(vec![1, 4]),
                        5 => Ok(vec![2, 3]),
                        6 => Ok(vec![2, 4]),
                        7 => Ok(vec![3, 4]),
                        _ => Err(()),
                    },
                    1 => Ok(vec![entity_number]),
                    2 => match entity_number {
                        0 => Ok(vec![0, 1]),
                        1 => Ok(vec![0, 2]),
                        2 => Ok(vec![1, 2]),
                        3 => Ok(vec![0, 3]),
                        4 => Ok(vec![1, 3]),
                        5 => Ok(vec![0, 4]),
                        6 => Ok(vec![2, 4]),
                        7 => Ok(vec![3, 4]),
                        _ => Err(()),
                    },
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            2 => {
                assert!(entity_number < 5);
                match connected_dim {
                    0 => match entity_number {
                        0 => Ok(vec![0, 1, 2, 3]),
                        1 => Ok(vec![0, 1, 4]),
                        2 => Ok(vec![0, 2, 4]),
                        3 => Ok(vec![1, 3, 4]),
                        4 => Ok(vec![2, 3, 4]),
                        _ => Err(()),
                    },
                    1 => match entity_number {
                        0 => Ok(vec![0, 1, 3, 5]),
                        1 => Ok(vec![0, 2, 4]),
                        2 => Ok(vec![1, 2, 6]),
                        3 => Ok(vec![3, 4, 7]),
                        4 => Ok(vec![5, 6, 7]),
                        _ => Err(()),
                    },
                    2 => Ok(vec![entity_number]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            3 => {
                assert!(entity_number == 0);
                match connected_dim {
                    0 => Ok(vec![0, 1, 2, 3, 4]),
                    1 => Ok(vec![0, 1, 2, 3, 4, 5, 6, 7]),
                    2 => Ok(vec![0, 1, 2, 3, 4]),
                    3 => Ok(vec![0]),
                    _ => Err(()),
                }
            }
            _ => Err(()),
        }
    }
}
