//! Cell types

/// The type of a reference cell
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ReferenceCellType {
    Point = 0,
    Interval = 1,
    Triangle = 2,
    Quadrilateral = 3,
    Tetrahedron = 4,
    Hexahedron = 5,
    Prism = 6,
    Pyramid = 7,
}

/// A (cell, local index) pair
///
/// The local index is the index of a subentity (eg vertex, edge) within the cell as it is numbered in the reference cell
#[derive(Debug, Clone)]
pub struct CellLocalIndexPair<IndexType: std::fmt::Debug + Eq + Copy> {
    pub cell: IndexType,
    pub local_index: usize,
}

impl<IndexType: std::fmt::Debug + Eq + Copy> CellLocalIndexPair<IndexType> {
    /// Create a (cell, local index) pair
    pub fn new(cell: IndexType, local_index: usize) -> Self {
        Self { cell, local_index }
    }
}
