//! Cell types

/// The type of a reference cell
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ReferenceCellType {
    /// A point
    Point = 0,
    /// A line interval
    Interval = 1,
    /// A triangle
    Triangle = 2,
    /// A quadrilateral
    Quadrilateral = 3,
    /// A tetrahedron (whose faces are all triangles)
    Tetrahedron = 4,
    /// A hexahedron (whose faces are all quadrilaterals)
    Hexahedron = 5,
    /// A triangular prism
    Prism = 6,
    /// A square-based pyramid
    Pyramid = 7,
}

/// A (cell, local index) pair
///
/// The local index is the index of a subentity (eg vertex, edge) within the cell as it is numbered in the reference cell
#[derive(Debug, Clone)]
pub struct CellLocalIndexPair<IndexType: std::fmt::Debug + Eq + Copy> {
    /// The cell's index
    pub cell: IndexType,
    /// The local index of the subentity
    pub local_index: usize,
}

impl<IndexType: std::fmt::Debug + Eq + Copy> CellLocalIndexPair<IndexType> {
    /// Create a (cell, local index) pair
    pub fn new(cell: IndexType, local_index: usize) -> Self {
        Self { cell, local_index }
    }
}
