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

impl ReferenceCellType {
    /// Create a reference cell type from a u8
    pub fn from(i: u8) -> Option<ReferenceCellType> {
        match i {
            0 => Some(ReferenceCellType::Point),
            1 => Some(ReferenceCellType::Interval),
            2 => Some(ReferenceCellType::Triangle),
            3 => Some(ReferenceCellType::Quadrilateral),
            4 => Some(ReferenceCellType::Tetrahedron),
            5 => Some(ReferenceCellType::Hexahedron),
            6 => Some(ReferenceCellType::Prism),
            7 => Some(ReferenceCellType::Pyramid),
            _ => None,
        }
    }
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

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_reference_cell_type() {
        assert_eq!(
            ReferenceCellType::Point,
            ReferenceCellType::from(ReferenceCellType::Point as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Interval,
            ReferenceCellType::from(ReferenceCellType::Interval as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Triangle,
            ReferenceCellType::from(ReferenceCellType::Triangle as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Quadrilateral,
            ReferenceCellType::from(ReferenceCellType::Quadrilateral as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Tetrahedron,
            ReferenceCellType::from(ReferenceCellType::Tetrahedron as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Hexahedron,
            ReferenceCellType::from(ReferenceCellType::Hexahedron as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Prism,
            ReferenceCellType::from(ReferenceCellType::Prism as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Pyramid,
            ReferenceCellType::from(ReferenceCellType::Pyramid as u8).unwrap()
        );
    }
}
