//! Cell types

/// The type of a reference cell
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ReferenceCell {
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

impl ReferenceCell {
    /// Create a reference cell type from a u8
    pub fn from(i: u8) -> Option<ReferenceCell> {
        match i {
            0 => Some(ReferenceCell::Point),
            1 => Some(ReferenceCell::Interval),
            2 => Some(ReferenceCell::Triangle),
            3 => Some(ReferenceCell::Quadrilateral),
            4 => Some(ReferenceCell::Tetrahedron),
            5 => Some(ReferenceCell::Hexahedron),
            6 => Some(ReferenceCell::Prism),
            7 => Some(ReferenceCell::Pyramid),
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
            ReferenceCell::Point,
            ReferenceCell::from(ReferenceCell::Point as u8).unwrap()
        );
        assert_eq!(
            ReferenceCell::Interval,
            ReferenceCell::from(ReferenceCell::Interval as u8).unwrap()
        );
        assert_eq!(
            ReferenceCell::Triangle,
            ReferenceCell::from(ReferenceCell::Triangle as u8).unwrap()
        );
        assert_eq!(
            ReferenceCell::Quadrilateral,
            ReferenceCell::from(ReferenceCell::Quadrilateral as u8).unwrap()
        );
        assert_eq!(
            ReferenceCell::Tetrahedron,
            ReferenceCell::from(ReferenceCell::Tetrahedron as u8).unwrap()
        );
        assert_eq!(
            ReferenceCell::Hexahedron,
            ReferenceCell::from(ReferenceCell::Hexahedron as u8).unwrap()
        );
        assert_eq!(
            ReferenceCell::Prism,
            ReferenceCell::from(ReferenceCell::Prism as u8).unwrap()
        );
        assert_eq!(
            ReferenceCell::Pyramid,
            ReferenceCell::from(ReferenceCell::Pyramid as u8).unwrap()
        );
    }
}
