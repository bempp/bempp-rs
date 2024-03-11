//! Topology of a cell

use super::GridType;
use crate::types::ReferenceCellType;

pub trait TopologyType {
    //! Cell topology

    /// The type of the grid that the cell is part of
    type Grid: GridType;
    /// The type used to index cells
    type IndexType: std::fmt::Debug + Eq + Copy;
    /// The type of the iterator over vertices
    type VertexIndexIter<'a>: std::iter::Iterator<Item = Self::IndexType>
    where
        Self: 'a;
    /// The type of the iterator over edges
    type EdgeIndexIter<'a>: std::iter::Iterator<Item = Self::IndexType>
    where
        Self: 'a;
    /// The type of the iterator over faces
    type FaceIndexIter<'a>: std::iter::Iterator<Item = Self::IndexType>
    where
        Self: 'a;

    /// Get an iterator over the vertices of the cell
    fn vertex_indices(&self) -> Self::VertexIndexIter<'_>;

    /// Get an iterator over the edges of the cell
    fn edge_indices(&self) -> Self::EdgeIndexIter<'_>;

    /// Get an iterator over the faces of the cell
    fn face_indices(&self) -> Self::FaceIndexIter<'_>;

    /// The cell type
    fn cell_type(&self) -> ReferenceCellType;

    /// Get the flat index from the index of an entity
    fn flat_index(&self, index: Self::IndexType) -> usize;
}
