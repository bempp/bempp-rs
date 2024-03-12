//! Definition of a cell.

use super::GridType;
use crate::types::ReferenceCellType;
use rlst_dense::types::RlstScalar;
use std::hash::Hash;

pub trait CellType {
    //! A cell

    /// The type of the grid that the cell is part of
    type Grid: GridType;
    /// The type of the cell topology
    type Topology<'a>: TopologyType<IndexType = <Self::Grid as GridType>::IndexType>
    where
        Self: 'a;
    /// The type of the cell geometry
    type Geometry<'a>: GeometryType
    where
        Self: 'a;

    /// The id of the cell
    fn id(&self) -> usize;

    /// The index of the cell
    fn index(&self) -> usize;

    /// Get the cell's topology
    fn topology(&self) -> Self::Topology<'_>;

    /// Get the grid that the cell is part of
    fn grid(&self) -> &Self::Grid;

    /// Get the cell's geometry
    fn geometry(&self) -> Self::Geometry<'_>;
}

pub trait TopologyType {
    //! Cell topology

    /// The type of the grid that the cell is part of
    type Grid: GridType;
    /// The type used to index cells
    type IndexType: std::fmt::Debug + Eq + Copy + Hash;
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

pub trait GeometryType {
    //! Cell geometry

    /// The type of the grid that the cell is part of
    type Grid: GridType;
    /// Type of iterator over vertices
    type VertexIterator<'iter>: std::iter::Iterator<Item = <Self::Grid as GridType>::Point<'iter>>
    where
        Self: 'iter;
    /// Type of iterator over points
    type PointIterator<'iter>: std::iter::Iterator<Item = <Self::Grid as GridType>::Point<'iter>>
    where
        Self: 'iter;

    /// The physical/geometric dimension of the cell
    fn physical_dimension(&self) -> usize;

    /// The midpoint of the cell
    fn midpoint(&self, point: &mut [<<Self::Grid as GridType>::T as RlstScalar>::Real]);

    /// The diameter of the cell
    fn diameter(&self) -> <<Self::Grid as GridType>::T as RlstScalar>::Real;

    /// The volume of the cell
    fn volume(&self) -> <<Self::Grid as GridType>::T as RlstScalar>::Real;

    /// The vertices of the cell
    ///
    /// The vertices are the points at the corners of the cell
    fn vertices(&self) -> Self::VertexIterator<'_>;

    /// The points of the cell
    ///
    /// The points are all points used to define the cell. For curved cells, this includes points on the edges and interior
    fn points(&self) -> Self::PointIterator<'_>;
}
