//! Definition of a cell.

use super::Grid;
use crate::traits::types::{Ownership, ReferenceCellType};
use rlst::RlstScalar;

pub trait Cell {
    //! A cell

    /// The type of the grid that the cell is part of
    type Grid: Grid;
    /// The type of the cell topology
    type Topology<'a>: Topology
    where
        Self: 'a;
    /// The type of the cell geometry
    type Geometry<'a>: Geometry
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

    /// Get the cell's ownership
    fn ownership(&self) -> Ownership;
}

pub trait Topology {
    //! Cell topology

    /// The type of the grid that the cell is part of
    type Grid: Grid;
    /// The type of the iterator over vertices
    type VertexIndexIter<'a>: std::iter::Iterator<Item = usize>
    where
        Self: 'a;
    /// The type of the iterator over edges
    type EdgeIndexIter<'a>: std::iter::Iterator<Item = usize>
    where
        Self: 'a;

    /// Get an iterator over the vertices of the cell
    fn vertex_indices(&self) -> Self::VertexIndexIter<'_>;

    /// Get an iterator over the edges of the cell
    fn edge_indices(&self) -> Self::EdgeIndexIter<'_>;

    /// The cell type
    fn cell_type(&self) -> ReferenceCellType;
}

pub trait Geometry {
    //! Cell geometry

    /// The type of the grid that the cell is part of
    type Grid: Grid;
    /// Type of iterator over vertices
    type VertexIterator<'iter>: std::iter::Iterator<Item = <Self::Grid as Grid>::Vertex<'iter>>
    where
        Self: 'iter;

    /// The physical/geometric dimension of the cell
    fn physical_dimension(&self) -> usize;

    /// The midpoint of the cell
    fn midpoint(&self, point: &mut [<<Self::Grid as Grid>::T as RlstScalar>::Real]);

    /// The diameter of the cell
    fn diameter(&self) -> <<Self::Grid as Grid>::T as RlstScalar>::Real;

    /// The volume of the cell
    fn volume(&self) -> <<Self::Grid as Grid>::T as RlstScalar>::Real;

    /// The corner vertices of the cell
    ///
    fn corner_vertices(&self) -> Self::VertexIterator<'_>;

    /// The vertices of the cell
    ///
    fn vertices(&self) -> Self::VertexIterator<'_>;
}
