//! Definition of a grid

use super::{Cell, Edge, ReferenceMap};
use crate::traits::types::{CellIterator, CellLocalIndexPair, ReferenceCellType, VertexIterator};
use rlst::RlstScalar;

pub trait Grid: std::marker::Sized {
    //! A grid

    /// The floating point type used for coordinates
    type T: num::Float + RlstScalar<Real = Self::T>;

    /// The type used for a point
    type Vertex<'a>: super::Vertex
    where
        Self: 'a;

    /// The edge type.    
    type Edge<'a>: Edge
    where
        Self: 'a;

    /// The type used for a cell
    type Cell<'a>: Cell<Grid = Self>
    where
        Self: 'a;

    /// The type of a reference map
    type ReferenceMap<'a>: ReferenceMap<Grid = Self>
    where
        Self: 'a;

    /// The number of vertices in the grid
    ///
    /// The vertices are the points at the corners of the cell
    fn number_of_vertices(&self) -> usize;

    fn number_of_corner_vertices(&self) -> usize;

    /// Get coordinates of a point.
    fn coordinates_from_vertex_index(&self, index: usize) -> [Self::T; 3];

    /// The number of edges in the grid
    fn number_of_edges(&self) -> usize;

    /// The number of cells in the grid
    fn number_of_cells(&self) -> usize;

    /// Get the index of a vertex from its id
    fn vertex_index_from_id(&self, id: usize) -> usize;

    /// Get the id of a vertex from its index
    fn vertex_id_from_index(&self, index: usize) -> usize;

    /// Get the index of a cell from its id
    fn cell_index_from_id(&self, id: usize) -> usize;

    /// Get the id of a cell from its index
    fn cell_id_from_index(&self, index: usize) -> usize;

    /// Get a point from its index
    fn vertex_from_index(&self, index: usize) -> Self::Vertex<'_>;

    /// Get a vertex from its index
    fn edge_from_index(&self, index: usize) -> Self::Edge<'_>;

    /// Get a cell from its index
    fn cell_from_index(&self, index: usize) -> Self::Cell<'_>;

    /// Get an iterator for a subset of points in the grid
    fn iter_vertices<Iter: std::iter::Iterator<Item = usize>>(
        &self,
        index_iter: Iter,
    ) -> VertexIterator<'_, Self, Iter> {
        VertexIterator::new(index_iter, self)
    }

    /// Get an iterator for all points in the grid
    fn iter_all_vertices(&self) -> VertexIterator<'_, Self, std::ops::Range<usize>> {
        self.iter_vertices(0..self.number_of_vertices())
    }

    /// Get an iterator for a subset of cells in the grid
    fn iter_cells<Iter: std::iter::Iterator<Item = usize>>(
        &self,
        index_iter: Iter,
    ) -> CellIterator<'_, Self, Iter> {
        CellIterator::new(index_iter, self)
    }

    /// Get an iterator for all cells in the grid
    fn iter_all_cells(&self) -> CellIterator<'_, Self, std::ops::Range<usize>> {
        self.iter_cells(0..self.number_of_cells())
    }

    /// Get the reference to physical map for a set of reference points
    fn reference_to_physical_map<'a>(
        &'a self,
        reference_points: &'a [<Self::T as RlstScalar>::Real],
    ) -> Self::ReferenceMap<'a>;

    /// Get the cells that are attached to a vertex
    fn vertex_to_cells(&self, vertex_index: usize) -> &[CellLocalIndexPair<usize>];

    /// Get the cells that are attached to an edge
    fn edge_to_cells(&self, edge_index: usize) -> &[CellLocalIndexPair<usize>];

    /// Get the cells that are attached to a face
    fn face_to_cells(&self, face_index: usize) -> &[CellLocalIndexPair<usize>];

    /// Check if the function space is stored in serial
    fn is_serial(&self) -> bool;

    /// The (topological) dimension of the reference cell
    fn domain_dimension(&self) -> usize;

    /// The (geometric) dimension of cells in the physical grid
    fn physical_dimension(&self) -> usize;

    /// Get the cell types included in this grid
    fn cell_types(&self) -> &[ReferenceCellType];
}
