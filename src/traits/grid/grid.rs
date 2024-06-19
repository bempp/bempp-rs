//! Definition of a grid

use super::{Cell, Edge, ReferenceMap};
use crate::{
    traits::types::{CellIterator, CellLocalIndexPair, Ownership, PointIterator, ReferenceCell},
    types::RealScalar,
};
use rlst::RlstScalar;

pub trait Grid: std::marker::Sized {
    //! A grid

    /// The floating point type used for coordinates
    type T: RealScalar;

    /// The type used for a point
    type Point<'a>: super::Point
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

    /// The number of global points in the grid.
    fn number_of_global_points(&self) -> usize;

    /// The number of global points in the grid.
    fn number_of_local_points(&self) -> usize;

    /// The number of global vertices (corner points) in the grid.
    fn number_of_global_corner_vertices(&self) -> usize;

    /// The number of local vertices (corner points) in the grid.
    fn number_of_local_corner_vertices(&self) -> usize;

    /// Get coordinates of a point.
    fn coordinates_from_point_index(&self, index: usize) -> [Self::T; 3];

    /// Get global index from a local point index.
    fn global_point_index(&self, local_index: usize) -> usize;

    /// Get global index from a local edge index.
    fn global_edge_index(&self, local_index: usize) -> usize;

    /// Get global index from a local cell index.
    fn global_cell_index(&self, local_index: usize) -> usize;

    /// The number of global edges in the grid
    fn number_of_global_edges(&self) -> usize;

    /// The number of global cells in the grid
    fn number_of_global_cells(&self) -> usize;

    /// The number of local edges in the grid
    fn number_of_local_edges(&self) -> usize;

    /// The number of local cells in the grid
    fn number_of_local_cells(&self) -> usize;

    /// Get the index of a point from its id
    fn point_index_from_id(&self, id: usize) -> usize;

    /// Get the id of a point from its index
    fn point_id_from_index(&self, index: usize) -> usize;

    /// Get the index of a cell from its id
    fn cell_index_from_id(&self, id: usize) -> usize;

    /// Get the id of a cell from its index
    fn cell_id_from_index(&self, index: usize) -> usize;

    /// Get a point from its index
    fn point_from_index(&self, index: usize) -> Self::Point<'_>;

    /// Get an edge from its index
    fn edge_from_index(&self, index: usize) -> Self::Edge<'_>;

    /// Get a cell from its index
    fn cell_from_index(&self, index: usize) -> Self::Cell<'_>;

    /// Get an iterator for a subset of points in the grid
    fn iter_points<Iter: std::iter::Iterator<Item = usize>>(
        &self,
        index_iter: Iter,
    ) -> PointIterator<'_, Self, Iter> {
        PointIterator::new(index_iter, self)
    }

    /// Get an iterator for all points in the grid
    fn iter_local_points(&self) -> PointIterator<'_, Self, std::ops::Range<usize>> {
        self.iter_points(0..self.number_of_local_points())
    }

    /// Get an iterator for a subset of cells in the grid
    fn iter_cells<Iter: std::iter::Iterator<Item = usize>>(
        &self,
        index_iter: Iter,
    ) -> CellIterator<'_, Self, Iter> {
        CellIterator::new(index_iter, self)
    }

    /// Get an iterator for all cells in the grid
    fn iter_local_cells(&self) -> CellIterator<'_, Self, std::ops::Range<usize>> {
        self.iter_cells(0..self.number_of_local_cells())
    }

    /// Get the reference to physical map for a set of reference points
    fn reference_to_physical_map<'a>(
        &'a self,
        reference_points: &'a [<Self::T as RlstScalar>::Real],
    ) -> Self::ReferenceMap<'a>;

    /// Get the cells that are attached to a vertex
    fn points_to_cells(&self, vertex_index: usize) -> &[CellLocalIndexPair<usize>];

    /// Get the points associated with an edge
    fn edge_to_points(&self, edge_index: usize) -> &[usize];

    /// Get the end points associated with an edge
    fn edge_to_end_points(&self, edge_index: usize) -> &[usize];

    /// Get the points of a cell
    fn cell_to_points(&self, cell_index: usize) -> &[usize];

    /// Get the points of a cell
    fn cell_to_corner_points(&self, cell_index: usize) -> &[usize];

    /// Get the edges of a cell
    fn cell_to_edges(&self, cell_index: usize) -> &[usize];

    /// Get the cells that are attached to an edge
    fn edge_to_cell(&self, edge_index: usize) -> &[CellLocalIndexPair<usize>];

    /// Get the cells that are attached to a face
    fn face_to_cell(&self, face_index: usize) -> &[CellLocalIndexPair<usize>];

    /// Check if the function space is stored in serial
    fn is_serial(&self) -> bool;

    /// The (topological) dimension of the reference cell
    fn domain_dimension(&self) -> usize;

    /// The (geometric) dimension of cells in the physical grid
    fn physical_dimension(&self) -> usize;

    /// Get the cell types included in this grid
    fn cell_types(&self) -> &[ReferenceCell];

    /// Get the cell type of a cell
    fn cell_type(&self, cell_index: usize) -> ReferenceCell;

    /// Get the ownership of a point from its local index
    fn point_ownership(&self, local_index: usize) -> Ownership;

    /// Get the ownership of an edge  from its local index
    fn edge_ownership(&self, local_index: usize) -> Ownership;

    /// Get the ownership of a cell  from its local index
    fn cell_ownership(&self, local_index: usize) -> Ownership;
}
