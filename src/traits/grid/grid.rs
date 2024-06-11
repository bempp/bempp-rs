//! Definition of a grid

use super::{Cell, Edge, Point, ReferenceMap};
use crate::traits::types::{CellIterator, CellLocalIndexPair, PointIterator, ReferenceCellType};
use rlst::RlstScalar;

pub trait Grid: std::marker::Sized {
    //! A grid

    /// The floating point type used for coordinates
    type T: RlstScalar;
    /// The type used for a point
    type Point<'a>: Point
    where
        Self: 'a;
    /// The type used for a vertex
    ///
    /// Vertices are the points that are at the vertex of a cell in the grid
    type Vertex<'a>: Point
    where
        Self: 'a;
    /// The type used for an edge
    ///
    /// Vertices are the points that are at the vertex of a cell in the grid
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

    /// The number of points in the grid
    ///
    /// The points are all points used to define the cell. For curved cells, this includes points on the edges and interior
    fn number_of_points(&self) -> usize;

    /// The number of edges in the grid
    fn number_of_edges(&self) -> usize;

    /// The number of cells in the grid
    fn number_of_cells(&self) -> usize;

    /// Get the index of a point from its id
    fn point_index_from_id(&self, id: usize) -> usize;

    /// Get the id of a point from its index
    fn point_id_from_index(&self, index: usize) -> usize;

    /// Get the index of a vertex from its id
    fn vertex_index_from_id(&self, id: usize) -> usize;

    /// Get the id of a vertex from its index
    fn vertex_id_from_index(&self, index: usize) -> usize;

    /// Get the index of a cell from its id
    fn cell_index_from_id(&self, id: usize) -> usize;

    /// Get the id of a cell from its index
    fn cell_id_from_index(&self, index: usize) -> usize;

    /// Get a point from its index
    fn point_from_index(&self, index: usize) -> Self::Point<'_>;

    /// Get a vertex from its index
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
    fn iter_all_points(&self) -> PointIterator<'_, Self, std::ops::Range<usize>> {
        self.iter_points(0..self.number_of_points())
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
