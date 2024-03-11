//! Definition of a grid

use super::{CellType, PointType};
use crate::types::cell_iterator::CellIterator;
use crate::types::point_iterator::PointIterator;
use crate::types::CellLocalIndexPair;
use rlst_dense::types::RlstScalar;

use super::ReferenceMapType;

pub trait GridType: std::marker::Sized {
    //! A grid

    /// The floating point type used for coordinates
    type T: RlstScalar;
    /// The type used to index cells
    type IndexType: std::fmt::Debug + Eq + Copy;
    /// The type used for a point
    type Point<'a>: PointType
    where
        Self: 'a;
    /// The type used for a cell
    type Cell<'a>: CellType
    where
        Self: 'a;
    /// The type of a reference map
    type ReferenceMap<'a>: ReferenceMapType
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

    /// The number of cells in the grid
    fn number_of_cells(&self) -> usize;

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
    fn vertex_to_cells(
        &self,
        vertex_index: Self::IndexType,
    ) -> &[CellLocalIndexPair<Self::IndexType>];

    /// Get the cells that are attached to an edge
    fn edge_to_cells(&self, edge_index: Self::IndexType) -> &[CellLocalIndexPair<Self::IndexType>];

    /// Get the cells that are attached to a face
    fn face_to_cells(&self, face_index: Self::IndexType) -> &[CellLocalIndexPair<Self::IndexType>];

    /// Check if the function space is stored in serial
    fn is_serial(&self) -> bool;

    /// The (topological) dimension of the reference cell
    fn domain_dimension(&self) -> usize;

    /// The (geometric) dimension of cells in the physical grid
    fn physical_dimension(&self) -> usize;

}
