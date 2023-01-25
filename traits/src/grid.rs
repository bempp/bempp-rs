//! Geometry and topology definitions

use crate::cell::ReferenceCellType;
use solvers_tools::arrays::AdjacencyList;
use std::ops::Range;

pub trait Geometry {
    //! Grid geometry
    //!
    //! Grid geometry provides information about the physical locations of mesh points in space

    /// The geometric dimension
    fn dim(&self) -> usize;

    /// Get the point with the index `i`
    fn point(&self, i: usize) -> Option<&[f64]>;

    /// The number of points stored in the geometry
    fn point_count(&self) -> usize;

    /// Get the vertex numbers of a cell
    fn cell_vertices(&self, index: usize) -> Option<&[usize]>;
}

pub trait Topology {
    //! Grid topology
    //!
    //! Grid topology provides information about which mesh entities are connected to other mesh entities

    /// The dimension of the grid
    fn dim(&self) -> usize;

    /// Return the index map from the input order to the storage order
    fn index_map(&self) -> &[usize];

    /// Get the indices of cells with the given cell type
    fn get_cells(&self, cell_type: ReferenceCellType) -> Vec<usize>;

    /// Get the indices of cells with the given cell type as a range (if they are contiguous
    fn get_cells_range(&self, cell_type: ReferenceCellType) -> Option<Range<usize>>;

    /// Convert local id of a cell to global id of the cell
    fn local2global(&self, local_id: usize) -> usize;

    /// Convert global id of a cell to local id of the cell
    fn global2local(&self, global_id: usize) -> Option<usize>;

    /// The number of entities of dimension `dim`
    fn entity_count(&self, dim: usize) -> usize;

    /// The indices of the vertices that from cell with index `index`
    fn cell(&self, index: usize) -> Option<&[usize]>;

    /// Create the connectivity of entities of dimension `dim0` to entities of dimension `dim1`
    ///
    /// If this function is called multiple times, it will do nothing after the first call
    fn create_connectivity(&mut self, dim0: usize, dim1: usize);

    /// Create the connectivity information for all dimensions
    fn create_connectivity_all(&mut self) {
        for dim0 in 0..self.dim() {
            for dim1 in 0..self.dim() {
                self.create_connectivity(dim0, dim1);
            }
        }
    }

    /// Get the connectivity of entities of dimension `dim0` to entities of dimension `dim1`
    fn connectivity(&self, dim0: usize, dim1: usize) -> &AdjacencyList<usize>;
}

pub trait Grid {
    //! A grid

    /// The type that implements [Topology]
    type Topology: Topology;

    /// The type that implements [Geometry]
    type Geometry: Geometry;

    /// Get the grid topology (See [Topology])
    fn topology(&self) -> &Self::Topology;

    /// Get a mutable version of the grid topology (See [Topology])
    fn topology_mut(&mut self) -> &mut Self::Topology;

    /// Get the grid geometry (See [Geometry])
    fn geometry(&self) -> &Self::Geometry;
}
