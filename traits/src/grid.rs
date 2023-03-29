//! Geometry and topology definitions

use crate::cell::ReferenceCellType;
use bempp_tools::arrays::{AdjacencyList, Array2D};
use std::cell::Ref;

/// The ownership of a mesh entity
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Ownership {
    Owned,
    Ghost(usize, usize),
}

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

    /// The number of cells
    fn cell_count(&self) -> usize;

    /// Return the index map from the input order to the storage order
    fn index_map(&self) -> &[usize];

    ///  Compute the physical coordinates of a set of points in a given cell
    fn compute_points(
        &self,
        points: &Array2D<f64>,
        cell: usize,
        reference_points: &mut Array2D<f64>,
    );

    /// Evaluate the jacobian at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobian(&self, points: &Array2D<f64>, cell: usize, jacobians: &mut Array2D<f64>);

    /// Evaluate the determinand of the jacobian at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobian_det(&self, points: &Array2D<f64>, cell: usize, jacobian_dets: &mut [f64]);

    /// Evaluate the jacobian inverse at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobian_inverse(
        &self,
        points: &Array2D<f64>,
        cell: usize,
        jacobians: &mut Array2D<f64>,
    );
}

pub trait Topology {
    //! Grid topology
    //!
    //! Grid topology provides information about which mesh entities are connected to other mesh entities

    /// The dimension of the grid
    fn dim(&self) -> usize;

    /// Return the index map from the input order to the storage order
    fn index_map(&self) -> &[usize];

    /// The number of entities of dimension `dim`
    fn entity_count(&self, dim: usize) -> usize;

    /// The indices of the vertices that from cell with index `index`
    fn cell(&self, index: usize) -> Option<Ref<[usize]>>;

    /// The indices of the vertices that from cell with index `index`
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType>;

    /// Create the connectivity of entities of dimension `dim0` to entities of dimension `dim1`
    ///
    /// If this function is called multiple times, it will do nothing after the first call
    fn create_connectivity(&self, dim0: usize, dim1: usize);

    /// Create the connectivity information for all dimensions
    fn create_connectivity_all(&self) {
        for dim0 in 0..self.dim() {
            for dim1 in 0..self.dim() {
                self.create_connectivity(dim0, dim1);
            }
        }
    }

    /// Get the connectivity of entities of dimension `dim0` to entities of dimension `dim1`
    fn connectivity(&self, dim0: usize, dim1: usize) -> Ref<AdjacencyList<usize>>;

    /// Get the ownership of a mesh entity
    fn entity_ownership(&self, dim: usize, index: usize) -> Ownership;
}

pub trait Grid {
    //! A grid

    /// The type that implements [Topology]
    type Topology: Topology;

    /// The type that implements [Geometry]
    type Geometry: Geometry;

    /// Get the grid topology (See [Topology])
    fn topology(&self) -> &Self::Topology;

    /// Get the grid geometry (See [Geometry])
    fn geometry(&self) -> &Self::Geometry;
}
