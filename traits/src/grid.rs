//! Geometry and topology definitions

use crate::arrays::{AdjacencyListAccess, Array2DAccess};
use crate::cell::ReferenceCellType;
use rlst_common::traits::{RandomAccessMut, Shape};

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

    /// Return the index map from the input cell numbers to the storage numbers
    fn index_map(&self) -> &[usize];

    /// Compute the physical coordinates of a set of points in a given cell
    fn compute_points<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        physical_points: &mut impl Array2DAccess<'a, f64>,
    );

    /// Compute the physical coordinates of a set of points in a given cell,
    /// with return in transposed arrangement
    fn compute_points_transpose<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        physical_points: &mut impl Array2DAccess<'a, f64>,
    );

    /// Compute the physical coordinates of a set of points in a given cell,
    /// with return in transposed arrangement
    fn compute_points_transpose_rlst<'a, T: RandomAccessMut<Item = f64> + Shape>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        physical_points: &mut T,
    );

    /// Compute the physical coordinates of a set of points in a given cell
    fn compute_points_rlst<'a, T: RandomAccessMut<Item = f64> + Shape>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        physical_points: &mut T,
    );

    /// Compute the normals to a set of points in a given cell
    fn compute_normals<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        normals: &mut impl Array2DAccess<'a, f64>,
    );

    /// Evaluate the jacobian at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobians<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        jacobians: &mut impl Array2DAccess<'a, f64>,
    );

    /// Evaluate the determinand of the jacobian at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobian_determinants<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        jacobian_determinants: &mut [f64],
    );

    /// Evaluate the jacobian inverse at a set of points in a given cell
    ///
    /// The input points should be given using coordinates on the reference element
    fn compute_jacobian_inverses<'a>(
        &self,
        points: &impl Array2DAccess<'a, f64>,
        cell: usize,
        jacobian_inverses: &mut impl Array2DAccess<'a, f64>,
    );
}

pub trait Topology<'a> {
    //! Grid topology
    //!
    //! Grid topology provides information about which mesh entities are connected to other mesh entities

    type Connectivity: AdjacencyListAccess<'a, usize>;

    /// The dimension of the grid
    fn dim(&self) -> usize;

    /// Return the index map from the input cell numbers to the storage numbers
    fn index_map(&self) -> &[usize];

    /// The number of entities of dimension `dim`
    fn entity_count(&self, dim: usize) -> usize;

    /// The indices of the vertices that from cell with index `index`
    fn cell(&self, index: usize) -> Option<&[usize]>;

    /// The indices of the vertices that from cell with index `index`
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType>;

    /// Get the connectivity of entities of dimension `dim0` to entities of dimension `dim1`
    fn connectivity(&self, dim0: usize, dim1: usize) -> &Self::Connectivity;

    /// Get the ownership of a mesh entity
    fn entity_ownership(&self, dim: usize, index: usize) -> Ownership;
}

pub trait Grid<'a> {
    //! A grid

    /// The type that implements [Topology]
    type Topology: Topology<'a>;

    /// The type that implements [Geometry]
    type Geometry: Geometry;

    /// Get the grid topology (See [Topology])
    fn topology(&self) -> &Self::Topology;

    /// Get the grid geometry (See [Geometry])
    fn geometry(&self) -> &Self::Geometry;

    // Check if the function space is stored in serial
    fn is_serial(&self) -> bool;
}
