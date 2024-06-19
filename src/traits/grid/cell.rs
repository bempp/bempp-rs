//! Definition of a cell.

use std::iter::Copied;

use super::Grid;
use crate::traits::types::{EdgeIterator, Ownership, PointIterator, ReferenceCell};
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

    /// The local index of the cell
    fn local_index(&self) -> usize;

    /// The global index of the cell
    fn global_index(&self) -> usize;

    /// Get the cell's topology
    fn topology(&self) -> Self::Topology<'_>;

    /// Get the grid that the cell is part of
    fn grid(&self) -> &Self::Grid;

    /// Get the cell's geometry
    fn geometry(&self) -> Self::Geometry<'_>;

    /// Get the ownership of the cell
    fn ownership(&self) -> Ownership;
}

/// Cell Topology.
pub trait Topology {
    /// The type of the grid that the cell is part of
    type G: Grid;

    /// Get an iterator over the vertices of the cell
    fn points(&self) -> PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>>;

    /// Get an iterator over corner indices.
    fn corner_points(&self) -> PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>>;

    /// Get an iterator over the edges of the cell
    fn edges(&self) -> EdgeIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>>;

    /// The cell type
    fn cell_type(&self) -> ReferenceCell;
}

pub trait Geometry {
    //! Cell geometry

    /// The type of the grid that the cell is part of
    type G: Grid;

    /// The physical/geometric dimension of the cell
    fn physical_dimension(&self) -> usize;

    /// The midpoint of the cell
    fn midpoint(&self, point: &mut [<<Self::G as Grid>::T as RlstScalar>::Real]);

    /// The diameter of the cell
    fn diameter(&self) -> <<Self::G as Grid>::T as RlstScalar>::Real;

    /// The volume of the cell
    fn volume(&self) -> <<Self::G as Grid>::T as RlstScalar>::Real;

    fn integration_element(&self) -> <Self::G as Grid>::T;
}
