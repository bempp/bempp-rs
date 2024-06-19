//! Definition of an edge.

use std::iter::Copied;

use super::Grid;
use crate::traits::types::{Ownership, PointIterator};

pub trait Edge {
    //! An edge

    type G: Grid;

    /// The index of the edge.
    fn local_index(&self) -> usize;

    /// The global index of the edge.
    fn global_index(&self) -> usize;

    /// The points associated with the edge.
    fn points(&self) -> PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>>;

    /// The end points associated with the edge.
    fn end_points(&self) -> PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>>;

    /// Get the ownership of the edge.
    fn ownership(&self) -> Ownership;
}
