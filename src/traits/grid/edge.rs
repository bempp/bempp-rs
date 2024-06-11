//! Definition of an edge.

use crate::traits::types::Ownership;

pub trait Edge {
    //! An edge

    /// The index of the edge
    fn index(&self) -> usize;

    /// Get the point's ownership
    fn ownership(&self) -> Ownership;
}
