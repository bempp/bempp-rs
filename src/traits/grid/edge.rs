//! Definition of an edge.

use super::GridType;
use crate::traits::types::{Ownership, ReferenceCellType};
use rlst::RlstScalar;

pub trait EdgeType {
    //! An edge

    /// The index of the edge
    fn index(&self) -> usize;

    /// Get the point's ownership
    fn ownership(&self) -> Ownership;
}
