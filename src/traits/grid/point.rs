//! Definition of a Point

use crate::traits::types::Ownership;

use super::Grid;

pub trait Point {
    //! A point

    type G: Grid;

    /// Get the coordinates of the point.
    fn coords(&self) -> [<Self::G as Grid>::T; 3];

    /// Get the point's index.
    fn local_index(&self) -> usize;

    /// Get the global index of the point.
    fn global_index(&self) -> usize;

    /// Get the ownership of the point.

    fn ownership(&self) -> Ownership;
}
