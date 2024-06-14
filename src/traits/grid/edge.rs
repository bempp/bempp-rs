//! Definition of an edge.

use super::Grid;

pub trait Edge {
    //! An edge

    type Grid: Grid;

    type Iter<'a>: std::iter::Iterator<Item = <Self::Grid as Grid>::Vertex<'a>>
    where
        Self: 'a;

    /// The index of the edge.
    fn index(&self) -> usize;

    /// The points associated with the edge.
    fn vertices(&self) -> Self::Iter<'_>;
}
