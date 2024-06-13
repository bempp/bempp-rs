//! Definition of an edge.

pub trait Edge {
    //! An edge

    type Iter<'a>: std::iter::Iterator<Item = usize>
    where
        Self: 'a;

    /// The index of the edge.
    fn index(&self) -> usize;

    /// The points associated with the edge.
    fn vertices(&self) -> Self::Iter<'_>;
}
