//! Iterator over points

use crate::traits::grid::Grid;

/// An iterator over points
pub struct VertexIterator<'a, G: Grid, Iter: std::iter::Iterator<Item = usize>> {
    iter: Iter,
    grid: &'a G,
}

impl<'a, G: Grid, Iter: std::iter::Iterator<Item = usize>> VertexIterator<'a, G, Iter> {
    /// Create an iterator over points
    pub fn new(iter: Iter, grid: &'a G) -> Self {
        Self { iter, grid }
    }
}

impl<'a, G: Grid, Iter: std::iter::Iterator<Item = usize>> std::iter::Iterator
    for VertexIterator<'a, G, Iter>
{
    type Item = G::Vertex<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.iter.next() {
            Some(self.grid.vertex_from_index(index))
        } else {
            None
        }
    }
}
