//! Iterator over cells

use crate::traits::grid::Grid;

/// A cell iterator
pub struct CellIterator<'a, G: Grid, Iter: std::iter::Iterator<Item = usize>> {
    iter: Iter,
    grid: &'a G,
}

impl<'a, G: Grid, Iter: std::iter::Iterator<Item = usize>> CellIterator<'a, G, Iter> {
    /// Create a cell iterator
    pub fn new(iter: Iter, grid: &'a G) -> Self {
        CellIterator { iter, grid }
    }
}

impl<'a, G: Grid, Iter: std::iter::Iterator<Item = usize>> std::iter::Iterator
    for CellIterator<'a, G, Iter>
{
    type Item = G::Cell<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.iter.next() {
            Some(self.grid.cell_from_index(index))
        } else {
            None
        }
    }
}
