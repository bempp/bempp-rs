//! Iterator over cells

use crate::traits::grid::GridType;

/// A cell iterator
pub struct CellIterator<'a, Grid: GridType, Iter: std::iter::Iterator<Item = usize>> {
    iter: Iter,
    grid: &'a Grid,
}

impl<'a, Grid: GridType, Iter: std::iter::Iterator<Item = usize>> CellIterator<'a, Grid, Iter> {
    /// Create a cell iterator
    pub fn new(iter: Iter, grid: &'a Grid) -> Self {
        CellIterator { iter, grid }
    }
}

impl<'a, Grid: GridType, Iter: std::iter::Iterator<Item = usize>> std::iter::Iterator
    for CellIterator<'a, Grid, Iter>
{
    type Item = Grid::Cell<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.iter.next() {
            Some(self.grid.cell_from_index(index))
        } else {
            None
        }
    }
}
