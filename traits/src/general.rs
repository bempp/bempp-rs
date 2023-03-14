//! Traits useful across packages.

// TODO Do we need this?
// An indexable data view.
pub trait IndexableView {
    // The Item type
    type Item;

    // Get data at position.
    fn get(&self, index: usize) -> Option<&Self::Item>;

    // Get mutable data at position.
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item>;

    // Get unchecked access to data.
    unsafe fn get_unchecked(&self, index: usize) -> Option<&Self::Item>;

    // Get unchecked mutable access to data.
    unsafe fn get_unchecked_mut(&self, index: usize) -> Option<&mut Self::Item>;

    // TODO: Iterator, Bracket notation
}

// Add view for 2d/3d ... container
