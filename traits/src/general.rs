//! Traits useful across packages.

/// An indexable data view.
pub trait IndexableView {
    /// The Item type
    type Item;

    /// Get data at position.
    fn get(&self, index: usize) -> Option<&Self::Item>;

    /// Get mutable data at position.
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item>;

    /// Get unchecked access to data.
    ///
    /// # Safety
    /// This function does not complete bound checks
    unsafe fn get_unchecked(&self, index: usize) -> Option<&Self::Item>;

    /// Get unchecked mutable access to data.
    ///
    /// # Safety
    /// This function does not complete bound checks
    unsafe fn get_unchecked_mut(&self, index: usize) -> Option<&mut Self::Item>;

    // TODO: Iterator
}
