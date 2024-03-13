//! Containers to store multi-dimensional data
use num::Num;

/// REMOVED
pub trait AdjacencyListAccess<'a, T: Num> {
    /// removed
    type I: Iterator;

    /// Get an item from the adjacency list
    fn get(&self, index0: usize, index1: usize) -> Option<&T>;

    /// Get a mutable item from the adjacency list
    fn get_mut(&mut self, index0: usize, index1: usize) -> Option<&mut T>;

    /// Get a row from the adjacency list
    fn row(&self, index: usize) -> Option<&[T]>;

    /// Get an item from the adjacency list without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked(&self, index0: usize, index1: usize) -> &T;

    /// Get a mutable item from the adjacency list without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked_mut(&mut self, index0: usize, index1: usize) -> &mut T;

    /// Get a row from the adjacency list without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn row_unchecked(&self, index: usize) -> &[T];

    /// Get the vector of offsets
    fn offsets(&self) -> &[usize];

    /// Get the number of rows
    fn num_rows(&self) -> usize;

    /// Iterate through the rows
    fn iter_rows(&'a self) -> Self::I;
}
