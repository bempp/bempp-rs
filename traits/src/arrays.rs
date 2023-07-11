//! Containers to store multi-dimensional data
use num::Num;

pub trait Array1DAccess<'a, T: Num> {
    type I: Iterator;

    /// Get an item from the array
    fn get(&self, index: usize) -> Option<&T>;

    /// Get a mutable item from the array
    fn get_mut(&mut self, index: usize) -> Option<&mut T>;

    /// Get an item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked(&self, index: usize) -> &T;

    /// Get a mutable item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T;

    /// Get the shape of the array
    fn shape(&self) -> &usize;

    /// Iterate through the values
    fn iter(&'a self) -> Self::I;
}

pub trait Array2DAccess<'a, T: Num> {
    type I: Iterator;

    /// Get an item from the array
    fn get(&self, index0: usize, index1: usize) -> Option<&T>;

    /// Get a mutable item from the array
    fn get_mut(&mut self, index0: usize, index1: usize) -> Option<&mut T>;

    /// Get a row of the array
    fn row(&self, index: usize) -> Option<&[T]>;

    /// Get an item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked(&self, index0: usize, index1: usize) -> &T;

    /// Get a mutable item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked_mut(&mut self, index0: usize, index1: usize) -> &mut T;

    /// Get a row of the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn row_unchecked(&self, index: usize) -> &[T];

    /// Get the shape of the array
    fn shape(&self) -> &(usize, usize);

    /// Iterate through the rows
    fn iter_rows(&'a self) -> Self::I;
}

pub trait Array3DAccess<T: Num> {
    /// Get an item from the array
    fn get(&self, index0: usize, index1: usize, index2: usize) -> Option<&T>;

    /// Get a mutable item from the array
    fn get_mut(&mut self, index0: usize, index1: usize, index2: usize) -> Option<&mut T>;

    /// Get an item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked(&self, index0: usize, index1: usize, index2: usize) -> &T;

    /// Get a mutable item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked_mut(&mut self, index0: usize, index1: usize, index2: usize) -> &mut T;

    /// Get the shape of the array
    fn shape(&self) -> &(usize, usize, usize);

    /// Get a pointer to the raw data in the array
    fn get_data(&self) -> &[T];
}

pub trait Array4DAccess<T: Num> {
    /// Get an item from the array
    fn get(&self, index0: usize, index1: usize, index2: usize, index3: usize) -> Option<&T>;

    /// Get a mutable item from the array
    fn get_mut(
        &mut self,
        index0: usize,
        index1: usize,
        index2: usize,
        index3: usize,
    ) -> Option<&mut T>;

    /// Get an item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked(
        &self,
        index0: usize,
        index1: usize,
        index2: usize,
        index3: usize,
    ) -> &T;

    /// Get a mutable item from the array without checking bounds
    ///
    /// # Safety
    /// This function does not perform bound checks
    unsafe fn get_unchecked_mut(
        &mut self,
        index0: usize,
        index1: usize,
        index2: usize,
        index3: usize,
    ) -> &mut T;

    /// Get the shape of the array
    fn shape(&self) -> &(usize, usize, usize, usize);
}

pub trait AdjacencyListAccess<'a, T: Num> {
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
