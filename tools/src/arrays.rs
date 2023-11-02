//! Containers to store multi-dimensional data
use bempp_traits::arrays::{AdjacencyListAccess, Array3DAccess, Array4DAccess};
use num::Num;
use rlst_dense::{operations::transpose::Scalar, rlst_dynamic_mat, UnsafeRandomAccessMut};
use std::clone::Clone;

pub type Mat<T> = rlst_dense::Matrix<
    T,
    rlst_dense::base_matrix::BaseMatrix<T, rlst_dense::VectorContainer<T>, rlst_dense::Dynamic>,
    rlst_dense::Dynamic,
>;

pub fn to_matrix<T: Scalar>(data: &[T], shape: (usize, usize)) -> Mat<T> {
    let mut mat = rlst_dynamic_mat![T, shape];
    for (i, d) in data.iter().enumerate() {
        unsafe {
            *mat.get_unchecked_mut(i % shape.0, i / shape.0) = *d;
        }
    }
    mat
}

pub fn zero_matrix<T: Scalar>(shape: (usize, usize)) -> Mat<T> {
    rlst_dynamic_mat![T, shape]
}

/// A three-dimensional rectangular array
#[derive(Clone)]
pub struct Array3D<T: Num> {
    /// The data in the array, in row-major order
    data: Vec<T>,
    /// The shape of the array
    shape: (usize, usize, usize),
}

impl<T: Num + Clone> Array3D<T> {
    /// Create an array from a data vector
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            data: vec![T::zero(); shape.0 * shape.1 * shape.2],
            shape,
        }
    }
    /// Create an array from a data vector
    pub fn from_data(data: Vec<T>, shape: (usize, usize, usize)) -> Self {
        assert_eq!(data.len(), shape.0 * shape.1 * shape.2);
        Self { data, shape }
    }
}

impl<T: Num> Array3DAccess<T> for Array3D<T> {
    fn get(&self, index0: usize, index1: usize, index2: usize) -> Option<&T> {
        if index0 >= self.shape.0 || index1 >= self.shape.1 || index2 >= self.shape.2 {
            None
        } else {
            unsafe { Some(self.get_unchecked(index0, index1, index2)) }
        }
    }
    fn get_mut(&mut self, index0: usize, index1: usize, index2: usize) -> Option<&mut T> {
        if index0 >= self.shape.0 || index1 >= self.shape.1 || index2 >= self.shape.2 {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index0, index1, index2)) }
        }
    }
    unsafe fn get_unchecked(&self, index0: usize, index1: usize, index2: usize) -> &T {
        self.data
            .get_unchecked((index0 * self.shape.1 + index1) * self.shape.2 + index2)
    }
    unsafe fn get_unchecked_mut(&mut self, index0: usize, index1: usize, index2: usize) -> &mut T {
        self.data
            .get_unchecked_mut((index0 * self.shape.1 + index1) * self.shape.2 + index2)
    }
    fn shape(&self) -> &(usize, usize, usize) {
        &self.shape
    }

    fn get_data(&self) -> &[T] {
        &self.data
    }

    fn get_data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// A four-dimensional rectangular array
pub struct Array4D<T: Num> {
    /// The data in the array, in row-major order
    data: Vec<T>,
    /// The shape of the array
    shape: (usize, usize, usize, usize),
}

impl<T: Num + Clone> Array4D<T> {
    /// Create an array from a data vector
    pub fn new(shape: (usize, usize, usize, usize)) -> Self {
        Self {
            data: vec![T::zero(); shape.0 * shape.1 * shape.2 * shape.3],
            shape,
        }
    }
    /// Create an array from a data vector
    pub fn from_data(data: Vec<T>, shape: (usize, usize, usize, usize)) -> Self {
        assert_eq!(data.len(), shape.0 * shape.1 * shape.2 * shape.3);
        Self { data, shape }
    }
}

impl<T: Num> Array4DAccess<T> for Array4D<T> {
    fn get(&self, index0: usize, index1: usize, index2: usize, index3: usize) -> Option<&T> {
        if index0 >= self.shape.0
            || index1 >= self.shape.1
            || index2 >= self.shape.2
            || index3 >= self.shape.3
        {
            None
        } else {
            unsafe { Some(self.get_unchecked(index0, index1, index2, index3)) }
        }
    }
    fn get_mut(
        &mut self,
        index0: usize,
        index1: usize,
        index2: usize,
        index3: usize,
    ) -> Option<&mut T> {
        if index0 >= self.shape.0
            || index1 >= self.shape.1
            || index2 >= self.shape.2
            || index3 >= self.shape.3
        {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index0, index1, index2, index3)) }
        }
    }
    unsafe fn get_unchecked(
        &self,
        index0: usize,
        index1: usize,
        index2: usize,
        index3: usize,
    ) -> &T {
        self.data.get_unchecked(
            ((index0 * self.shape.1 + index1) * self.shape.2 + index2) * self.shape.3 + index3,
        )
    }
    unsafe fn get_unchecked_mut(
        &mut self,
        index0: usize,
        index1: usize,
        index2: usize,
        index3: usize,
    ) -> &mut T {
        self.data.get_unchecked_mut(
            ((index0 * self.shape.1 + index1) * self.shape.2 + index2) * self.shape.3 + index3,
        )
    }
    fn shape(&self) -> &(usize, usize, usize, usize) {
        &self.shape
    }
}

/// An adjacency list
///
/// An adjacency list stores two-dimensional data where each row may have a different number of items
pub struct AdjacencyList<T: Num> {
    /// The data in the array, in row-major order
    data: Vec<T>,
    /// The starting index of each row, plus a final entry that is the length of data
    offsets: Vec<usize>,
}

impl<T: Num + Copy> Default for AdjacencyList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Num + Copy> AdjacencyList<T> {
    /// Create an adjacency list
    pub fn new() -> Self {
        Self {
            data: vec![T::zero(); 0],
            offsets: vec![0],
        }
    }
    /// Add a new row of data to the end
    pub fn add_row(&mut self, row: &[T]) {
        for i in row {
            self.data.push(*i);
        }
        self.offsets.push(self.offsets.last().unwrap() + row.len());
    }
    /// Create an adjacency list
    pub fn from_data(data: Vec<T>, offsets: Vec<usize>) -> Self {
        if offsets[offsets.len() - 1] != data.len() {
            panic!("Final offset must be the length of the data.");
        }
        Self { data, offsets }
    }
}

impl<'a, T: Num + 'a> AdjacencyListAccess<'a, T> for AdjacencyList<T> {
    type I = AdjacencyListRowIterator<'a, T>;
    /// Get an item from the adjacency list
    fn get(&self, index0: usize, index1: usize) -> Option<&T> {
        if index0 >= self.offsets.len() - 1
            || self.offsets[index0] + index1 >= self.offsets[index0 + 1]
        {
            None
        } else {
            unsafe { Some(self.get_unchecked(index0, index1)) }
        }
    }
    /// Get a mutable item from the adjacency list
    fn get_mut(&mut self, index0: usize, index1: usize) -> Option<&mut T> {
        if index0 >= self.offsets.len() - 1
            || self.offsets[index0] + index1 >= self.offsets[index0 + 1]
        {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index0, index1)) }
        }
    }
    /// Get a row from the adjacency list
    fn row(&self, index: usize) -> Option<&[T]> {
        if index >= self.offsets.len() - 1 {
            None
        } else {
            unsafe { Some(self.row_unchecked(index)) }
        }
    }
    /// Get an item from the adjacency list without checking bounds
    unsafe fn get_unchecked(&self, index0: usize, index1: usize) -> &T {
        self.data.get_unchecked(self.offsets[index0] + index1)
    }
    /// Get a mutable item from the adjacency list without checking bounds
    unsafe fn get_unchecked_mut(&mut self, index0: usize, index1: usize) -> &mut T {
        self.data.get_unchecked_mut(self.offsets[index0] + index1)
    }
    /// Get a row from the adjacency list without checking bounds
    unsafe fn row_unchecked(&self, index: usize) -> &[T] {
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }
    /// Get the vector of offsets
    fn offsets(&self) -> &[usize] {
        &self.offsets
    }
    /// Get the number of rows
    fn num_rows(&self) -> usize {
        self.offsets.len() - 1
    }
    /// Iterate through the rows
    fn iter_rows(&'a self) -> Self::I {
        AdjacencyListRowIterator::<T> {
            alist: self,
            index: 0,
        }
    }
}

pub struct AdjacencyListRowIterator<'a, T: Num> {
    alist: &'a AdjacencyList<T>,
    index: usize,
}

impl<'a, T: Num> Iterator for AdjacencyListRowIterator<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        self.alist.row(self.index - 1)
    }
}

#[cfg(test)]
mod test {
    use crate::arrays::*;

    #[test]
    fn test_adjacency_list() {
        let mut arr = AdjacencyList::from_data(vec![1, 2, 3, 4, 5, 6], vec![0, 2, 3, 6]);
        assert_eq!(*arr.get(0, 0).unwrap(), 1);
        assert_eq!(*arr.get(0, 1).unwrap(), 2);
        assert_eq!(*arr.get(1, 0).unwrap(), 3);
        assert_eq!(*arr.get(2, 0).unwrap(), 4);
        assert_eq!(*arr.get(2, 1).unwrap(), 5);
        assert_eq!(*arr.get(2, 2).unwrap(), 6);

        let row1 = arr.row(1).unwrap();
        assert_eq!(row1.len(), 1);
        assert_eq!(row1[0], 3);

        *arr.get_mut(2, 0).unwrap() = 7;
        assert_eq!(*arr.get(2, 0).unwrap(), 7);

        for (index, row) in arr.iter_rows().enumerate() {
            assert_eq!(*arr.get(index, 0).unwrap(), row[0]);
        }

        let mut arr2 = AdjacencyList::<f64>::new();
        assert_eq!(arr2.num_rows(), 0);
        arr2.add_row(&[1.0, 2.0, 3.0]);
        arr2.add_row(&[4.0]);
        arr2.add_row(&[5.0, 6.0, 7.0, 8.0]);

        assert_eq!(arr2.num_rows(), 3);
        assert_eq!(*arr2.get(0, 0).unwrap(), 1.0)
    }
}
