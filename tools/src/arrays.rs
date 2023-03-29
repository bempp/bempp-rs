//! Containers to store multi-dimensional data
use num::Num;
use std::clone::Clone;

/// A two-dimensional rectangular array
pub struct Array2D<T> {
    /// The data in the array, in row-major order
    data: Vec<T>,
    /// The shape of the array
    shape: (usize, usize),
}

impl<T: Num + Clone> Array2D<T> {
    /// Create an array from a data vector
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            data: vec![T::zero(); shape.0 * shape.1],
            shape: shape,
        }
    }
}

impl<T> Array2D<T> {
    /// Create an array from a data vector
    pub fn from_data(data: Vec<T>, shape: (usize, usize)) -> Self {
        Self {
            data: data,
            shape: shape,
        }
    }
    /// Get an item from the array
    pub fn get(&self, index0: usize, index1: usize) -> Option<&T> {
        if index0 >= self.shape.0 || index1 >= self.shape.1 {
            None
        } else {
            unsafe { Some(self.get_unchecked(index0, index1)) }
        }
    }
    /// Get a mutable item from the array
    pub fn get_mut(&mut self, index0: usize, index1: usize) -> Option<&mut T> {
        if index0 >= self.shape.0 || index1 >= self.shape.1 {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index0, index1)) }
        }
    }
    /// Get a row of the array
    pub fn row(&self, index: usize) -> Option<&[T]> {
        if index >= self.shape.0 {
            None
        } else {
            unsafe { Some(&self.row_unchecked(index)) }
        }
    }
    /// Get an item from the array without checking bounds
    pub unsafe fn get_unchecked(&self, index0: usize, index1: usize) -> &T {
        self.data.get_unchecked(index0 * self.shape.1 + index1)
    }
    /// Get a mutable item from the array without checking bounds
    pub unsafe fn get_unchecked_mut(&mut self, index0: usize, index1: usize) -> &mut T {
        self.data.get_unchecked_mut(index0 * self.shape.1 + index1)
    }
    /// Get a row of the array without checking bounds
    pub unsafe fn row_unchecked(&self, index: usize) -> &[T] {
        &self.data[index * self.shape.1..(index + 1) * self.shape.1]
    }
    /// Get the shape of the array
    pub fn shape(&self) -> &(usize, usize) {
        &self.shape
    }
    /// Iterate through the rows
    pub fn iter_rows(&self) -> Array2DRowIterator<'_, T> {
        Array2DRowIterator::<T> {
            array: &self,
            index: 0,
        }
    }
}

pub struct Array2DRowIterator<'a, T> {
    array: &'a Array2D<T>,
    index: usize,
}

impl<'a, T> Iterator for Array2DRowIterator<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        self.array.row(self.index - 1)
    }
}

/// A three-dimensional rectangular array
pub struct Array3D<T> {
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
            shape: shape,
        }
    }
}

impl<T> Array3D<T> {
    /// Create an array from a data vector
    pub fn from_data(data: Vec<T>, shape: (usize, usize, usize)) -> Self {
        Self {
            data: data,
            shape: shape,
        }
    }
    /// Get an item from the array
    pub fn get(&self, index0: usize, index1: usize, index2: usize) -> Option<&T> {
        if index0 >= self.shape.0 || index1 >= self.shape.1 || index2 >= self.shape.2 {
            None
        } else {
            unsafe { Some(self.get_unchecked(index0, index1, index2)) }
        }
    }
    /// Get a mutable item from the array
    pub fn get_mut(&mut self, index0: usize, index1: usize, index2: usize) -> Option<&mut T> {
        if index0 >= self.shape.0 || index1 >= self.shape.1 || index2 >= self.shape.2 {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index0, index1, index2)) }
        }
    }
    /// Get an item from the array without checking bounds
    pub unsafe fn get_unchecked(&self, index0: usize, index1: usize, index2: usize) -> &T {
        self.data
            .get_unchecked((index0 * self.shape.1 + index1) * self.shape.2 + index2)
    }
    /// Get a mutable item from the array without checking bounds
    pub unsafe fn get_unchecked_mut(
        &mut self,
        index0: usize,
        index1: usize,
        index2: usize,
    ) -> &mut T {
        self.data
            .get_unchecked_mut((index0 * self.shape.1 + index1) * self.shape.2 + index2)
    }
    /// Get the shape of the array
    pub fn shape(&self) -> &(usize, usize, usize) {
        &self.shape
    }
}

/// A four-dimensional rectangular array
pub struct Array4D<T> {
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
            shape: shape,
        }
    }
}

impl<T> Array4D<T> {
    /// Create an array from a data vector
    pub fn from_data(data: Vec<T>, shape: (usize, usize, usize, usize)) -> Self {
        Self {
            data: data,
            shape: shape,
        }
    }
    /// Get an item from the array
    pub fn get(&self, index0: usize, index1: usize, index2: usize, index3: usize) -> Option<&T> {
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
    /// Get a mutable item from the array
    pub fn get_mut(
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
    /// Get an item from the array without checking bounds
    pub unsafe fn get_unchecked(
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
    /// Get a mutable item from the array without checking bounds
    pub unsafe fn get_unchecked_mut(
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
    /// Get the shape of the array
    pub fn shape(&self) -> &(usize, usize, usize, usize) {
        &self.shape
    }
}

/// An adjacency list
///
/// An adjacency list stores two-dimensional data where each row may have a different number of items
pub struct AdjacencyList<T> {
    /// The data in the array, in row-major order
    data: Vec<T>,
    /// The starting index of each row, plus a final entry that is the length of data
    offsets: Vec<usize>,
}

impl<T: Num + Clone> AdjacencyList<T> {
    /// Create an adjacency list
    pub fn new() -> Self {
        Self {
            data: vec![T::zero(); 0],
            offsets: vec![0],
        }
    }
}

impl<T: Num + Copy> AdjacencyList<T> {
    /// Add a new row of data to the end
    pub fn add_row(&mut self, row: &[T]) {
        for i in row {
            self.data.push(*i);
        }
        self.offsets.push(self.offsets.last().unwrap() + row.len());
    }
}

impl<T> AdjacencyList<T> {
    /// Create an adjacency list
    pub fn from_data(data: Vec<T>, offsets: Vec<usize>) -> Self {
        if offsets[offsets.len() - 1] != data.len() {
            panic!("Final offset must be the length of the data.");
        }
        Self {
            data: data,
            offsets: offsets,
        }
    }
    /// Get an item from the adjacency list
    pub fn get(&self, index0: usize, index1: usize) -> Option<&T> {
        if index0 >= self.offsets.len() - 1
            || self.offsets[index0] + index1 >= self.offsets[index0 + 1]
        {
            None
        } else {
            unsafe { Some(self.get_unchecked(index0, index1)) }
        }
    }
    /// Get a mutable item from the adjacency list
    pub fn get_mut(&mut self, index0: usize, index1: usize) -> Option<&mut T> {
        if index0 >= self.offsets.len() - 1
            || self.offsets[index0] + index1 >= self.offsets[index0 + 1]
        {
            None
        } else {
            unsafe { Some(self.get_unchecked_mut(index0, index1)) }
        }
    }
    /// Get a row from the adjacency list
    pub fn row(&self, index: usize) -> Option<&[T]> {
        if index >= self.offsets.len() - 1 {
            None
        } else {
            unsafe { Some(self.row_unchecked(index)) }
        }
    }
    /// Get an item from the adjacency list without checking bounds
    pub unsafe fn get_unchecked(&self, index0: usize, index1: usize) -> &T {
        self.data.get_unchecked(self.offsets[index0] + index1)
    }
    /// Get a mutable item from the adjacency list without checking bounds
    pub unsafe fn get_unchecked_mut(&mut self, index0: usize, index1: usize) -> &mut T {
        self.data.get_unchecked_mut(self.offsets[index0] + index1)
    }
    /// Get a row from the adjacency list without checking bounds
    pub unsafe fn row_unchecked(&self, index: usize) -> &[T] {
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }
    /// Get the vector of offsets
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }
    /// Get the number of rows
    pub fn num_rows(&self) -> usize {
        self.offsets.len() - 1
    }
    /// Iterate through the rows
    pub fn iter_rows(&self) -> AdjacencyListRowIterator<'_, T> {
        AdjacencyListRowIterator::<T> {
            alist: &self,
            index: 0,
        }
    }
}

pub struct AdjacencyListRowIterator<'a, T> {
    alist: &'a AdjacencyList<T>,
    index: usize,
}

impl<'a, T> Iterator for AdjacencyListRowIterator<'a, T> {
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
    fn test_array_2d() {
        let mut arr = Array2D::from_data(vec![1, 2, 3, 4, 5, 6], (2, 3));
        assert_eq!(*arr.get(0, 0).unwrap(), 1);
        assert_eq!(*arr.get(0, 1).unwrap(), 2);
        assert_eq!(*arr.get(0, 2).unwrap(), 3);
        assert_eq!(*arr.get(1, 0).unwrap(), 4);
        assert_eq!(*arr.get(1, 1).unwrap(), 5);
        assert_eq!(*arr.get(1, 2).unwrap(), 6);

        let row1 = arr.row(1).unwrap();
        assert_eq!(row1.len(), 3);
        assert_eq!(row1[0], 4);
        assert_eq!(row1[1], 5);
        assert_eq!(row1[2], 6);

        *arr.get_mut(1, 2).unwrap() = 7;
        assert_eq!(*arr.get(1, 2).unwrap(), 7);

        let mut index = 0;
        for row in arr.iter_rows() {
            assert_eq!(*arr.get(index, 0).unwrap(), row[0]);
            index += 1;
        }

        let mut arr2 = Array2D::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        assert_eq!(*arr2.get(0, 0).unwrap(), 1.0);
        assert_eq!(*arr2.get(0, 1).unwrap(), 2.0);
        assert_eq!(*arr2.get(0, 2).unwrap(), 3.0);
        assert_eq!(*arr2.get(1, 0).unwrap(), 4.0);
        assert_eq!(*arr2.get(1, 1).unwrap(), 5.0);
        assert_eq!(*arr2.get(1, 2).unwrap(), 6.0);

        let row1 = arr2.row(1).unwrap();
        assert_eq!(row1.len(), 3);
        assert_eq!(row1[0], 4.0);
        assert_eq!(row1[1], 5.0);
        assert_eq!(row1[2], 6.0);

        *arr2.get_mut(1, 2).unwrap() = 7.;
        assert_eq!(*arr2.get(1, 2).unwrap(), 7.0);

        let mut arr3 = Array2D::<usize>::new((4, 5));
        assert_eq!(*arr3.get(1, 2).unwrap(), 0);
        *arr3.get_mut(1, 2).unwrap() = 5;
        assert_eq!(*arr3.get(1, 2).unwrap(), 5);

        let mut arr4 = Array2D::<f32>::new((4, 5));
        assert_eq!(*arr4.get(1, 2).unwrap(), 0.0);
        *arr4.get_mut(1, 2).unwrap() = 5.0;
        assert_eq!(*arr4.get(1, 2).unwrap(), 5.0);
    }

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

        let mut index = 0;
        for row in arr.iter_rows() {
            assert_eq!(*arr.get(index, 0).unwrap(), row[0]);
            index += 1;
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
