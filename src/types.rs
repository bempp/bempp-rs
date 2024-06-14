//! Types specific to bempp-rs

use rlst::{LinAlg, RlstScalar};

pub trait RealScalar: num::Float + LinAlg + RlstScalar<Real = Self> {}

impl<T: num::Float + LinAlg + RlstScalar<Real = T>> RealScalar for T {}

// A simple Integer Matrix type for storing indices.

pub struct IntegerArray2 {
    data: Vec<usize>,
    dim: [usize; 2],
}

impl IntegerArray2 {
    pub fn new(dim: [usize; 2]) -> Self {
        let nelems = dim.iter().product();
        Self {
            data: vec![0; nelems],
            dim,
        }
    }

    pub fn new_from_slice(data: &[usize], dim: [usize; 2]) -> Self {
        let nelems = dim.iter().product();
        assert_eq!(data.len(), nelems);
        Self {
            data: data.to_vec(),
            dim,
        }
    }

    pub fn dim(&self) -> [usize; 2] {
        self.dim
    }

    pub fn col_iter(&self) -> ColIter<'_> {
        ColIter {
            arr: self,
            index: 0,
        }
    }

    pub fn col_iter_mut(&mut self) -> ColIterMut<'_> {
        ColIterMut {
            arr: self,
            index: 0,
        }
    }
}

impl std::ops::Index<[usize; 2]> for IntegerArray2 {
    type Output = usize;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[self.dim[0] * index[1] + index[0]]
    }
}

impl std::ops::IndexMut<[usize; 2]> for IntegerArray2 {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.data[self.dim[0] * index[1] + index[0]]
    }
}

pub struct ColIter<'a> {
    arr: &'a IntegerArray2,
    index: usize,
}

impl<'a> std::iter::Iterator for ColIter<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.arr.dim()[0];
        let index = self.index;
        self.index += 1;
        if self.index < self.arr.dim[1] {
            Some(&self.arr.data[index * nrows..(1 + index) * nrows])
        } else {
            None
        }
    }
}

pub struct ColIterMut<'a> {
    arr: &'a mut IntegerArray2,
    index: usize,
}

impl<'a> std::iter::Iterator for ColIterMut<'a> {
    type Item = &'a mut [usize];

    fn next(&mut self) -> Option<Self::Item> {
        let nrows = self.arr.dim()[0];
        let index = self.index;
        self.index += 1;
        if self.index < self.arr.dim[1] {
            Some(unsafe {
                std::mem::transmute(&mut self.arr.data[index * nrows..(1 + index) * nrows])
            })
        } else {
            None
        }
    }
}
