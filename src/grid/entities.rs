//! Default definitions of entities.

use std::marker::PhantomData;

use num::Float;
use rlst::RlstScalar;

/// A point
pub struct Point<T: Float + RlstScalar<Real = T>> {
    index: usize,
    coordinates: [T; 3],
}

impl<T: Float + RlstScalar<Real = T>> Point<T> {
    pub fn new(index: usize, coordinates: [T; 3]) -> Self {
        Self { index, coordinates }
    }
}

impl<T: Float + RlstScalar<Real = T>> crate::traits::grid::Point for Point<T> {
    type T = T;

    fn coords(&self) -> [Self::T; 3] {
        self.coordinates
    }

    fn index(&self) -> usize {
        self.index
    }
}

/// An edge
pub struct Edge {
    index: usize,
    point_indices: (usize, usize),
}

/// A cell
pub struct Cell<T: Float + RlstScalar<Real = T>> {
    index: usize,
    _t: PhantomData<T>,
}
