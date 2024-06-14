//! Default definitions of entities.

use std::iter::Copied;

use num::Float;
use rlst::{LinAlg, RlstScalar};

use crate::{
    traits::{grid::Grid, types::ReferenceCellType},
    types::RealScalar,
};

use super::FlatTriangleGrid;

/// A point
pub struct Vertex<T: RealScalar> {
    index: usize,
    coordinates: [T; 3],
}

impl<T: RealScalar> Vertex<T> {
    pub fn new(index: usize, coordinates: [T; 3]) -> Self {
        Self { index, coordinates }
    }
}

impl<T: RealScalar> crate::traits::grid::Vertex for Vertex<T> {
    type T = T;

    fn coords(&self) -> [Self::T; 3] {
        self.coordinates
    }

    fn index(&self) -> usize {
        self.index
    }
}

/// An edge
pub struct Edge<'a, T: RealScalar> {
    index: usize,
    vertices: [usize; 2],
    grid: &'a FlatTriangleGrid<T>,
}

impl<'a, T: RealScalar> Edge<'a, T> {
    pub fn new(index: usize, vertices: [usize; 2], grid: &'a FlatTriangleGrid<T>) -> Self {
        Self {
            index,
            vertices,
            grid,
        }
    }
}

impl<'a, T: RealScalar> crate::traits::grid::Edge for Edge<'a, T> {
    type Grid = FlatTriangleGrid<T>;

    type Iter<'b> = crate::traits::types::VertexIterator<'b, FlatTriangleGrid<T>, Copied<std::slice::Iter<'b, usize>>>
    where
        Self: 'b;

    fn index(&self) -> usize {
        self.index
    }

    fn vertices(&self) -> Self::Iter<'_> {
        Self::Iter::<'_>::new(self.vertices.as_slice().iter().copied(), self.grid)
    }
}

pub struct Cell<'a, T: RealScalar> {
    index: usize,
    vertices: [usize; 3],
    grid: &'a FlatTriangleGrid<T>,
}

pub struct Topology<T: RealScalar> {
    vertex_indices: [usize; 3],
    edge_indices: [usize; 3],
    cell_type: ReferenceCellType,
    index: usize,
}

impl<T: RealScalar> crate::traits::grid::Topology for Topology<T> {
    type Grid = FlatTriangleGrid<T>;

    type VertexIndexIter<'b> = Copied<std::slice::Iter<'b, usize>>
    where
        Self: 'b;

    type EdgeIndexIter<'b> = Copied<std::slice::Iter<'b, usize>>
    where
        Self: 'b;

    fn vertex_indices(&self) -> Self::VertexIndexIter<'_> {
        self.vertex_indices.iter().copied()
    }

    fn edge_indices(&self) -> Self::EdgeIndexIter<'_> {
        self.edge_indices.iter().copied()
    }

    fn cell_type(&self) -> crate::traits::types::ReferenceCellType {
        todo!()
    }
}
