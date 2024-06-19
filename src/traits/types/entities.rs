//! Default definitions of entities.

use std::iter::Copied;

use num::Float;
use rlst::{LinAlg, RlstScalar};

use crate::traits::{
    grid::Grid,
    types::{PointIterator, ReferenceCell},
};
use crate::types::RealScalar;

use super::{EdgeIterator, Ownership};

/// A point
pub struct Point<'a, G: Grid> {
    index: usize,
    grid: &'a G,
}

impl<'a, G: Grid> Point<'a, G> {
    pub fn new(index: usize, grid: &'a G) -> Self {
        Self { index, grid }
    }
}

impl<'a, G: Grid> crate::traits::grid::Point for Point<'a, G> {
    type G = G;

    fn coords(&self) -> [<Self::G as Grid>::T; 3] {
        self.grid.coordinates_from_point_index(self.index)
    }

    fn local_index(&self) -> usize {
        self.index
    }

    fn global_index(&self) -> usize {
        self.grid.global_point_index(self.index)
    }

    fn ownership(&self) -> Ownership {
        self.grid.point_ownership(self.index)
    }
}

/// An edge
pub struct Edge<'a, G: Grid> {
    index: usize,
    grid: &'a G,
}

impl<'a, G: Grid> Edge<'a, G> {
    pub fn new(index: usize, grid: &'a G) -> Self {
        Self { index, grid }
    }
}

impl<'a, G: Grid> crate::traits::grid::Edge for Edge<'a, G> {
    type G = G;

    fn local_index(&self) -> usize {
        self.index
    }

    fn points(&self) -> super::PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>> {
        let point_indices = self.grid.edge_to_points(self.index);
        PointIterator::new(point_indices.iter().copied(), self.grid)
    }

    fn end_points(&self) -> super::PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>> {
        let point_indices = self.grid.edge_to_end_points(self.index);
        PointIterator::new(point_indices.iter().copied(), self.grid)
    }

    fn global_index(&self) -> usize {
        self.grid.global_edge_index(self.index)
    }

    fn ownership(&self) -> Ownership {
        self.grid.edge_ownership(self.index)
    }
}

pub struct Cell<'a, G: Grid> {
    index: usize,
    grid: &'a G,
}

pub struct Topology<'a, G: Grid> {
    index: usize,
    grid: &'a G,
}

impl<'a, G: Grid> crate::traits::grid::Topology for Topology<'a, G> {
    type G = G;

    fn points(&self) -> PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>> {
        let points = self.grid.cell_to_points(self.index);
        PointIterator::new(points.iter().copied(), self.grid)
    }

    fn corner_points(&self) -> PointIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>> {
        let points = self.grid.cell_to_points(self.index);
        PointIterator::new(points.iter().copied(), self.grid)
    }

    fn edges(&self) -> super::EdgeIterator<'_, Self::G, Copied<std::slice::Iter<'_, usize>>> {
        let edges = self.grid.cell_to_edges(self.index);
        EdgeIterator::new(edges.iter().copied(), self.grid)
    }

    fn cell_type(&self) -> ReferenceCell {
        self.grid.cell_type(self.index)
    }
}
