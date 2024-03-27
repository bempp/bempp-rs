//! A parallel implementation of a grid
use crate::grid::traits::{Geometry, Grid, Topology};
use crate::traits::types::{CellLocalIndexPair, Ownership, ReferenceCellType};
use mpi::topology::Communicator;
use std::collections::HashMap;

/// Parallel grid
pub struct ParallelGrid<'comm, C: Communicator, G: Grid> {
    pub(crate) comm: &'comm C,
    pub(crate) serial_grid: G,
    vertex_ownership: HashMap<<<G as Grid>::Topology as Topology>::IndexType, Ownership>,
    cell_ownership: HashMap<<<G as Grid>::Topology as Topology>::IndexType, Ownership>,
}

impl<'comm, C: Communicator, G: Grid> ParallelGrid<'comm, C, G> {
    /// Create new parallel grid
    pub fn new(
        comm: &'comm C,
        serial_grid: G,
        vertex_ownership: HashMap<<<G as Grid>::Topology as Topology>::IndexType, Ownership>,
        cell_ownership: HashMap<<<G as Grid>::Topology as Topology>::IndexType, Ownership>,
    ) -> Self {
        Self {
            comm,
            serial_grid,
            vertex_ownership,
            cell_ownership,
        }
    }
}

impl<'comm, C: Communicator, G: Grid> Geometry for ParallelGrid<'comm, C, G> {
    type IndexType = <<G as Grid>::Geometry as Geometry>::IndexType;
    type T = <<G as Grid>::Geometry as Geometry>::T;
    type Element = <<G as Grid>::Geometry as Geometry>::Element;
    type Evaluator<'a> = <<G as Grid>::Geometry as Geometry>::Evaluator<'a> where Self: 'a;

    fn dim(&self) -> usize {
        self.serial_grid.geometry().dim()
    }

    fn index_map(&self) -> &[Self::IndexType] {
        self.serial_grid.geometry().index_map()
    }
    fn coordinate(&self, point_index: usize, coord_index: usize) -> Option<&Self::T> {
        self.serial_grid
            .geometry()
            .coordinate(point_index, coord_index)
    }
    fn point_count(&self) -> usize {
        self.serial_grid.geometry().point_count()
    }
    fn cell_points(&self, index: Self::IndexType) -> Option<&[usize]> {
        self.serial_grid.geometry().cell_points(index)
    }
    fn cell_count(&self) -> usize {
        self.serial_grid.geometry().cell_count()
    }
    fn cell_element(&self, index: Self::IndexType) -> Option<&Self::Element> {
        self.serial_grid.geometry().cell_element(index)
    }
    fn element_count(&self) -> usize {
        self.serial_grid.geometry().element_count()
    }
    fn element(&self, i: usize) -> Option<&Self::Element> {
        self.serial_grid.geometry().element(i)
    }
    fn cell_indices(&self, i: usize) -> Option<&[Self::IndexType]> {
        self.serial_grid.geometry().cell_indices(i)
    }
    fn midpoint(&self, index: Self::IndexType, point: &mut [Self::T]) {
        self.serial_grid.geometry().midpoint(index, point)
    }
    fn diameter(&self, index: Self::IndexType) -> Self::T {
        self.serial_grid.geometry().diameter(index)
    }
    fn volume(&self, index: Self::IndexType) -> Self::T {
        self.serial_grid.geometry().volume(index)
    }
    fn get_evaluator<'a>(&'a self, points: &'a [Self::T]) -> Self::Evaluator<'a> {
        self.serial_grid.geometry().get_evaluator(points)
    }
    fn point_index_to_id(&self, index: usize) -> usize {
        self.serial_grid.geometry().point_index_to_id(index)
    }
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize {
        self.serial_grid.geometry().cell_index_to_id(index)
    }
    fn point_id_to_index(&self, id: usize) -> usize {
        self.serial_grid.geometry().point_id_to_index(id)
    }
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType {
        self.serial_grid.geometry().cell_id_to_index(id)
    }
}

impl<'comm, C: Communicator, G: Grid> Topology for ParallelGrid<'comm, C, G> {
    type IndexType = <<G as Grid>::Topology as Topology>::IndexType;

    fn dim(&self) -> usize {
        self.serial_grid.topology().dim()
    }
    fn index_map(&self) -> &[Self::IndexType] {
        self.serial_grid.topology().index_map()
    }
    fn entity_count(&self, etype: ReferenceCellType) -> usize {
        self.serial_grid.topology().entity_count(etype)
    }
    fn entity_count_by_dim(&self, dim: usize) -> usize {
        self.serial_grid.topology().entity_count_by_dim(dim)
    }
    fn cell(&self, index: Self::IndexType) -> Option<&[Self::IndexType]> {
        self.serial_grid.topology().cell(index)
    }
    fn cell_type(&self, index: Self::IndexType) -> Option<ReferenceCellType> {
        self.serial_grid.topology().cell_type(index)
    }
    fn entity_types(&self, dim: usize) -> &[ReferenceCellType] {
        self.serial_grid.topology().entity_types(dim)
    }
    fn cell_to_entities(&self, index: Self::IndexType, dim: usize) -> Option<&[Self::IndexType]> {
        self.serial_grid.topology().cell_to_entities(index, dim)
    }
    fn cell_to_flat_entities(&self, index: Self::IndexType, dim: usize) -> Option<&[usize]> {
        self.serial_grid
            .topology()
            .cell_to_flat_entities(index, dim)
    }
    fn entity_to_cells(
        &self,
        dim: usize,
        index: Self::IndexType,
    ) -> Option<&[CellLocalIndexPair<Self::IndexType>]> {
        self.serial_grid.topology().entity_to_cells(dim, index)
    }
    fn entity_to_flat_cells(
        &self,
        dim: usize,
        index: Self::IndexType,
    ) -> Option<&[CellLocalIndexPair<usize>]> {
        self.serial_grid.topology().entity_to_flat_cells(dim, index)
    }
    fn entity_vertices(&self, dim: usize, index: Self::IndexType) -> Option<&[Self::IndexType]> {
        self.serial_grid.topology().entity_vertices(dim, index)
    }
    fn cell_ownership(&self, index: Self::IndexType) -> Ownership {
        self.cell_ownership[&index]
    }
    fn vertex_ownership(&self, index: Self::IndexType) -> Ownership {
        self.vertex_ownership[&index]
    }
    fn vertex_index_to_id(&self, index: Self::IndexType) -> usize {
        self.serial_grid.topology().vertex_index_to_id(index)
    }
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize {
        self.serial_grid.topology().cell_index_to_id(index)
    }
    fn vertex_id_to_index(&self, id: usize) -> Self::IndexType {
        self.serial_grid.topology().vertex_id_to_index(id)
    }
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType {
        self.serial_grid.topology().cell_id_to_index(id)
    }
    fn vertex_index_to_flat_index(&self, index: Self::IndexType) -> usize {
        self.serial_grid
            .topology()
            .vertex_index_to_flat_index(index)
    }
    fn edge_index_to_flat_index(&self, index: Self::IndexType) -> usize {
        self.serial_grid.topology().edge_index_to_flat_index(index)
    }
    fn face_index_to_flat_index(&self, index: Self::IndexType) -> usize {
        self.serial_grid.topology().face_index_to_flat_index(index)
    }
    fn vertex_flat_index_to_index(&self, index: usize) -> Self::IndexType {
        self.serial_grid
            .topology()
            .vertex_flat_index_to_index(index)
    }
    fn edge_flat_index_to_index(&self, index: usize) -> Self::IndexType {
        self.serial_grid.topology().edge_flat_index_to_index(index)
    }
    fn face_flat_index_to_index(&self, index: usize) -> Self::IndexType {
        self.serial_grid.topology().face_flat_index_to_index(index)
    }
    fn cell_types(&self) -> &[ReferenceCellType] {
        self.serial_grid.topology().cell_types()
    }
}

impl<'a, C: Communicator, G: Grid> Grid for ParallelGrid<'a, C, G> {
    type T = G::T;
    type Topology = Self;
    type Geometry = Self;

    fn topology(&self) -> &Self::Topology {
        self
    }

    fn geometry(&self) -> &Self::Geometry {
        self
    }

    fn is_serial(&self) -> bool {
        false
    }
}
