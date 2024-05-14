//! A parallel implementation of a grid
use crate::grid::traits::{Grid, Topology};
use crate::traits::types::{CellLocalIndexPair, Ownership, ReferenceCellType};
use mpi::topology::Communicator;
use std::collections::HashMap;

/// Grid local to a process
pub struct LocalGrid<G: Grid> {
    rank: usize,
    serial_grid: G,
    vertex_ownership: HashMap<usize, Ownership>,
    edge_ownership: HashMap<usize, Ownership>,
    cell_ownership: HashMap<<<G as Grid>::Topology as Topology>::IndexType, Ownership>,
}

impl<G: Grid> Topology for LocalGrid<G> {
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
    fn cell(&self, index: Self::IndexType) -> Option<&[usize]> {
        self.serial_grid.topology().cell(index)
    }
    fn cell_type(&self, index: Self::IndexType) -> Option<ReferenceCellType> {
        self.serial_grid.topology().cell_type(index)
    }
    fn entity_types(&self, dim: usize) -> &[ReferenceCellType] {
        self.serial_grid.topology().entity_types(dim)
    }
    fn cell_to_entities(&self, index: Self::IndexType, dim: usize) -> Option<&[usize]> {
        self.serial_grid.topology().cell_to_entities(index, dim)
    }
    fn entity_to_cells(
        &self,
        dim: usize,
        index: usize,
    ) -> Option<&[CellLocalIndexPair<Self::IndexType>]> {
        self.serial_grid.topology().entity_to_cells(dim, index)
    }
    fn entity_to_flat_cells(
        &self,
        dim: usize,
        index: usize,
    ) -> Option<&[CellLocalIndexPair<usize>]> {
        self.serial_grid.topology().entity_to_flat_cells(dim, index)
    }
    fn entity_vertices(&self, dim: usize, index: usize) -> Option<&[usize]> {
        self.serial_grid.topology().entity_vertices(dim, index)
    }
    fn cell_ownership(&self, index: Self::IndexType) -> Ownership {
        self.cell_ownership[&index]
    }
    fn vertex_ownership(&self, index: usize) -> Ownership {
        self.vertex_ownership[&index]
    }
    fn edge_ownership(&self, index: usize) -> Ownership {
        self.edge_ownership[&index]
    }
    fn vertex_index_to_id(&self, index: usize) -> usize {
        self.serial_grid.topology().vertex_index_to_id(index)
    }
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize {
        self.serial_grid.topology().cell_index_to_id(index)
    }
    fn vertex_id_to_index(&self, id: usize) -> usize {
        self.serial_grid.topology().vertex_id_to_index(id)
    }
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType {
        self.serial_grid.topology().cell_id_to_index(id)
    }
    fn face_index_to_flat_index(&self, index: Self::IndexType) -> usize {
        self.serial_grid.topology().face_index_to_flat_index(index)
    }
    fn face_flat_index_to_index(&self, index: usize) -> Self::IndexType {
        self.serial_grid.topology().face_flat_index_to_index(index)
    }
    fn cell_types(&self) -> &[ReferenceCellType] {
        self.serial_grid.topology().cell_types()
    }
}

impl<G: Grid> Grid for LocalGrid<G> {
    type T = G::T;
    type Topology = Self;
    type Geometry = <G as Grid>::Geometry;

    fn mpi_rank(&self) -> usize {
        self.rank
    }

    fn topology(&self) -> &Self::Topology {
        self
    }

    fn geometry(&self) -> &G::Geometry {
        self.serial_grid.geometry()
    }

    fn is_serial(&self) -> bool {
        true
    }
}

/// Parallel grid
pub struct ParallelGrid<'comm, C: Communicator, G: Grid> {
    pub(crate) comm: &'comm C,
    pub(crate) local_grid: LocalGrid<G>,
}

impl<'comm, C: Communicator, G: Grid> ParallelGrid<'comm, C, G> {
    /// Create new parallel grid
    pub fn new(
        comm: &'comm C,
        serial_grid: G,
        vertex_ownership: HashMap<usize, Ownership>,
        edge_ownership: HashMap<usize, Ownership>,
        cell_ownership: HashMap<<<G as Grid>::Topology as Topology>::IndexType, Ownership>,
    ) -> Self {
        let local_grid = LocalGrid {
            rank: comm.rank() as usize,
            serial_grid,
            vertex_ownership,
            edge_ownership,
            cell_ownership,
        };
        Self { comm, local_grid }
    }
}

impl<'comm, C: Communicator, G: Grid> Topology for ParallelGrid<'comm, C, G> {
    type IndexType = <<G as Grid>::Topology as Topology>::IndexType;

    fn dim(&self) -> usize {
        self.local_grid.topology().dim()
    }
    fn index_map(&self) -> &[Self::IndexType] {
        self.local_grid.topology().index_map()
    }
    fn entity_count(&self, etype: ReferenceCellType) -> usize {
        self.local_grid.topology().entity_count(etype)
    }
    fn entity_count_by_dim(&self, dim: usize) -> usize {
        self.local_grid.topology().entity_count_by_dim(dim)
    }
    fn cell(&self, index: Self::IndexType) -> Option<&[usize]> {
        self.local_grid.topology().cell(index)
    }
    fn cell_type(&self, index: Self::IndexType) -> Option<ReferenceCellType> {
        self.local_grid.topology().cell_type(index)
    }
    fn entity_types(&self, dim: usize) -> &[ReferenceCellType] {
        self.local_grid.topology().entity_types(dim)
    }
    fn cell_to_entities(&self, index: Self::IndexType, dim: usize) -> Option<&[usize]> {
        self.local_grid.topology().cell_to_entities(index, dim)
    }
    fn entity_to_cells(
        &self,
        dim: usize,
        index: usize,
    ) -> Option<&[CellLocalIndexPair<Self::IndexType>]> {
        self.local_grid.topology().entity_to_cells(dim, index)
    }
    fn entity_to_flat_cells(
        &self,
        dim: usize,
        index: usize,
    ) -> Option<&[CellLocalIndexPair<usize>]> {
        self.local_grid.topology().entity_to_flat_cells(dim, index)
    }
    fn entity_vertices(&self, dim: usize, index: usize) -> Option<&[usize]> {
        self.local_grid.topology().entity_vertices(dim, index)
    }
    fn cell_ownership(&self, index: Self::IndexType) -> Ownership {
        self.local_grid.topology().cell_ownership(index)
    }
    fn vertex_ownership(&self, index: usize) -> Ownership {
        self.local_grid.topology().vertex_ownership(index)
    }
    fn edge_ownership(&self, index: usize) -> Ownership {
        self.local_grid.topology().edge_ownership(index)
    }
    fn vertex_index_to_id(&self, index: usize) -> usize {
        self.local_grid.topology().vertex_index_to_id(index)
    }
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize {
        self.local_grid.topology().cell_index_to_id(index)
    }
    fn vertex_id_to_index(&self, id: usize) -> usize {
        self.local_grid.topology().vertex_id_to_index(id)
    }
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType {
        self.local_grid.topology().cell_id_to_index(id)
    }
    fn face_index_to_flat_index(&self, index: Self::IndexType) -> usize {
        self.local_grid.topology().face_index_to_flat_index(index)
    }
    fn face_flat_index_to_index(&self, index: usize) -> Self::IndexType {
        self.local_grid.topology().face_flat_index_to_index(index)
    }
    fn cell_types(&self) -> &[ReferenceCellType] {
        self.local_grid.topology().cell_types()
    }
}

impl<'a, C: Communicator, G: Grid> Grid for ParallelGrid<'a, C, G> {
    type T = G::T;
    type Topology = Self;
    type Geometry = <G as Grid>::Geometry;

    fn mpi_rank(&self) -> usize {
        self.comm.rank() as usize
    }

    fn topology(&self) -> &Self::Topology {
        self
    }

    fn geometry(&self) -> &G::Geometry {
        self.local_grid.geometry()
    }

    fn is_serial(&self) -> bool {
        false
    }
}
