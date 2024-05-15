//! A parallel implementation of a grid
use crate::element::reference_cell;
use crate::grid::traits::{Grid, Topology};
use crate::traits::grid::{Builder, GridType, ParallelBuilder};
use crate::traits::types::{CellLocalIndexPair, Ownership, ReferenceCellType};
use mpi::{
    request::{LocalScope, WaitGuard},
    topology::{Communicator, Process},
    traits::{Buffer, Destination, Equivalence, Source},
};
use rlst::{rlst_dynamic_array2, Array, BaseArray, RandomAccessMut, VectorContainer};
use std::collections::HashMap;

type RlstMat<T> = Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>;

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
    fn edge_index_to_id(&self, index: usize) -> usize {
        self.serial_grid.topology().edge_index_to_id(index)
    }
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize {
        self.serial_grid.topology().cell_index_to_id(index)
    }
    fn vertex_id_to_index(&self, id: usize) -> usize {
        self.serial_grid.topology().vertex_id_to_index(id)
    }
    fn edge_id_to_index(&self, id: usize) -> usize {
        self.serial_grid.topology().edge_id_to_index(id)
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
        vertex_owners: HashMap<usize, usize>,
        edge_owners: HashMap<usize, usize>,
        cell_owners: HashMap<usize, usize>,
    ) -> Self {
        let rank = comm.rank() as usize;
        let size = comm.size() as usize;

        // Create cell ownership
        let mut cell_ownership = HashMap::new();
        let mut cells_to_query = vec![vec![]; size];

        for (id, owner) in cell_owners.iter() {
            if *owner != rank {
                cells_to_query[*owner].push(*id);
            }
        }
        mpi::request::scope(|scope| {
            for (p, cq) in cells_to_query.iter().enumerate() {
                if p != rank {
                    let _ =
                        WaitGuard::from(comm.process_at_rank(p as i32).immediate_send(scope, cq));
                }
            }
        });
        for p in 0..size {
            if p != rank {
                let (cells_queried, _status) =
                    comm.process_at_rank(p as i32).receive_vec::<usize>();
                let send_back =
                    cells_queried
                        .iter()
                        .map(|id| {
                            serial_grid.topology().face_index_to_flat_index(
                                Topology::cell_id_to_index(serial_grid.topology(), *id),
                            )
                        })
                        .collect::<Vec<_>>();
                mpi::request::scope(|scope| {
                    let _ = WaitGuard::from(
                        comm.process_at_rank(p as i32)
                            .immediate_send(scope, &send_back),
                    );
                });
            }
        }
        let mut cell_info = vec![vec![]; size];
        for (p, ci) in cell_info.iter_mut().enumerate() {
            if p != rank {
                (*ci, _) = comm.process_at_rank(p as i32).receive_vec::<usize>();
            }
        }

        let mut indices = vec![0; size];
        for (id, owner) in cell_owners.iter() {
            cell_ownership.insert(
                Topology::cell_id_to_index(serial_grid.topology(), *id),
                if *owner == rank {
                    Ownership::Owned
                } else {
                    indices[*owner] += 1;
                    Ownership::Ghost(*owner, cell_info[*owner][indices[*owner] - 1])
                },
            );
        }

        // Create vertex ownership
        let mut vertex_ownership = HashMap::new();
        let mut vertices_to_query = vec![vec![]; size];

        for (id, owner) in vertex_owners.iter() {
            if *owner != rank {
                vertices_to_query[*owner].push(*id);
            }
        }
        mpi::request::scope(|scope| {
            for (p, vq) in vertices_to_query.iter().enumerate() {
                if p != rank {
                    let _ =
                        WaitGuard::from(comm.process_at_rank(p as i32).immediate_send(scope, vq));
                }
            }
        });
        for p in 0..size {
            if p != rank {
                let (vertices_queried, _status) =
                    comm.process_at_rank(p as i32).receive_vec::<usize>();
                let send_back = vertices_queried
                    .iter()
                    .map(|id| Topology::vertex_id_to_index(serial_grid.topology(), *id))
                    .collect::<Vec<_>>();
                mpi::request::scope(|scope| {
                    let _ = WaitGuard::from(
                        comm.process_at_rank(p as i32)
                            .immediate_send(scope, &send_back),
                    );
                });
            }
        }
        let mut vertex_info = vec![vec![]; size];
        for (p, vi) in vertex_info.iter_mut().enumerate() {
            if p != rank {
                (*vi, _) = comm.process_at_rank(p as i32).receive_vec::<usize>();
            }
        }

        let mut indices = vec![0; size];
        for (id, owner) in vertex_owners.iter() {
            vertex_ownership.insert(
                Topology::vertex_id_to_index(serial_grid.topology(), *id),
                if *owner == rank {
                    Ownership::Owned
                } else {
                    indices[*owner] += 1;
                    Ownership::Ghost(*owner, vertex_info[*owner][indices[*owner] - 1])
                },
            );
        }

        // Create edge ownership
        let mut edge_ownership = HashMap::new();
        let mut edges_to_query = vec![vec![]; size];

        for (id, owner) in edge_owners.iter() {
            if *owner != rank {
                edges_to_query[*owner].push(*id);
            }
        }
        mpi::request::scope(|scope| {
            for (p, eq) in edges_to_query.iter().enumerate() {
                if p != rank {
                    let _ =
                        WaitGuard::from(comm.process_at_rank(p as i32).immediate_send(scope, eq));
                }
            }
        });
        for p in 0..size {
            if p != rank {
                let (edges_queried, _status) =
                    comm.process_at_rank(p as i32).receive_vec::<usize>();
                let send_back = edges_queried
                    .iter()
                    .map(|id| Topology::edge_id_to_index(serial_grid.topology(), *id))
                    .collect::<Vec<_>>();
                mpi::request::scope(|scope| {
                    let _ = WaitGuard::from(
                        comm.process_at_rank(p as i32)
                            .immediate_send(scope, &send_back),
                    );
                });
            }
        }
        let mut edge_info = vec![vec![]; size];
        for (p, ei) in edge_info.iter_mut().enumerate() {
            if p != rank {
                (*ei, _) = comm.process_at_rank(p as i32).receive_vec::<usize>();
            }
        }

        let mut indices = vec![0; size];
        for (id, owner) in edge_owners.iter() {
            edge_ownership.insert(
                Topology::edge_id_to_index(serial_grid.topology(), *id),
                if *owner == rank {
                    Ownership::Owned
                } else {
                    indices[*owner] += 1;
                    Ownership::Ghost(*owner, edge_info[*owner][indices[*owner] - 1])
                },
            );
        }

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
    fn edge_index_to_id(&self, index: usize) -> usize {
        self.local_grid.topology().edge_index_to_id(index)
    }
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize {
        self.local_grid.topology().cell_index_to_id(index)
    }
    fn vertex_id_to_index(&self, id: usize) -> usize {
        self.local_grid.topology().vertex_id_to_index(id)
    }
    fn edge_id_to_index(&self, id: usize) -> usize {
        self.local_grid.topology().edge_id_to_index(id)
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
        Topology::cell_types(self.local_grid.topology())
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

// TODO: pub(crate)
/// Internal trait for building parallel grids
pub trait ParallelGridBuilder {
    /// The serial grid type used on each process
    type G: Grid;

    /// Extra cell info
    type ExtraCellInfo: Clone;

    /// TODO
    fn push_extra_cell_info(&self, _extra_cell_info: &mut Self::ExtraCellInfo, _cell_id: usize) {}
    /// TODO
    fn send_extra_cell_info(
        &self,
        _scope: &LocalScope,
        _process: &Process,
        _extra_cell_info: &Self::ExtraCellInfo,
    ) {
    }
    /// TODO
    fn receive_extra_cell_info(
        &self,
        _process: &Process,
        _extra_cell_info: &mut Self::ExtraCellInfo,
    ) {
    }
    /// TODO
    fn new_extra_cell_info(&self) -> Self::ExtraCellInfo;

    /// The id of each point
    fn point_indices_to_ids(&self) -> &[usize];

    /// The coordinates of each point
    fn points(&self) -> &[<<Self as ParallelGridBuilder>::G as GridType>::T];

    /// The id of each cell
    fn cell_indices_to_ids(&self) -> &[usize];

    /// The point of a cell
    fn cell_points(&self, index: usize) -> &[usize];

    /// The vertices of a cell
    fn cell_vertices(&self, index: usize) -> &[usize];

    /// The cell type of a cell
    fn cell_type(&self, index: usize) -> ReferenceCellType;

    #[allow(clippy::too_many_arguments)]
    /// Create a serial grid on one process
    fn create_serial_grid(
        &self,
        points: RlstMat<<<Self as ParallelGridBuilder>::G as GridType>::T>,
        cells: &[usize],
        point_indices_to_ids: Vec<usize>,
        point_ids_to_indices: HashMap<usize, usize>,
        cell_indices_to_ids: Vec<usize>,
        cell_ids_to_indices: HashMap<usize, usize>,
        edge_ids: HashMap<[usize; 2], usize>,
        extra_cell_info: &Self::ExtraCellInfo,
    ) -> Self::G;

    #[allow(clippy::too_many_arguments)]
    /// Internal function to create a parallel grid
    fn create_internal<'a, C: Communicator>(
        &self,
        comm: &'a C,
        points: &[<<Self as ParallelGridBuilder>::G as GridType>::T],
        point_ids: &[usize],
        cells: &[usize],
        cell_owners: &[usize],
        cell_ids: &[usize],
        vertex_owners: &[usize],
        vertex_ids: &[usize],
        edges: &[usize],
        edge_owners: &[usize],
        edge_ids: &[usize],
        extra_cell_info: &Self::ExtraCellInfo,
    ) -> ParallelGrid<'a, C, Self::G> {
        let npts = point_ids.len();

        let mut coordinates =
            rlst_dynamic_array2!(<<Self as ParallelGridBuilder>::G as GridType>::T, [npts, 3]);
        for i in 0..npts {
            for j in 0..3 {
                *coordinates.get_mut([i, j]).unwrap() = points[i * 3 + j];
            }
        }

        let mut point_ids_to_indices = HashMap::new();
        for (index, id) in point_ids.iter().enumerate() {
            point_ids_to_indices.insert(*id, index);
        }
        let mut cell_ids_to_indices = HashMap::new();
        for (index, id) in cell_ids.iter().enumerate() {
            cell_ids_to_indices.insert(*id, index);
        }

        let mut edge_id_map = HashMap::new();
        for (n, id) in edge_ids.iter().enumerate() {
            edge_id_map.insert([edges[2 * n], edges[2 * n + 1]], *id);
        }

        let serial_grid = self.create_serial_grid(
            coordinates,
            cells,
            point_ids.to_vec(),
            point_ids_to_indices,
            cell_ids.to_vec(),
            cell_ids_to_indices,
            edge_id_map,
            extra_cell_info,
        );

        let mut vertex_owner_map = HashMap::new();
        for (id, owner) in vertex_ids.iter().zip(vertex_owners) {
            vertex_owner_map.insert(*id, *owner);
        }
        let mut edge_owner_map = HashMap::new();
        for (id, owner) in edge_ids.iter().zip(edge_owners) {
            edge_owner_map.insert(*id, *owner);
        }
        let mut cell_owner_map = HashMap::new();
        for (id, owner) in cell_ids.iter().zip(cell_owners) {
            cell_owner_map.insert(*id, *owner);
        }

        ParallelGrid::new(
            comm,
            serial_grid,
            vertex_owner_map,
            edge_owner_map,
            cell_owner_map,
        )
    }
}

impl<const GDIM: usize, B: ParallelGridBuilder + Builder<GDIM>> ParallelBuilder<GDIM> for B
where
    Vec<<<B as ParallelGridBuilder>::G as GridType>::T>: Buffer,
    <<B as ParallelGridBuilder>::G as GridType>::T: Equivalence,
{
    type ParallelGridType<'a, C: Communicator + 'a> = ParallelGrid<'a, C, B::G>;

    fn create_parallel_grid<'a, C: Communicator>(
        self,
        comm: &'a C,
        cell_owners: &HashMap<usize, usize>,
    ) -> Self::ParallelGridType<'a, C> {
        let rank = comm.rank() as usize;
        let size = comm.size() as usize;

        let npts = self.point_indices_to_ids().len();
        let ncells = self.cell_indices_to_ids().len();

        // data used in computation
        let mut vertex_owners = vec![(-1, 0); npts];
        let mut vertex_counts = vec![0; size];
        let mut cell_indices_per_proc = vec![vec![]; size];
        let mut vertex_indices_per_proc = vec![vec![]; size];
        let mut edge_owners = HashMap::new();
        let mut edge_ids = HashMap::new();
        let mut edge_counts = vec![0; size];
        let mut edges_included_per_proc = vec![vec![]; size];
        let mut edge_id = 0;

        // data to send to other processes
        let mut points_per_proc = vec![vec![]; size];
        let mut point_ids_per_proc = vec![vec![]; size];
        let mut cells_per_proc = vec![vec![]; size];
        let mut cell_owners_per_proc = vec![vec![]; size];
        let mut cell_ids_per_proc = vec![vec![]; size];
        let mut vertex_owners_per_proc = vec![vec![]; size];
        let mut edges_per_proc = vec![vec![]; size];
        let mut edge_owners_per_proc = vec![vec![]; size];
        let mut edge_ids_per_proc = vec![vec![]; size];
        let mut extra_cell_info_per_proc = vec![self.new_extra_cell_info(); size];

        for (index, id) in self.cell_indices_to_ids().iter().enumerate() {
            let owner = cell_owners[id];
            // TODO: only assign owners to the first 3 or 4 vertices
            for v in self.cell_points(index) {
                if vertex_owners[*v].0 == -1 {
                    vertex_owners[*v] = (owner as i32, vertex_counts[owner]);
                }
                if !vertex_indices_per_proc[owner].contains(v) {
                    vertex_indices_per_proc[owner].push(*v);
                    vertex_owners_per_proc[owner].push(vertex_owners[*v].0 as usize);
                    for i in 0..GDIM {
                        points_per_proc[owner].push(self.points()[v * GDIM + i])
                    }
                    point_ids_per_proc[owner].push(self.point_indices_to_ids()[*v]);
                    vertex_counts[owner] += 1;
                }
            }
        }

        for (index, id) in self.cell_indices_to_ids().iter().enumerate() {
            let ref_conn = &reference_cell::connectivity(self.cell_type(index))[1];
            let owner = cell_owners[id];
            for e in ref_conn {
                let cell = self.cell_vertices(index);
                let mut v0 = cell[e[0][0]];
                let mut v1 = cell[e[0][1]];
                if v0 > v1 {
                    std::mem::swap(&mut v0, &mut v1);
                }
                if edge_owners.get_mut(&(v0, v1)).is_none() {
                    edge_owners.insert((v0, v1), (owner, edge_counts[owner]));
                    edge_ids.insert((v0, v1), edge_id);
                    edge_id += 1;
                    edges_included_per_proc[owner].push((v0, v1));
                    edges_per_proc[owner].push(v0);
                    edges_per_proc[owner].push(v1);
                    edge_owners_per_proc[owner].push(edge_owners[&(v0, v1)].0);
                    edge_ids_per_proc[owner].push(edge_ids[&(v0, v1)]);
                    edge_counts[owner] += 1;
                }
            }
        }

        for index in 0..ncells {
            for p in 0..size {
                for v in self.cell_points(index) {
                    if vertex_indices_per_proc[p].contains(v) {
                        cell_indices_per_proc[p].push(index);
                        break;
                    }
                }
            }
        }

        for p in 0..size {
            for index in &cell_indices_per_proc[p] {
                let id = self.cell_indices_to_ids()[*index];
                // TODO: only assign owners to the first 3 or 4 vertices
                for v in self.cell_points(*index) {
                    if !vertex_indices_per_proc[p].contains(v) {
                        vertex_indices_per_proc[p].push(*v);
                        vertex_owners_per_proc[p].push(vertex_owners[*v].0 as usize);
                        for i in 0..GDIM {
                            points_per_proc[p].push(self.points()[v * GDIM + i]);
                        }
                        point_ids_per_proc[p].push(self.point_indices_to_ids()[*v]);
                    }
                    cells_per_proc[p].push(
                        vertex_indices_per_proc[p]
                            .iter()
                            .position(|&r| r == *v)
                            .unwrap(),
                    );
                }
                let ref_conn = &reference_cell::connectivity(self.cell_type(*index))[1];

                for e in ref_conn {
                    let cell = self.cell_vertices(*index);
                    let mut v0 = cell[e[0][0]];
                    let mut v1 = cell[e[0][1]];
                    if v0 > v1 {
                        std::mem::swap(&mut v0, &mut v1);
                    }
                    if !edges_included_per_proc[p].contains(&(v0, v1)) {
                        edges_included_per_proc[p].push((v0, v1));
                        edges_per_proc[p].push(v0);
                        edges_per_proc[p].push(v1);
                        edge_owners_per_proc[p].push(edge_owners[&(v0, v1)].0);
                        edge_ids_per_proc[p].push(edge_ids[&(v0, v1)]);
                    }
                }

                cell_ids_per_proc[p].push(id);
                cell_owners_per_proc[p].push(cell_owners[&id]);
                self.push_extra_cell_info(&mut extra_cell_info_per_proc[p], id);
            }
        }

        mpi::request::scope(|scope| {
            for p in 1..size {
                let process = comm.process_at_rank(p as i32);
                let _ = WaitGuard::from(process.immediate_send(scope, &points_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &point_ids_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &cells_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &cell_owners_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &cell_ids_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &vertex_owners_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &edges_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &edge_owners_per_proc[p]));
                let _ = WaitGuard::from(process.immediate_send(scope, &edge_ids_per_proc[p]));
                self.send_extra_cell_info(scope, &process, &extra_cell_info_per_proc[rank]);
            }
        });

        self.create_internal(
            comm,
            &points_per_proc[rank],
            &point_ids_per_proc[rank],
            &cells_per_proc[rank],
            &cell_owners_per_proc[rank],
            &cell_ids_per_proc[rank],
            &vertex_owners_per_proc[rank],
            &point_ids_per_proc[rank],
            &edges_per_proc[rank],
            &edge_owners_per_proc[rank],
            &edge_ids_per_proc[rank],
            &extra_cell_info_per_proc[rank],
        )
    }

    fn receive_parallel_grid<C: Communicator>(
        self,
        comm: &C,
        root_rank: usize,
    ) -> ParallelGrid<'_, C, B::G> {
        let root_process = comm.process_at_rank(root_rank as i32);

        let (points, _status) =
            root_process.receive_vec::<<<Self as ParallelGridBuilder>::G as GridType>::T>();
        let (point_ids, _status) = root_process.receive_vec::<usize>();
        let (cells, _status) = root_process.receive_vec::<usize>();
        let (cell_owners, _status) = root_process.receive_vec::<usize>();
        let (cell_ids, _status) = root_process.receive_vec::<usize>();
        let (vertex_owners, _status) = root_process.receive_vec::<usize>();
        let (edges, _status) = root_process.receive_vec::<usize>();
        let (edge_owners, _status) = root_process.receive_vec::<usize>();
        let (edge_ids, _status) = root_process.receive_vec::<usize>();
        let mut extra_cell_info = self.new_extra_cell_info();
        self.receive_extra_cell_info(&root_process, &mut extra_cell_info);

        self.create_internal(
            comm,
            &points,
            &point_ids,
            &cells,
            &cell_owners,
            &cell_ids,
            &vertex_owners,
            &point_ids,
            &edges,
            &edge_owners,
            &edge_ids,
            &extra_cell_info,
        )
    }
}
