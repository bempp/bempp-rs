//! A parallel implementation of a grid
//use crate::grid::grid::{SerialGeometry, SerialTopology};
//use crate::element::ciarlet::CiarletElement;
//use crate::tools::arrays::AdjacencyList;
//use crate::traits::arrays::AdjacencyListAccess;
//use crate::traits::cell::ReferenceCellType;
//use crate::traits::element::FiniteElement;
//use crate::traits::grid::{Geometry, GeometryEvaluator, Grid, Ownership, Topology};
use mpi::{request::WaitGuard, topology::Communicator, traits::*};
use rlst::{
    Array,
    BaseArray,
    VectorContainer,
    rlst_dynamic_array2,
    RandomAccessByRef, RandomAccessMut, Shape, RlstScalar
};
use num::Float;
use crate::grid::flat_triangle_grid::{SerialFlatTriangleGridBuilder, SerialFlatTriangleGrid};
use crate::grid::traits_impl::WrappedGrid;
use std::collections::HashMap;
use crate::grid::traits::Geometry;

// TODO: update this

impl<T: Float + RlstScalar<Real=T>> SerialFlatTriangleGridBuilder<T> {
    fn create_parallel_grid<C: Communicator>(
        self, comm: &C, cell_owners: HashMap<usize, usize>,
    ) -> WrappedGrid<SerialFlatTriangleGrid<T>> {
        panic!();
    }
    fn create_parallel_grid_receive<C: Communicator>(
        self, comm: &C,
    ) -> WrappedGrid<SerialFlatTriangleGrid<T>> {
        panic!();
    }
}


/// Geometry of a parallel grid
pub struct ParallelGeometry<'comm, C: Communicator, G: Geometry> {
    comm: &'comm C,
    serial_geometry: G,
}
/*
impl<'a, C: Communicator> ParallelGeometry<'a, C> {
    pub fn new(
        comm: &'a C,
        coordinates: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
        cells: &AdjacencyList<usize>,
        cell_types: &Vec<ReferenceCellType>,
    ) -> Self {
        Self {
            comm: comm,
            serial_geometry: SerialGeometry::new(coordinates, cells, cell_types),
        }
    }

    pub fn coordinate_elements(&self) -> &Vec<CiarletElement<f64>> {
        self.serial_geometry.coordinate_elements()
    }

    pub fn element_changes(&self) -> &Vec<usize> {
        self.serial_geometry.element_changes()
    }

    pub fn comm(&self) -> &'a C {
        self.comm
    }
}
*/

impl<'comm, C: Communicator, G: Geometry> Geometry for ParallelGeometry<'comm, C, G> {
    type IndexType = G::IndexType;
    type T = G::T;
    type Element = G::Element;
    type Evaluator<'a> = G::Evaluator<'a> where Self: 'a;

    fn dim(&self) -> usize {
        self.serial_geometry.dim()
    }

    fn index_map(&self) -> &[Self::IndexType] {
        self.serial_geometry.index_map()
    }
    fn coordinate(&self, point_index: usize, coord_index: usize) -> Option<&Self::T> {
        self.serial_geometry.coordinate(point_index, coord_index)
    }
    fn point_count(&self) -> usize {
        self.serial_geometry.point_count()
    }
    fn cell_points(&self, index: Self::IndexType) -> Option<&[usize]> {
        self.serial_geometry.cell_points(index)
    }
    fn cell_count(&self) -> usize {
        self.serial_geometry.cell_count()
    }
    fn cell_element(&self, index: Self::IndexType) -> Option<&Self::Element> {
        self.serial_geometry.cell_element(index)
    }
    fn element_count(&self) -> usize {
        self.serial_geometry.element_count()
    }
    fn element(&self, i: usize) -> Option<&Self::Element> {
        self.serial_geometry.element(i)
    }
    fn cell_indices(&self, i: usize) -> Option<&[Self::IndexType]> {
        self.serial_geometry.cell_indices(i)
    }
    fn midpoint(&self, index: Self::IndexType, point: &mut [Self::T]) {
        self.serial_geometry.midpoint(index, point)
    }
    fn diameter(&self, index: Self::IndexType) -> Self::T {
        self.serial_geometry.diameter(index)
    }
    fn volume(&self, index: Self::IndexType) -> Self::T {
        self.serial_geometry.volume(index)
    }
    fn get_evaluator<'a>(&'a self, points: &'a [Self::T]) -> Self::Evaluator<'a>{
        self.serial_geometry.get_evaluator(points)
    }
    fn point_index_to_id(&self, index: usize) -> usize {
        self.serial_geometry.point_index_to_id(index)
    }
    fn cell_index_to_id(&self, index: Self::IndexType) -> usize {
        self.serial_geometry.cell_index_to_id(index)
    }
    fn point_id_to_index(&self, id: usize) -> usize {
        self.serial_geometry.point_id_to_index(id)
    }
    fn cell_id_to_index(&self, id: usize) -> Self::IndexType {
        self.serial_geometry.cell_id_to_index(id)
    }
}

/*
/// Topology of a parallel grid
pub struct ParallelTopology<'a, C: Communicator> {
    comm: &'a C,
    serial_topology: SerialTopology,
    ownership: Vec<Vec<Ownership>>,
}

impl<'a, C: Communicator> ParallelTopology<'a, C> {
    pub fn new(
        comm: &'a C,
        cells: &AdjacencyList<usize>,
        cell_types: &Vec<ReferenceCellType>,
        cell_ownership: Vec<Ownership>,
        vertex_ownership: Vec<Ownership>,
    ) -> Self {
        let t = SerialTopology::new(cells, cell_types);
        let mut ownership = vec![vec![]; t.dim() + 1];
        ownership[0] = vertex_ownership;
        ownership[t.dim()] = cell_ownership;
        Self {
            comm: comm,
            serial_topology: t,
            ownership: ownership,
        }
    }

    pub fn comm(&self) -> &'a C {
        self.comm
    }
}

impl<'a, C: Communicator> Topology<'a> for ParallelTopology<'a, C> {
    type Connectivity = AdjacencyList<usize>;

    fn index_map(&self) -> &[usize] {
        self.serial_topology.index_map()
    }
    fn dim(&self) -> usize {
        self.serial_topology.dim()
    }
    fn entity_count(&self, dim: usize) -> usize {
        self.serial_topology.entity_count(dim)
    }
    fn cell(&self, index: usize) -> Option<&[usize]> {
        self.serial_topology.cell(index)
    }
    fn cell_type(&self, index: usize) -> Option<ReferenceCellType> {
        self.serial_topology.cell_type(index)
    }

    fn connectivity(&self, dim0: usize, dim1: usize) -> &Self::Connectivity {
        self.serial_topology.connectivity(dim0, dim1)
    }

    fn entity_ownership(&self, dim: usize, index: usize) -> Ownership {
        if dim != 0 && dim != self.dim() {
            panic!("Entity ownership for these entities not implemented yet");
        }
        self.ownership[dim][index]
    }
}

/// Parallel grid
pub struct ParallelGrid<'a, C: Communicator> {
    comm: &'a C,
    topology: ParallelTopology<'a, C>,
    geometry: ParallelGeometry<'a, C>,
}

impl<'a, C: Communicator> ParallelGrid<'a, C> {
    pub fn new(
        comm: &'a C,
        coordinates: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2>,
        cells: AdjacencyList<usize>,
        cell_types: Vec<ReferenceCellType>,
        cell_owners: Vec<usize>,
    ) -> Self {
        let rank = comm.rank() as usize;
        let size = comm.size() as usize;

        // data used in computation
        let mut vertex_owners = vec![(-1, 0); coordinates.shape()[0]];
        let mut vertex_counts = vec![0; size];
        let mut cell_indices_per_proc = vec![vec![]; size];

        // data to send to other processes
        let mut cells_per_proc = vec![vec![]; size];
        let mut cell_types_per_proc = vec![vec![]; size];
        let mut cell_sizes_per_proc = vec![vec![]; size];
        let mut cell_local_indices_per_proc = vec![vec![]; size];
        let mut cell_owners_per_proc = vec![vec![]; size];
        let mut coordinates_per_proc = vec![vec![]; size];
        let mut vertex_indices_per_proc = vec![vec![]; size];
        let mut vertex_local_indices_per_proc = vec![vec![]; size];
        let mut vertex_owners_per_proc = vec![vec![]; size];

        for (c, o) in cells.iter_rows().zip(cell_owners.iter()) {
            let p = *o as usize;
            for v in c {
                if vertex_owners[*v].0 == -1 {
                    vertex_owners[*v] = (*o as i32, vertex_counts[p]);
                    vertex_counts[p] += 1;
                }
                if !vertex_indices_per_proc[p].contains(v) {
                    vertex_indices_per_proc[p].push(*v);
                    vertex_owners_per_proc[p].push(vertex_owners[*v].0 as usize);
                    vertex_local_indices_per_proc[p].push(vertex_owners[*v].1);
                    for i in 0..coordinates.shape()[1] {
                        coordinates_per_proc[p].push(*coordinates.get([*v, i]).unwrap())
                    }
                }
            }
        }
        for (i, c) in cells.iter_rows().enumerate() {
            for p in 0..size {
                for v in c {
                    if vertex_indices_per_proc[p].contains(v) {
                        cell_indices_per_proc[p].push(i);
                        break;
                    }
                }
            }
        }
        for p in 0..size {
            for c in &cell_indices_per_proc[p] {
                for v in cells.row(*c).unwrap() {
                    if !vertex_indices_per_proc[p].contains(v) {
                        vertex_indices_per_proc[p].push(*v);
                        vertex_owners_per_proc[p].push(vertex_owners[*v].0 as usize);
                        vertex_local_indices_per_proc[p].push(vertex_owners[*v].1);
                        for i in 0..coordinates.shape()[1] {
                            coordinates_per_proc[p].push(*coordinates.get([*v, i]).unwrap())
                        }
                    }
                    cells_per_proc[p].push(
                        vertex_indices_per_proc[p]
                            .iter()
                            .position(|&r| r == *v)
                            .unwrap(),
                    );
                }
                cell_sizes_per_proc[p].push(cells.row(*c).unwrap().len());
                cell_types_per_proc[p].push(cell_types[*c] as u8);
                cell_owners_per_proc[p].push(cell_owners[*c] as usize);
                cell_local_indices_per_proc[p].push(
                    cell_indices_per_proc[cell_owners[*c]]
                        .iter()
                        .position(|&r| r == *c)
                        .unwrap(),
                );
            }
        }

        mpi::request::scope(|scope| {
            for p in 1..size {
                let _sreq2 = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cells_per_proc[p][..]),
                );
                let _sreq = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cell_types_per_proc[p][..]),
                );
                let _sreq = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cell_sizes_per_proc[p][..]),
                );
                let _sreq = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cell_local_indices_per_proc[p][..]),
                );
                let _sreq = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cell_owners_per_proc[p][..]),
                );
                let _sreq3 = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &coordinates_per_proc[p][..]),
                );
                let _sreq3 = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &vertex_indices_per_proc[p][..]),
                );
                let _sreq3 = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &vertex_local_indices_per_proc[p][..]),
                );
                let _sreq3 = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &vertex_owners_per_proc[p][..]),
                );
            }
        });

        Self::new_internal(
            comm,
            &cells_per_proc[rank],
            &cell_types_per_proc[rank],
            &cell_sizes_per_proc[rank],
            &cell_local_indices_per_proc[rank],
            &cell_owners_per_proc[rank],
            &coordinates_per_proc[rank],
            &vertex_indices_per_proc[rank],
            &vertex_local_indices_per_proc[rank],
            &vertex_owners_per_proc[rank],
        )
    }

    pub fn new_subprocess(comm: &'a C, root_rank: usize) -> Self {
        let root_process = comm.process_at_rank(root_rank as i32);
        let (cells, _status) = root_process.receive_vec::<usize>();
        let (cell_types, _status) = root_process.receive_vec::<u8>();
        let (cell_sizes, _status) = root_process.receive_vec::<usize>();
        let (cell_local_indices, _status) = root_process.receive_vec::<usize>();
        let (cell_owners, _status) = root_process.receive_vec::<usize>();
        let (coordinates, _status) = root_process.receive_vec::<f64>();
        let (vertex_indices, _status) = root_process.receive_vec::<usize>();
        let (vertex_local_indices, _status) = root_process.receive_vec::<usize>();
        let (vertex_owners, _status) = root_process.receive_vec::<usize>();
        Self::new_internal(
            comm,
            &cells,
            &cell_types,
            &cell_sizes,
            &cell_local_indices,
            &cell_owners,
            &coordinates,
            &vertex_indices,
            &vertex_local_indices,
            &vertex_owners,
        )
    }

    fn new_internal(
        comm: &'a C,
        flat_cells: &Vec<usize>,
        cell_types_u8: &Vec<u8>,
        cell_sizes: &Vec<usize>,
        cell_local_indices: &Vec<usize>,
        cell_owners: &Vec<usize>,
        flat_coordinates: &Vec<f64>,
        vertex_indices: &Vec<usize>,
        vertex_local_indices: &Vec<usize>,
        vertex_owners: &Vec<usize>,
    ) -> Self {
        let rank = comm.rank() as usize;
        let gdim = flat_coordinates.len() / vertex_indices.len();

        let mut cells = AdjacencyList::<usize>::new();
        let mut i = 0;
        for s in cell_sizes {
            let mut c = vec![0; *s];
            for a in 0..*s {
                c[a] = flat_cells[i + a];
            }
            cells.add_row(&c);
            i += s;
        }
        let mut cell_types = vec![];
        for c in cell_types_u8 {
            cell_types.push(if *c == ReferenceCellType::Point as u8 {
                ReferenceCellType::Point
            } else if *c == ReferenceCellType::Interval as u8 {
                ReferenceCellType::Interval
            } else if *c == ReferenceCellType::Triangle as u8 {
                ReferenceCellType::Triangle
            } else if *c == ReferenceCellType::Quadrilateral as u8 {
                ReferenceCellType::Quadrilateral
            } else {
                panic!("Unsupported cell type");
            });
        }
        let mut coordinates = rlst_dynamic_array2!(f64, [vertex_indices.len(), gdim]);
        for i in 0..vertex_indices.len() {
            for j in 0..gdim {
                *coordinates.get_mut([i, j]).unwrap() = flat_coordinates[i * gdim + j];
            }
        }

        let mut cell_ownership = vec![];
        for (o, i) in cell_owners.iter().zip(cell_local_indices.iter()) {
            cell_ownership.push(if *o == rank {
                Ownership::Owned
            } else {
                Ownership::Ghost(*o, *i)
            });
        }
        let mut vertex_ownership = vec![];
        for (o, i) in vertex_owners.iter().zip(vertex_local_indices.iter()) {
            vertex_ownership.push(if *o == rank {
                Ownership::Owned
            } else {
                Ownership::Ghost(*o, *i)
            });
        }

        Self {
            comm,
            topology: ParallelTopology::new(
                comm,
                &cells,
                &cell_types,
                cell_ownership,
                vertex_ownership,
            ),
            geometry: ParallelGeometry::new(comm, coordinates, &cells, &cell_types),
        }
    }

    pub fn comm(&self) -> &'a C {
        self.comm
    }
}
impl<'a, C: Communicator> Grid<'a> for ParallelGrid<'a, C> {
    type Topology = ParallelTopology<'a, C>;
    type Geometry = ParallelGeometry<'a, C>;

    fn topology(&self) -> &Self::Topology {
        &self.topology
    }

    fn geometry(&self) -> &Self::Geometry {
        &self.geometry
    }

    fn is_serial(&self) -> bool {
        false
    }
}
*/
