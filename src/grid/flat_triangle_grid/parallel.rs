//! Parallel grid builder

use crate::element::reference_cell;
use crate::grid::flat_triangle_grid::{FlatTriangleGrid, FlatTriangleGridBuilder};
use crate::grid::parallel_grid::ParallelGrid;
use crate::traits::grid::ParallelBuilder;
use crate::traits::types::ReferenceCellType;
use mpi::{
    request::WaitGuard,
    topology::Communicator,
    traits::{Buffer, Destination, Equivalence, Source},
};
use num::Float;
use rlst::{
    dense::array::views::ArrayViewMut, rlst_dynamic_array2, Array, BaseArray, MatrixInverse,
    RandomAccessMut, RlstScalar, VectorContainer,
};
use std::collections::HashMap;

impl<T: Float + RlstScalar<Real = T> + Equivalence> ParallelBuilder<3>
    for FlatTriangleGridBuilder<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
    [T]: Buffer,
{
    type ParallelGridType<'a, C: Communicator + 'a> = ParallelGrid<'a, C, FlatTriangleGrid<T>>;
    fn create_parallel_grid<'a, C: Communicator>(
        self,
        comm: &'a C,
        cell_owners: &HashMap<usize, usize>,
    ) -> ParallelGrid<'a, C, FlatTriangleGrid<T>> {
        let rank = comm.rank() as usize;
        let size = comm.size() as usize;

        let npts = self.point_indices_to_ids.len();
        let ncells = self.cell_indices_to_ids.len();

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

        for (index, id) in self.cell_indices_to_ids.iter().enumerate() {
            let owner = cell_owners[id];
            for v in &self.cells[3 * index..3 * (index + 1)] {
                if vertex_owners[*v].0 == -1 {
                    vertex_owners[*v] = (owner as i32, vertex_counts[owner]);
                }
                if !vertex_indices_per_proc[owner].contains(v) {
                    vertex_indices_per_proc[owner].push(*v);
                    vertex_owners_per_proc[owner].push(vertex_owners[*v].0 as usize);
                    for i in 0..3 {
                        points_per_proc[owner].push(self.points[v * 3 + i])
                    }
                    point_ids_per_proc[owner].push(self.point_indices_to_ids[*v]);
                    vertex_counts[owner] += 1;
                }
            }
        }

        let ref_conn = &reference_cell::connectivity(ReferenceCellType::Triangle)[1];

        for (index, id) in self.cell_indices_to_ids.iter().enumerate() {
            let owner = cell_owners[id];
            for e in ref_conn {
                let v0 = self.cells[3 * index + e[0][0]];
                let v1 = self.cells[3 * index + e[0][1]];
                let edge = if v0 < v1 { [v0, v1] } else { [v1, v0] };
                if edge_owners.get_mut(&edge).is_none() {
                    edge_owners.insert(edge, (owner, edge_counts[owner]));
                    edge_ids.insert(edge, edge_id);
                    edge_id += 1;
                    edges_included_per_proc[owner].push(edge);
                    edges_per_proc[owner].push(edge[0]);
                    edges_per_proc[owner].push(edge[1]);
                    edge_owners_per_proc[owner].push(edge_owners[&edge].0);
                    edge_ids_per_proc[owner].push(edge_ids[&edge]);
                    edge_counts[owner] += 1;
                }
            }
        }

        for index in 0..ncells {
            for p in 0..size {
                for v in &self.cells[3 * index..3 * (index + 1)] {
                    if vertex_indices_per_proc[p].contains(v) {
                        cell_indices_per_proc[p].push(index);
                        break;
                    }
                }
            }
        }

        for p in 0..size {
            for index in &cell_indices_per_proc[p] {
                let id = self.cell_indices_to_ids[*index];
                for v in &self.cells[3 * index..3 * (index + 1)] {
                    if !vertex_indices_per_proc[p].contains(v) {
                        vertex_indices_per_proc[p].push(*v);
                        vertex_owners_per_proc[p].push(vertex_owners[*v].0 as usize);
                        for i in 0..3 {
                            points_per_proc[p].push(self.points[v * 3 + i])
                        }
                        point_ids_per_proc[p].push(self.point_indices_to_ids[*v])
                    }
                    cells_per_proc[p].push(
                        vertex_indices_per_proc[p]
                            .iter()
                            .position(|&r| r == *v)
                            .unwrap(),
                    );
                }

                for e in ref_conn {
                    let v0 = self.cells[3 * index + e[0][0]];
                    let v1 = self.cells[3 * index + e[0][1]];
                    let edge = if v0 < v1 { [v0, v1] } else { [v1, v0] };
                    if !edges_included_per_proc[p].contains(&edge) {
                        edges_included_per_proc[p].push(edge);
                        edges_per_proc[p].push(edge[0]);
                        edges_per_proc[p].push(edge[1]);
                        edge_owners_per_proc[p].push(edge_owners[&edge].0);
                        edge_ids_per_proc[p].push(edge_ids[&edge]);
                    }
                }

                cell_ids_per_proc[p].push(id);
                cell_owners_per_proc[p].push(cell_owners[&id]);
            }
        }

        mpi::request::scope(|scope| {
            for p in 1..size {
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &points_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &point_ids_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cells_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cell_owners_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &cell_ids_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &vertex_owners_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &edges_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &edge_owners_per_proc[p]),
                );
                let _ = WaitGuard::from(
                    comm.process_at_rank(p as i32)
                        .immediate_send(scope, &edge_ids_per_proc[p]),
                );
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
        )
    }
    fn receive_parallel_grid<C: Communicator>(
        self,
        comm: &C,
        root_rank: usize,
    ) -> ParallelGrid<'_, C, FlatTriangleGrid<T>> {
        let root_process = comm.process_at_rank(root_rank as i32);

        let (points, _status) = root_process.receive_vec::<T>();
        let (point_ids, _status) = root_process.receive_vec::<usize>();
        let (cells, _status) = root_process.receive_vec::<usize>();
        let (cell_owners, _status) = root_process.receive_vec::<usize>();
        let (cell_ids, _status) = root_process.receive_vec::<usize>();
        let (vertex_owners, _status) = root_process.receive_vec::<usize>();
        let (edges, _status) = root_process.receive_vec::<usize>();
        let (edge_owners, _status) = root_process.receive_vec::<usize>();
        let (edge_ids, _status) = root_process.receive_vec::<usize>();
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
        )
    }
}

impl<T: Float + RlstScalar<Real = T>> FlatTriangleGridBuilder<T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
{
    #[allow(clippy::too_many_arguments)]
    fn create_internal<'a, C: Communicator>(
        self,
        comm: &'a C,
        points: &[T],
        point_ids: &[usize],
        cells: &[usize],
        cell_owners: &[usize],
        cell_ids: &[usize],
        vertex_owners: &[usize],
        vertex_ids: &[usize],
        edges: &[usize],
        edge_owners: &[usize],
        edge_ids: &[usize],
    ) -> ParallelGrid<'a, C, FlatTriangleGrid<T>> {
        let npts = point_ids.len();

        let mut coordinates = rlst_dynamic_array2!(T, [npts, 3]);
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

        let serial_grid = FlatTriangleGrid::new(
            coordinates,
            cells,
            point_ids.to_vec(),
            point_ids_to_indices,
            cell_ids.to_vec(),
            cell_ids_to_indices,
            Some(edge_id_map),
        );

        ParallelGrid::new(
            comm,
            serial_grid,
            vertex_ids,
            vertex_owners,
            edge_ids,
            edge_owners,
            cell_ids,
            cell_owners,
            //vertex_ownership,
            //edge_ownership,
            //cell_ownership,
        )
    }
}
