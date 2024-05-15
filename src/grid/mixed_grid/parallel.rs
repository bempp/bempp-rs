//! Parallel grid builder

use crate::element::reference_cell;
use crate::grid::mixed_grid::{MixedGrid, MixedGridBuilder};
use crate::grid::parallel_grid::ParallelGridBuilder;
use crate::traits::types::ReferenceCellType;
use mpi::{
    request::{LocalScope, WaitGuard},
    topology::Process,
    traits::{Buffer, Destination, Equivalence, Source},
};
use num::Float;
use rlst::{
    dense::array::views::ArrayViewMut, Array, BaseArray, MatrixInverse, RlstScalar, VectorContainer,
};
use std::collections::HashMap;

impl<T: Float + RlstScalar<Real = T> + Equivalence> ParallelGridBuilder for MixedGridBuilder<3, T>
where
    for<'a> Array<T, ArrayViewMut<'a, T, BaseArray<T, VectorContainer<T>, 2>, 2>, 2>: MatrixInverse,
    [T]: Buffer,
{
    type G = MixedGrid<T>;
    type ExtraCellInfo = (Vec<u8>, Vec<usize>);

    fn new_extra_cell_info(&self) -> (Vec<u8>, Vec<usize>) {
        (vec![], vec![])
    }

    fn push_extra_cell_info(&self, extra_cell_info: &mut Self::ExtraCellInfo, cell_id: usize) {
        extra_cell_info
            .0
            .push(self.cell_types[self.cell_ids_to_indices[&cell_id]] as u8);
        extra_cell_info
            .1
            .push(self.cell_degrees[self.cell_ids_to_indices[&cell_id]]);
    }
    fn send_extra_cell_info<'a>(
        &self,
        scope: &LocalScope<'a>,
        process: &Process,
        extra_cell_info: &'a Self::ExtraCellInfo,
    ) {
        let _ = WaitGuard::from(process.immediate_send(scope, &extra_cell_info.0));
        let _ = WaitGuard::from(process.immediate_send(scope, &extra_cell_info.1));
    }
    fn receive_extra_cell_info(
        &self,
        root_process: &Process,
        extra_cell_info: &mut Self::ExtraCellInfo,
    ) {
        let (extra0, _status) = root_process.receive_vec::<u8>();
        for e in extra0 {
            extra_cell_info.0.push(e);
        }
        let (extra1, _status) = root_process.receive_vec::<usize>();
        for e in extra1 {
            extra_cell_info.1.push(e);
        }
    }

    fn point_indices_to_ids(&self) -> &[usize] {
        &self.point_indices_to_ids
    }
    fn points(&self) -> &[T] {
        &self.points
    }
    fn cell_indices_to_ids(&self) -> &[usize] {
        &self.cell_indices_to_ids
    }
    fn cell_points(&self, index: usize) -> &[usize] {
        let cell_start = self
            .cell_types
            .iter()
            .zip(&self.cell_degrees)
            .take(index)
            .map(|(i, j)| self.elements_to_npoints[&(*i, *j)])
            .sum::<usize>();
        let cell_npts =
            self.elements_to_npoints[&(self.cell_types[index], self.cell_degrees[index])];
        &self.cells[cell_start..cell_start + cell_npts]
    }
    fn cell_vertices(&self, index: usize) -> &[usize] {
        let cell_start = self
            .cell_types
            .iter()
            .zip(&self.cell_degrees)
            .take(index)
            .map(|(i, j)| self.elements_to_npoints[&(*i, *j)])
            .sum::<usize>();
        let cell_nvertices = reference_cell::entity_counts(self.cell_types[index])[0];
        &self.cells[cell_start..cell_start + cell_nvertices]
    }
    fn cell_type(&self, index: usize) -> ReferenceCellType {
        self.cell_types[index]
    }
    fn create_serial_grid(
        &self,
        points: Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>,
        cells: &[usize],
        point_indices_to_ids: Vec<usize>,
        point_ids_to_indices: HashMap<usize, usize>,
        cell_indices_to_ids: Vec<usize>,
        _cell_ids_to_indices: HashMap<usize, usize>,
        edge_ids: HashMap<[usize; 2], usize>,
        extra_cell_info: &(Vec<u8>, Vec<usize>),
    ) -> Self::G {
        let cell_types = extra_cell_info
            .0
            .iter()
            .map(|e| ReferenceCellType::from(*e).unwrap())
            .collect::<Vec<_>>();

        MixedGrid::new(
            points,
            cells,
            &cell_types,
            &extra_cell_info.1,
            point_indices_to_ids,
            point_ids_to_indices,
            cell_indices_to_ids,
            Some(edge_ids),
        )
    }
}
